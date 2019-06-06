import random
import modules.transforms_functional as F
import numpy as np


class Compose:
    def __init__(self, transforms, prob=1.0):
        self.transforms = [t for t in transforms if t is not None]
        self.prob = prob

    def __call__(self, **data):
        if random.random() < self.prob:
            for t in self.transforms:
                data = t(**data)
        return data


class OneOf:
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, **data):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            data = t(**data)
        return data


class Transform:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, **kwargs):
        if random.random() < self.prob:
            params = self.get_params()
            return {k: self.apply(a, **params) if k in self.targets else a
                    for k, a in kwargs.items()}
        return kwargs

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        raise NotImplementedError


class ImageOnlyTransform(Transform):
    @property
    def targets(self):
        return 'image'


class DualTransform(Transform):
    @property
    def targets(self):
        return 'mask', 'image'


class VFlip(DualTransform):
    def apply(self, img, **params):
        return F.vflip(img)


class HFlip(DualTransform):
    def apply(self, img, **params):
        return F.hflip(img)


class RandomFlip(DualTransform):
    def apply(self, img, d=0):
        return F.random_flip(img, d)

    def get_params(self):
        return {'d': random.randint(-1, 1)}


class Transpose(DualTransform):
    def apply(self, img, **params):
        return F.transpose(img)


class RandomRotate90(DualTransform):
    def apply(self, img, factor=0):
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        return {'factor': random.randint(0, 4)}


class Rotate(DualTransform):
    def __init__(self, limit=90, prob=.5):
        super().__init__(prob)
        self.limit = limit

    def apply(self, img, angle=0):
        return F.rotate(img, angle)

    def get_params(self):
        return {'angle': random.uniform(-self.limit, self.limit)}


class ShiftScaleRotate(DualTransform):
    def __init__(
            self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45,
            prob=0.5):
        super().__init__(prob)
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit

    def apply(self, img, angle=0, scale=0, dx=0, dy=0):
        return F.shift_scale_rotate(img, angle, scale, dx, dy)

    def get_params(self):
        return {'angle': random.uniform(-self.rotate_limit,
                                        self.rotate_limit),
                'scale': random.uniform(1-self.scale_limit,
                                        1+self.scale_limit),
                'dx': round(random.uniform(-self.shift_limit,
                                           self.shift_limit)),
                'dy': round(random.uniform(-self.shift_limit,
                                           self.shift_limit))}


class CenterCrop(DualTransform):
    def __init__(self, height, width, prob=0.5):
        super().__init__(prob)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        return F.center_crop(img, self.height, self.width)


class RandomCrop(DualTransform):
    def __init__(self, height, width, prob=0.5):
        super().__init__(prob)
        self.height = height
        self.width = width

    def apply(self, img, dx=0, dy=0):
        return F.crop(img, self.height, self.width, dx, dy)

    def get_params(self):
        return {'dx': random.random(), 'dy': random.random()}


class ToThreeChannelGray(ImageOnlyTransform):
    def __init__(self, prob=1.):
        super().__init__(prob)

    def apply(self, img, **params):
        return F.to_three_channel_gray(img)


class ToGray(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.to_gray(img)


class AddChannel(ImageOnlyTransform):
    def __init__(self, prob=1.):
        super().__init__(prob)

    def apply(self, img, **params):
        return F.add_channel(img)


class FixMask(Transform):
    def __init__(self, prob=1.):
        super().__init__(prob)

    def apply(self, img, **params):
        return F.fix_mask(img)

    @property
    def targets(self):
        return ('mask',)


class ToTensor(DualTransform):
    def __init__(self):
        super().__init__(1.)

    def apply(self, img, **params):
        return F.to_tensor(img)
