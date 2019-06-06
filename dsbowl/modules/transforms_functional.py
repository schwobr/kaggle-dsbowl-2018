import cv2
import numpy as np
import math
from torchvision.transforms.functional import to_tensor


def vflip(img):
    return cv2.flip(img, 0)


def hflip(img):
    return cv2.flip(img, 1)


def random_flip(img, code):
    return cv2.flip(img, code)


def transpose(img):
    return img.transpose(
        1, 0, 2) if len(
        img.shape) > 2 else img.transpose(
        1, 0)


def rot90(img, factor):
    img = np.rot90(img, factor)
    return np.ascontiguousarray(img)


def rotate(img, angle):
    height, width = img.shape[0:2]
    mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    img = cv2.warpAffine(img, mat, (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return img


def shift_scale_rotate(img, angle, scale, dx, dy):
    height, width = img.shape[:2]

    cc = math.cos(angle/180*math.pi) * scale
    ss = math.sin(angle/180*math.pi) * scale
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0],  [width, height], [0, height], ])
    box1 = box0 - np.array([width/2, height/2])
    box1 = np.dot(
        box1, rotate_matrix.T) + np.array([width/2+dx*width,
                                           height/2+dy*height])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    img = cv2.warpPerspective(img, mat, (width, height),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    return img


def crop(img, height, width, dx, dy):
    h, w, c = img.shape
    xmax = max(0, h-height)
    ymax = max(0, w-width)
    x1 = math.ceil(dx*xmax)
    y1 = math.ceil(dy*ymax)
    return img[x1:x1+height, y1:y1+width, :]


def center_crop(img, height, width):
    h, w, c = img.shape
    dy = (h-height)//2
    dx = (w-width)//2
    return img[dx:dx+height, dy:dy+width, :]


def to_three_channel_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    invgray = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    if np.mean(invgray) < np.mean(gray):
        invgray, gray = gray, invgray
    res = [invgray, gray, clahe.apply(invgray)]
    return cv2.merge(res)


def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) > 127:
        gray = 255 - gray
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def add_channel(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(21, 21))
    lab = clahe.apply(lab[:, :, 0])
    if lab.mean() > 127:
        lab = 255 - lab
    return np.dstack((img, lab))


def fix_mask(msk):
    msk = (msk > 127)
    return msk.astype(np.uint8) * 255


def img_to_tensor(im):
    return to_tensor(im)
