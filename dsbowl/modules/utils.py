import numpy as np
import cv2
from skimage.morphology import label


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    mask = np.zeros_like(lab_img)
    for i in range(1, lab_img.max() + 1):
        img = ((lab_img == i).astype(np.uint8) * 255)
        img = cv2.dilate(img, np.ones((3, 3), np.uint8),
                         iterations=1).astype('float32')/255
        mask[img > cutoff] = i
    for i in range(1, mask.max() + 1):
        yield rle_encoding(mask == i)
