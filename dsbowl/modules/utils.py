import numpy as np
import cv2
import os
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


def getNextFilePath(output_folder):
    highest_num = 0
    for f in os.listdir(output_folder):
        if os.path.isfile(os.path.join(output_folder, f)):
            file_name = os.path.splitext(f)[0]
            try:
                split = file_name.split('.')
                split = split[0].split('_')
                file_num = int(split[-1])
                if file_num > highest_num:
                    highest_num = file_num
            except ValueError:
                'The file name "%s" is not an integer. Skipping' % file_name

    output_file = highest_num + 1
    return output_file
