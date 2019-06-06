import torch
import numpy as np
from skimage.morphology import label


def mean_iou(y_pred, y_true, smooth=1e-6):
    scores = np.zeros(y_true.shape[0])
    y_true = y_true.cpu().squeeze(1)
    y_pred = torch.sigmoid(y_pred.cpu()).squeeze(1)
    for i in range(y_true.shape[0]):
        labels_pred = label(y_pred.to('cpu').numpy()[i] > 0.5)
        labels_true = label(y_true.to('cpu').numpy()[i])
        score = 0
        cnt = 0
        n_masks_pred = np.max(labels_pred)
        n_masks_true = np.max(labels_true)
        inter_union = np.zeros((n_masks_pred, n_masks_true, 2), dtype=np.int)
        for k in range(y_true.shape[1]):
            for l in range(y_true.shape[2]):
                m = labels_pred[k, l]
                n = labels_true[k, l]
                if m != 0:
                    inter_union[m-1, :, 1] += 1
                if n != 0:
                    inter_union[:, n-1, 1] += 1
                if m != 0 and n != 0:
                    inter_union[m-1, n-1, 0] += 1
        ious = inter_union[:, :, 0]/(
            inter_union[:, :, 1]-inter_union[:, :, 0]+smooth)
        for t in np.arange(0.5, 1.0, 0.05):
            cnt += 1
            tp = 0
            fp = 0
            fn = 0
            fn_tests = np.ones(n_masks_true, dtype=np.bool)
            for m in range(n_masks_pred):
                fp_test = True
                for n in range(n_masks_true):
                    if ious[m, n] > t:
                        tp += 1
                        fp_test = False
                        fn_tests[n] = False
                if fp_test:
                    fp += 1
            fn = np.count_nonzero(fn_tests)
            try:
                score += tp/(tp+fp+fn)
            except ZeroDivisionError:
                pass
        score = score/cnt
        scores[i] = score
    return torch.tensor(scores).mean()
