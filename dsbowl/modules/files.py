import os
import pandas as pd
import cv2
import numpy as np


def getNextId(output_folder):
    highest_num = -1
    for d in os.listdir(output_folder):
        dir_name = os.path.splitext(d)[0]
        try:
            i = int(dir_name)
            if i > highest_num:
                highest_num = i
        except ValueError:
            'The dir name "%s" is not an integer. Skipping' % dir_name

    new_id = highest_num + 1
    return new_id


def getNextFilePath(output_folder, base_name):
    highest_num = 0
    for f in os.listdir(output_folder):
        if os.path.isfile(output_folder / f):
            try:
                if f.split('_')[:-1] == base_name.split('_'):
                    split = f.split('_')
                    file_num = int(split[-1])
                    if file_num > highest_num:
                        highest_num = file_num
            except ValueError:
                'The file name "%s" is incorrect. Skipping' % f

    output_file = highest_num + 1
    return output_file


def create_csv(dir_path, save_path):
    name = dir_path.name
    df = pd.DataFrame(columns=['ImageId', 'Path',
                               'Height', 'Width', 'Channels'])
    print(f'Creating csv for {dir_path}...')
    for k, i in enumerate(next(os.walk(dir_path))[1]):
        path = dir_path / str(i) / 'images' / f'{i}.png'
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        try:
            h, w, c = img.shape
        except ValueError:
            h, w = img.shape
            c = 1
        df.loc[k] = [i, str(path), h, w, c]
    df.to_csv(save_path / f'{name}.csv')


def get_sizes(file, ids):
    df = pd.read_csv(file, index_col=0)
    sizes = np.zeros((len(ids), 2))
    for k, i in enumerate(ids):
        sizes[k, :] = df.loc[df['ImageId'] == i, 'Height':'Width'].values
    return sizes.astype('int')
