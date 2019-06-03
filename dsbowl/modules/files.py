import os


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
