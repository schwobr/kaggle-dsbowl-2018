from dsbowl import train, predict, initialize
from argparse import ArgumentParser

if __name__ == '__main__':
    initialize.run()

    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--predict', help='create a submission using specified model',
        type=str)
    pred = parser.parse_args().predict
    if pred:
        predict.run(pred)
    else:
        train.run()
