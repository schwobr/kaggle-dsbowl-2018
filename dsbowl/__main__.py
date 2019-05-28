from dsbowl import train, predict
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--predict', help='create a submission using specified model',
        type=str)
    pred = parser.parse_args.predict
    if pred:
        predict.run(pred)
    else:
        train.run()
