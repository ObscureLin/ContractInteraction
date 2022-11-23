import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NODE_NUM', default=1)
    parser.add_argument('--NUM_CLASSES', default=4)
    parser.add_argument('--BATCH_SIZE', default=64)
    parser.add_argument('--IMAGE_HEIGHT', default=24)
    parser.add_argument('--IMAGE_WIDTH', default=36)
    parser.add_argument('--IMAGE_CHANNEL', default=3)
    args = parser.parse_args()
    return args
