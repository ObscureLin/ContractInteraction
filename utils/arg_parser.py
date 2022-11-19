import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NODE_NUM', default=1)
    args = parser.parse_args()
    return args
