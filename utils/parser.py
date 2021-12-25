import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = '', help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
