import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = '', help = '')

    # For Loader
    parser.add_argument('--dataset_path', default = 'dataset', help = '')
    parser.add_argument('--dataset_name', default = 'ptb', choices = ['text8', 'ptb'], help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
