import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = '', help = '')

    # For Loader
    parser.add_argument('--dataset_path', default = 'dataset', help = '')
    parser.add_argument('--dataset_name', default = 'ptb', choices = ['text8', 'ptb'], help = '')

    parser.add_argument('--min_freq', type = int, default = 5, help = '')
    parser.add_argument('--max_numb', type = int, default = 1000000, help = '')

    parser.add_argument('--max_window_size', type = int, default = 5, help = '')
    parser.add_argument('--neg_sample_size', type = int, default = 5, help = '')

    # For Module
    parser.add_argument('--emb_dim', type = int, default = 100, help = '')

    # For Train
    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--num_epochs', type = int, default = 10, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
