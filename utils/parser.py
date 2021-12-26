import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = '', help = '')

    # For Loader
    parser.add_argument('--sources_path', default = 'datasources', help = '')
    parser.add_argument('--targets_path', default = 'datatargets', help = '')
    parser.add_argument('--dataset_name', default = 'ptb', choices = ['text8', 'ptb'], help = '')

    parser.add_argument('--min_freq', type = int, default = 5, help = '')
    parser.add_argument('--max_numb', type = int, default = 1000000, help = '')

    parser.add_argument('--max_window_size', type = int, default = 5, help = '')
    parser.add_argument('--neg_sample_size', type = int, default = 5, help = '')

    parser.add_argument('--use_cache', action = 'store_true', help = '')

    # For Module
    parser.add_argument('--module_type', default = 'SG_NS', choices = ['CBOW_HS', 'CBOW_NS', 'SG_HS', 'SG_NS'], help = '')

    parser.add_argument('--emb_dim', type = int, default = 100, help = '')

    # For Train
    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--num_epochs', type = int, default = 70, help = '')

    parser.add_argument('--learning_rate', type = float, default = 0.01, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
