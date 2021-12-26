import os
import subprocess

from utils.parser import get_parser

parser = get_parser()
option = parser.parse_args()

subprocess.run('mkdir -p %s' % option.sources_path, shell = True)
subprocess.run('mkdir -p %s' % option.targets_path, shell = True)

if option.dataset_name == 'text8':

    onweb_path = 'https://data.deepai.org/text8.zip'

    ziped_path = os.path.join(option.sources_path, 'text8.zip')
    unzip_path = os.path.join(option.sources_path, 'text8')

if option.dataset_name == 'ptb':

    onweb_path = 'https://data.deepai.org/ptbdataset.zip'

    ziped_path = os.path.join(option.sources_path, 'ptbdataset.zip')
    unzip_path = os.path.join(option.sources_path, 'ptb')

if not os.path.exists(ziped_path):
    os.system('wget  %s -O %s' % (onweb_path, ziped_path))

if not os.path.exists(unzip_path):
    os.system('unzip %s -d %s' % (ziped_path, unzip_path))
