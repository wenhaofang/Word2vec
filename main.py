import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tqdm
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

seed = 77

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils.parser import get_parser
from utils.logger import get_logger

parser = get_parser()
option = parser.parse_args()

if  option.name == '':
    option.name = option.module_type + '_' + option.dataset_name

root_folder = os.path.join('result', option.name)

subprocess.run('mkdir -p %s' % root_folder, shell = True)

logger = get_logger(option.name, os.path.join(root_folder, 'main.log'))

from loaders.CBOW_HS_Loader import get_loader as get_CBOW_HS_Loader
from loaders.CBOW_NS_Loader import get_loader as get_CBOW_NS_Loader

from loaders.SG_HS_Loader import get_loader as get_SG_HS_Loader
from loaders.SG_NS_Loader import get_loader as get_SG_NS_Loader

from modules.CBOW_HS_Module import get_module as get_CBOW_HS_Module
from modules.CBOW_NS_Module import get_module as get_CBOW_NS_Module

from modules.SG_HS_Module import get_module as get_SG_HS_Module
from modules.SG_NS_Module import get_module as get_SG_NS_Module

from utils.misc import save_embedding

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader & module')

if  option.module_type == 'CBOW_HS':
    loader = get_CBOW_HS_Loader(option)
    module = get_CBOW_HS_Module(option, loader.get_vocab_size(), device).to(device)
if  option.module_type == 'CBOW_NS':
    loader = get_CBOW_NS_Loader(option)
    module = get_CBOW_NS_Module(option, loader.get_vocab_size(), device).to(device)
if  option.module_type == 'SG_HS':
    loader = get_SG_HS_Loader(option)
    module = get_SG_HS_Module(option, loader.get_vocab_size(), device).to(device)
if  option.module_type == 'SG_NS':
    loader = get_SG_NS_Loader(option)
    module = get_SG_NS_Module(option, loader.get_vocab_size(), device).to(device)

logger.info('prepare envs')

optimizer = optim.Adam(module.parameters(), lr = option.learning_rate)

logger.info('start train')

patience = 5
no_progress_num = 0
best_epoch_loss = float('inf')
for epoch in range(option.num_epochs):
    loader.reset()
    module.train()

    epoch_loss = 0
    for mini_batch in tqdm.tqdm(loader):
        batch_loss = module(*mini_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss

    logger.info(
        'epoch: %d, epoch_loss: %.7f, best_epoch_loss: %.7f' %
        (epoch, epoch_loss, best_epoch_loss)
    )

    if  best_epoch_loss > epoch_loss:
        best_epoch_loss = epoch_loss
        no_progress_num = 0
        save_embedding(loader, module, os.path.join(root_folder, 'best.ckpt'))
    else:
        no_progress_num += 1

    if  no_progress_num > patience:
        break
