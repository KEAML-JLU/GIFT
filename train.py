from __future__ import division
from __future__ import print_function
import os
import argparse
from utils import save_res, set_seed
from Trainer import Trainer
import torch
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "model params")
    parser.add_argument("--gpu", type=int, default=0,
                            help="choose which GPU")
    parser.add_argument("--dataset", "-d", type=str, default='twitter',
                            help="choose the dataset: 'snippets' or 'twitter'")
    parser.add_argument("--data_path", "-d_path", type=str, default='./data/',
                            help="choose the data path if necessary")
    parser.add_argument("--save_path", type=str, default="./",
                            help="save path")
    parser.add_argument('--disable_cuda', action='store_true',
                            help='disable CUDA')
    parser.add_argument("--seed", type=int, default=42, 
                            help="seeds for random initial")
    parser.add_argument("--hidden_size", type=int, default=128, 
                            help="hidden size")                        
    parser.add_argument("--lr", type=float, default=1e-3,
                            help="learning rate of the optimizer")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                            help="adjust the learning rate via epochs")
    parser.add_argument("--drop_out", type=float, default=0.9,
                            help="dropout rate")
    parser.add_argument("--max_epoch", type=int, default=100,
                            help="max numer of epochs")
    parser.add_argument("--concat_word_emb", type=bool, default=True,
                            help="concat word embedding with pretrained model")
    
    parser.add_argument('--svd_q', default=15, type=int,
                    help='svd rank')    
    
    # The CL setting
    parser.add_argument('--ucl_temp', type=float, default=0.5, help="temperature required by ucl loss")
    parser.add_argument('--wcl_temp', type=float, default=0.5, help="temperature required by wcl loss")

    parser.add_argument('--alpha', type=float, default=1, help="control unsupervised contrastive learning")
    parser.add_argument('--beta', type=float, default=1, help="control weakly-supervised contrastive learning")
    parser.add_argument('--theta', type=float, default=1, help="control supervised cross entropy")
    
    params = parser.parse_args()
    params.type_num_node = ['query', 'tag', 'word', 'entity']
    params.data_path = params.data_path + './{}_data/'.format(params.dataset)
    params.save_name = params.save_path + './result_torch_{}.json'.format(params.dataset)
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')
 
    print(params.device)
    print(params.dataset)
    set_seed(params.seed)
    trainer = Trainer(params)
    test_acc,best_f1 = trainer.train()
    save_res(params, test_acc, best_f1)
    del trainer