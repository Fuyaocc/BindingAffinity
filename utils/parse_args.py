import argparse
import os
import sys
import logging

def get_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"                   
    parser = argparse.ArgumentParser(description=description)    
    parser.add_argument("--outdir",default="./tmp/",help="Output directory, defaults to tmp")
    parser.add_argument('--device', type=str, default='cuda:1',help='set device')
    parser.add_argument('--inputdir',type=str,default="./data/",help='input data dictionary')
    parser.add_argument('--foldxdir',type=str,default="../foldx/",help='foldx result path')
    parser.add_argument('--dim',type=int,default=256,help='model input dim')
    parser.add_argument('--epoch',type=int,default=3000,help='trainning epoch')
    parser.add_argument('--batch_size',type=int,default=128,help='batch size ')
    parser.add_argument('--epsilon',type=float,default=0.6,help='adv_sample epsilon')
    parser.add_argument('--alpha',type=float,default=0.5,help='the weight in loss ')
    parser.add_argument('--padding',type=int,default=180,help='make feature same length')
    parser.add_argument('--interfacedis',type=float,default=12.0,help='resdisues distance in protein')
    parser.add_argument('--logdir',type=str,default='./log/val',help='log dir,defaults to log')
    parser.add_argument('--datadir',type=str,default='/mnt/data/xukeyu/PPA_Pred/',help='pdb dir')
    parser.add_argument('--featdir',type=str,default="/mnt/data/xukeyu/PPA_Pred/",help='complex graph node feat and edge feat dir')
    parser.add_argument('--modeldir',type=str,default="./models/saved/",help='models dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)