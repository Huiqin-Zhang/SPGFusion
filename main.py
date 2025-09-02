# coding: utf-8
import argparse
import os
from train import SPGTrain
import pprint
from hydra import compose, initialize
parser = argparse.ArgumentParser(description='PyTorch Training Example')

parser.add_argument('--epoch', type=int,default=120,help="Number of epoch [10]")
parser.add_argument('--batch_size',type=int,default=4, help="The size of batch images [128]")
parser.add_argument('--checkpoint',type=str, default="CHECKPOINT", help="Name of checkpoint directory [checkpoint]")
parser.add_argument('--savePTH',type=str, default="savePTH", help="savePTH")
parser.add_argument('--summary_dir',type=str, default="log", help="Name of log directory [log]")
parser.add_argument('--config', default = 'clip_dinoiser.yaml', help='config file path')

args = parser.parse_args()
pp = pprint.PrettyPrinter()

def main(args):
    pp.pprint(vars(args))

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
        
    initialize(config_path="configs", version_base=None)            
    cfg = compose(config_name=args.config)
    
    model = SPGTrain(args, cfg)
    model.train()

if __name__ == '__main__':
    main(args)