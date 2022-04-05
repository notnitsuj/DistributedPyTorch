import argparse
from ctypes.wintypes import WORD
import logging
import warnings
import sys
import pandas as pd

import torch
import torch.multiprocessing as mp

from model import UNet
from utils.utils import Loss, set_seed
from utils.train_utils import fit_1GPU, fit_DP, fit_DDP

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='Train UNet on images and target masks')
    parser.add_argument('--train-method', '-t', type=str, default='singleGPU', help='Training method')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percentage of data used as validation')
    parser.add_argument('--load', '-l', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--checkpoint', '-c', type=str, default=None, help='File name of the checkpoint to load')

    return parser.parse_args()


def train(model, train_method: str, criterion, epochs=10, batch_size=4, 
            learning_rate=1e-4, val_percent=0.1, newsize=[960, 640]):      
    try:
        if train_method == 'singleGPU':
            model, train_losses, val_losses = fit_1GPU(model, criterion=criterion, epochs=epochs, batch_size=batch_size, 
                                                        learning_rate=learning_rate, val_percent=val_percent, newsize=newsize)
        elif train_method == 'DP':
            model, train_losses, val_losses = fit_DP(model, criterion=criterion, epochs=epochs, batch_size=batch_size, 
                                                        learning_rate=learning_rate, val_percent=val_percent, newsize=newsize)
        elif train_method == 'DDP':
            mp.spawn(fit_DDP, args=(2, 'nccl', model, criterion, epochs, batch_size, learning_rate, val_percent, newsize,),
                        nprocs=2, join=True)
        
        torch.save(model.state_dict(), f"./checkpoints/{args.train_method}.pth")
        train_losses_df = pd.DataFrame(train_losses, columns=['Step', 'Loss'])
        train_losses_df.to_pickle(f"./loss/{args.train_method}/train_loss.pkl")
        val_losses_df = pd.DataFrame(val_losses, columns=['Step', 'Loss'])
        val_losses_df.to_pickle(f"./loss/{args.train_method}/val_loss.pkl")
    except:
        torch.save(model.state_dict(), f"./checkpoints/{args.train_method}_INTERRUPTED.pth")
        logging.info('Interrupt saved')
        sys.exit(0)

WORLD_SIZE = torch.cuda.device_count()
BACKEND = 'nccl'

if __name__ == '__main__':
    args = get_args()
    set_seed(42)

    logging.basicConfig(filename=f"./logs/{args.train_method}.log", filemode='a', 
                        level=logging.INFO, format='%(message)s')
    logging.info(f"Net for Carvana Image Masking (Segmentation)")
    
    model = UNet()
    criterion = Loss()

    if args.train_method == 'singleGPU':
        device = torch.device('cuda')
        logging.info(f"Using {device} for single-GPU training.")
        fit_1GPU(model=model, criterion=criterion, epochs=args.epochs, batch_size=args.batch_size, 
                    learning_rate=args.lr, val_percent=args.validation)

    else:
        assert WORLD_SIZE >= 2, f"Requires at least 2 GPUs to run, but got {WORLD_SIZE}"

        mp.spawn(fit_DDP, args=(WORLD_SIZE, BACKEND, model, criterion, args.epochs, args.batch_size, args.lr, args.validation,),
                        nprocs=WORLD_SIZE, join=True)