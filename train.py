import argparse
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
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, 
                        help='Percentage of data used as validation')
    parser.add_argument('--load', '-l', type=str, default=False, help='Load model from a .pth file')

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
            mp.spawn(fit_DDP, args=(2, 'nccl', model, criterion, epochs, batch_size, learning_rate, val_percent, newsize),
                        nprocs=2, join=True)
        
        torch.save(model.state_dict(), './checkpoints/'+args.train_method+'.pth')
        train_losses_df = pd.DataFrame(train_losses, columns=['Step', 'Loss'])
        train_losses_df.to_pickle('./loss/'+args.train_method+'/train_loss.pkl')
        val_losses_df = pd.DataFrame(val_losses, columns=['Step', 'Loss'])
        val_losses_df.to_pickle('./loss/'+args.train_method+'/val_loss.pkl')
    except:
        torch.save(model.state_dict(), './checkpoints/'+args.train_method+'_INTERRUPTED.pth')
        logging.info('Interrupt saved')
        sys.exit(0)


if __name__ == '__main__':
    args = get_args()
    set_seed(42)

    logging.basicConfig(filename='./logs/'+args.train_method+'.log', filemode='a', 
                        level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'UNet for Carvana Image Masking (Segmentation)')
    
    model = UNet()
    criterion = Loss()

    if args.train_method == 'singleGPU':
        device = torch.device('cuda')
        logging.info(f'Using {device} for single-GPU training.')
        train(model, args.train_method, criterion)

    else:
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

        train(model, args.train_method, criterion)