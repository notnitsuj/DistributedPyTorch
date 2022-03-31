import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from utils.dataloading import BasicDataset, CarvanaDataset
from utils.utils import Loss
from evaluate import evaluate
from model import UNet


dir_img = Path('./data/train_hq/')
dir_mask = Path('./data/train_masks/')


def get_args():
    parser = argparse.ArgumentParser(description='Train UNet on images and target masks')
    parser.add_argument('--train-method', '-t', type=str, default='singleGPU', help='Training method')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, 
                        help='Percentage of data used as validation')
    parser.add_argument('--load', '-l', type=str, default=False, help='Load model from a .pth file')

    return parser.parse_args()


def fit(model, device, criterion, epochs: int = 10, batch_size: int = 4, learning_rate: float = 1e-4, 
        val_percent: float = 0.1, newsize = [960, 640]):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, newsize)
        logging.info(f'Carvana Dataset')
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, newsize)
        logging.info(f'Basic Dataset')

    # 2. Split into train/validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loader
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
    global_step = 0

    # 5. Start training
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        losses = []
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            mean_loss = 0
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)

                pred_masks = model(images)
                loss = criterion(pred_masks, true_masks) 
                
                optimizer.zero_grad()
                losses.append(loss.item())       

                (batch_size * loss).backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1

                if global_step % 10 == 0:
                    mean_loss = np.mean(losses[-10:])
                    #print(f' At step {global_step}, the training loss is {mean_loss}.')
                    train_losses.append([global_step, mean_loss])

                if global_step % 200 == 0:
                    # Evaluate regularly to prevent overfitting
                    val_loss = evaluate(model, dataloader=val_loader, criterion=criterion, device=device)
                    val_losses.append([global_step, val_loss])
                    #print(f' At step {global_step}, the validation loss is {val_loss}.')
                    scheduler.step(val_loss)

        # Evaluate after each epoch
        val_loss = evaluate(model, dataloader=val_loader, device=device)
        val_losses.append([global_step, val_loss])
        #print(f' After {epoch} epoch, the validation loss is {val_loss}.')
        scheduler.step(val_loss)

    return model, train_losses, val_losses

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(filename='./logs/'+args.train_method+'.log', filemode='a', 
                        level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'UNet for Carvana Image Masking (Segmentation)')

    if args.train_method == 'singleGPU':
        device = torch.device('cuda')
        logging.info(f'Using {device} for single-GPU training.')

        model = UNet().to(device)
        criterion = Loss()

        try:
            model, train_losses, val_losses = fit(model, device=device, criterion=criterion, epochs=10, batch_size=4, 
                                                    learning_rate=1e-4, val_percent=0.1, newsize=[960, 640])
            torch.save(model.state_dict(), './checkpoints/'+args.train_method+'.pth')
            train_losses_df = pd.DataFrame(train_losses, columns=['Step', 'Loss'])
            train_losses_df.to_pickle('./loss/'+args.train_method+'/train_loss.pkl')
            val_losses_df = pd.DataFrame(val_losses, columns=['Step', 'Loss'])
            val_losses_df.to_pickle('./loss/'+args.train_method+'/val_loss.pkl')
        except:
            torch.save(model.state_dict(), './checkpoints/'+args.train_method+'_INTERRUPTED.pth')
            logging.info('Interrupt saved')
            sys.exit(0)