import argparse
#from ast import arg
#from json import load
import logging
import sys
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

from utils.dataloading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from model import UNet


dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')


def get_args():
    parser = argparse.ArgumentParser(description='Train UNet on images and target masks')
    parser.add_argument('--train-method', '-t', type=str, default='singleGPU', help='Training method')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, 
                        help='Percentage of data used as validation')
    parser.add_argument('--load', '-l', type=str, default=False, help='Load model from a .pth file')

    return parser.parse_args()


def fit(model, device, epochs: int = 10, batch_size: int = 4, learning_rate: float = 1e-4, val_percent: float = 0.1, newsize = [960, 640]):
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Start training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.long)

                pred_masks = model(images)
                loss = criterion(pred_masks, true_masks) + dice_loss(F.softmax(pred_masks, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(), multiclass=True)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()   

                if global_step % 100 == 0:
                    logging.info(f'At step {global_step}, the training loss is {loss.item()}.')

                if global_step % 50 == 0:
                    # Evaluate
                    val_score = evaluate(model, dataloader=val_loader, device=device)
                    logging.info(f'Validation dice score after epoch {epoch+1} is {val_score}')
                    scheduler.step(val_score)


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(filename='./logs/'+args.train_method+'.log', filemode='a', 
                        level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'UNet for Carvana Image Masking (Segmentation)')

    if args.train_method == 'singleGPU':
        device = torch.device('cuda:0')
        logging.info(f'Using {device} for single-GPU training.')

        model = UNet().to(device)
        try:
            fit(model, device=device, epochs=10, batch_size=4, learning_rate=1e-4, val_percent=0.1, newsize=[960, 460])
            torch.save(model.state_dict(), './checkpoints/'+args.train_method+'.pth')
        except:
            torch.save(model.state_dict(), './checkpoints/INTERRUPTED.pth')
            logging.info('Interrupt saved')
            sys.exit(0)