import numpy as np
import torch
from torch import optim
from torch.nn import Module
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
import tqdm
import logging
from pathlib import Path
from dataloading import CarvanaDataset, BasicDataset
from evaluate import evaluate

dir_img = Path('./data/train_hq/')
dir_mask = Path('./data/train_masks/')

def fit_1GPU(model: Module, criterion, epochs: int = 10, batch_size: int = 4, 
                learning_rate: float = 1e-4, val_percent: float = 0.1, newsize = [960, 640]):

    model.cuda()

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
                images = batch['image'].cuda().dtype(torch.float32)
                true_masks = batch['mask'].cuda().dtype(torch.float32)

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
                    val_loss = evaluate(model, dataloader=val_loader, criterion=criterion)
                    val_losses.append([global_step, val_loss])
                    #print(f' At step {global_step}, the validation loss is {val_loss}.')
                    scheduler.step(val_loss)

        # Evaluate after each epoch
        val_loss = evaluate(model, dataloader=val_loader)
        val_losses.append([global_step, val_loss])
        #print(f' After {epoch} epoch, the validation loss is {val_loss}.')
        scheduler.step(val_loss)

    return model, train_losses, val_losses


def fit_DP(model: Module, criterion, epochs: int = 10, batch_size: int = 4, 
            learning_rate: float = 1e-4, val_percent: float = 0.1, newsize = [960, 640]):
    
    model = DP(model, device_ids=[0, 1])

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
                images = batch['image'].cuda().dtype(torch.float32)
                true_masks = batch['mask'].cuda().dtype(torch.float32)

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
                    val_loss = evaluate(model, dataloader=val_loader, criterion=criterion)
                    val_losses.append([global_step, val_loss])
                    #print(f' At step {global_step}, the validation loss is {val_loss}.')
                    scheduler.step(val_loss)

        # Evaluate after each epoch
        val_loss = evaluate(model, dataloader=val_loader)
        val_losses.append([global_step, val_loss])
        #print(f' After {epoch} epoch, the validation loss is {val_loss}.')
        scheduler.step(val_loss)