import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm
import logging
from pathlib import Path

import torch
from torch import optim
from torch.nn import Module
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader, random_split
from utils.dataloading import CarvanaDataset, BasicDataset
from evaluate import evaluate

dir_img = Path('./data/train_hq/')
dir_mask = Path('./data/train_masks/')

def fit_1GPU(model: Module, criterion, epochs: int = 10, batch_size: int = 4, 
                learning_rate: float = 1e-4, val_percent: float = 10.0):

    model.cuda(0)

    # 1. Create dataset
    newsize = [960, 640]
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, newsize)
        logging.info(f'Carvana Dataset')
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, newsize)
        logging.info(f'Basic Dataset')

    # 2. Split into train/validation partitions
    n_val = int(len(dataset) * val_percent/100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loader
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer and the learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
    global_step = 0

    # 5. Start training
    train_losses = []
    val_losses = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            mean_loss = 0
            for batch in train_loader:
                images = batch['image'].cuda(0).to(torch.float32)
                true_masks = batch['mask'].cuda(0).to(torch.float32).unsqueeze(1)

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
                    train_time = time.time() - start_time
                    train_losses.append([global_step, train_time, mean_loss])

        # Evaluate after each epoch
        val_loss = evaluate(model, dataloader=val_loader, criterion=criterion)
        val_time = time.time() - start_time
        val_losses.append([global_step, val_time, val_loss])
        #print(f' After {epoch} epoch, the validation loss is {val_loss}.')
        scheduler.step(val_loss)

    torch.save(model.state_dict(), f"./checkpoints/singleGPU.pth")
    train_losses_df = pd.DataFrame(train_losses, columns=['Step', 'Time', 'Loss'])
    train_losses_df.to_pickle(f"./loss/singleGPU/train_loss.pkl")
    val_losses_df = pd.DataFrame(val_losses, columns=['Step', 'Time', 'Loss'])
    val_losses_df.to_pickle(f"./loss/singleGPU/val_loss.pkl")


def fit_DP(model: Module, criterion, epochs: int = 10, batch_size: int = 4, 
            learning_rate: float = 1e-4, val_percent: float = 0.1, device_ids: list = [0, 1]):
    
    model = DP(model, device_ids=device_ids)

    # 1. Create dataset
    newsize = [960, 640]
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, newsize)
        logging.info(f'Carvana Dataset')
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, newsize)
        logging.info(f'Basic Dataset')

    # 2. Split into train/validation partitions
    n_val = int(len(dataset) * val_percent/100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loader
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer and the learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
    global_step = 0

    # 5. Start training
    train_losses = []
    val_losses = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            mean_loss = 0
            for batch in train_loader:
                images = batch['image'].cuda().to(torch.float32)
                true_masks = batch['mask'].cuda().to(torch.float32)

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
                    train_time = time.time() - start_time
                    train_losses.append([global_step, train_time, mean_loss])

        # Evaluate after each epoch
        val_loss = evaluate(model, dataloader=val_loader, criterion=criterion)
        val_time = time.time() - start_time
        val_losses.append([global_step, val_time, val_loss])
        #print(f' After {epoch} epoch, the validation loss is {val_loss}.')
        scheduler.step(val_loss)

    torch.save(model.state_dict(), f"./checkpoints/DP.pth")
    train_losses_df = pd.DataFrame(train_losses, columns=['Step', 'Time', 'Loss'])
    train_losses_df.to_pickle(f"./loss/DP/train_loss.pkl")
    val_losses_df = pd.DataFrame(val_losses, columns=['Step', 'Time', 'Loss'])
    val_losses_df.to_pickle(f"./loss/DP/val_loss.pkl")


def fit_DDP(world_size, model: Module, criterion, epochs: int = 10, batch_size: int = 4, 
            learning_rate: float = 1e-4, val_percent: float = 0.1):
    rank = int(os.environ["LOCAL_RANK"])
    #if rank == 0:
    # Create dataset
    newsize = [960, 640]
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, newsize)
        logging.info(f'Carvana Dataset')
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, newsize)
        logging.info(f'Basic Dataset')

    # 2. Split into train/validation partitions
    n_val = int(len(dataset) * val_percent/100)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator())

    # Data Loader
    train_sampler = DistributedSampler(train_set)#, rank=rank, num_replicas=world_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
    #dist.barrier()

    # Send model to the correspoding GPU
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Set up the optimizer and the learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate*world_size, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
    global_step = 0

    # Start training
    train_losses = []
    val_losses = []
    if rank == 0:
        start_time = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []

        for batch in train_loader:
            images = batch['image'].to(rank).to(torch.float32)
            true_masks = batch['mask'].to(rank).to(torch.float32).unsqueeze(1)

            pred_masks = model(images)
            loss = criterion(pred_masks, true_masks)
            
            optimizer.zero_grad()
            losses.append(loss.item())
            if global_step % 20 == 0:
                print('At step {} in rank {}, the training loss is {}'.format(global_step, rank, loss.item()))

            (batch_size * loss).backward()
            optimizer.step()

            global_step += 1

            if global_step % 10 == 0 and rank == 0:
                mean_loss = np.mean(losses[-10:])
                #print(f' At step {global_step}, the training loss is {mean_loss}.')
                train_time = time.time() - start_time
                train_losses.append([global_step, train_time, mean_loss])

        if rank == 0:
            # Evaluate after each epoch
            val_loss = evaluate(model, dataloader=val_loader, criterion=criterion)
            val_time = time.time() - start_time
            val_losses.append([global_step, val_time, val_loss])
            #print(f' After {epoch} epoch, the validation loss is {val_loss}.')
            scheduler.step(val_loss)
    
    if rank == 0:
        torch.save(model.state_dict(), f"./checkpoints/DDP.pth")
        train_losses_df = pd.DataFrame(train_losses, columns=['Step', 'Time', 'Loss'])
        train_losses_df.to_pickle(f"./loss/DDP/train_loss.pkl")
        val_losses_df = pd.DataFrame(val_losses, columns=['Step', 'Time', 'Loss'])
        val_losses_df.to_pickle(f"./loss/DDP/val_loss.pkl")