import numpy as np
import torch
from tqdm import tqdm


def evaluate(model, dataloader, criterion):
    model.eval()
    num_val_batches = len(dataloader)
    losses = []

    # Iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch'):
        images = batch['image'].cuda().to(torch.float32)
        true_masks = batch['mask'].cuda().to(torch.float32).unsqueeze(1)

        with torch.no_grad():
          pred_masks = model(images)
          loss = criterion(pred_masks, true_masks)
          losses.append(loss.item())
    
    valid_loss = np.mean(losses)  # type: float

    model.train()

    return valid_loss      