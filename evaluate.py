from functools import reduce
from numpy import dtype
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(model, dataloader, device):
    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # Iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batches', leave=False):
        image, true_mask = batch['image'], batch['mask']
        # Move images and labels to the correct device and type
        image = image.to(device=device, dtype=torch.float32)
        true_mask = true_mask.to(device=device, dtype=torch.long)
        true_mask = F.one_hot(true_mask, model.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # Predict the mask
            pred_mask = model(image)

            # Convert prediction to one-hot
            if model.n_classes == 1:
                pred_mask = (F.sigmoid(pred_mask) > 0.5).float()
                # Compute the dice score
                dice_score += dice_coeff(pred_mask, true_mask, reduce_batch_first=False)
            else:
                pred_mask = F.one_hot(pred_mask.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                # Compute the dice score, ignoring the background
                dice_score += multiclass_dice_coeff(pred_mask[:, 1:, ...], true_mask[:, 1:, ...], reduce_batch_first=False)

    model.train()

    # Fix a potential division by zero error
    if num_val_batches == 0:
        return dice_score

    return dice_score / num_val_batches            