import torch
import matplotlib.pyplot as plt
import numpy as np
import config

def compute_dice_score(pred_mask, gt_mask):
    pred_mask = (pred_mask > 0.6).float() 
    gt_mask = gt_mask.squeeze(1).float()
    intersection = (pred_mask * gt_mask).sum()
    dice = (2. * intersection + 1e-8) / (pred_mask.sum() + gt_mask.sum() + 1e-8)
    return dice.item()

def visualize_prediction(query_img, pred_mask, gt_mask, filename):
    query_img = query_img.to(config.DEVICE)
    pred_mask = pred_mask.to(config.DEVICE)
    gt_mask = gt_mask.to(config.DEVICE)

    mean = torch.tensor([0.485, 0.456, 0.406], device=config.DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225], device=config.DEVICE)
    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)
    img = query_img * std + mean
    img = img.clamp(0, 1)

    img = img.permute(1, 2, 0).cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img)
    ax1.set_title("Query Image")
    ax2.imshow(pred_mask, cmap='gray')
    ax2.set_title("Predicted Mask")
    ax3.imshow(gt_mask, cmap='gray')
    ax3.set_title("Ground Truth Mask")
    plt.savefig(f"{config.VIS_DIR}/{filename}")

    plt.close()
