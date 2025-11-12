# train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


import sys
sys.path.append(PROJECT_ROOT)


from config import DEVICE, IMAGE_SIZE, NUM_FG_DESCRIPTORS, NUM_BG_DESCRIPTORS, FEATURE_DIM
from config import NUM_EPOCHS, ITERATIONS_PER_EPOCH, DECAY_RATE, LOSS_WEIGHTS
from config import MODEL_DIR, VIS_DIR, OUTPUT_DIR
from dataset import get_episode_loader
from model import CPCLNet


def calculate_dice(pred, target):
    """Dice = 2 * |P∩G| / (|P| + |G|)"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return (2. * inter) / (union + 1e-8) if union > 0 else 0.0

def intra_inter_loss(fg_desc, bg_desc):

    mean_fg = fg_desc.mean(dim=0, keepdim=True)
    mean_bg = bg_desc.mean(dim=0, keepdim=True)
    intra_fg = F.mse_loss(fg_desc, mean_fg.expand_as(fg_desc))
    intra_bg = F.mse_loss(bg_desc, mean_bg.expand_as(bg_desc))
    loss_intra = (intra_fg + intra_bg) / 2.0

   
    cos = F.cosine_similarity(mean_fg, mean_bg, dim=1)
    loss_inter = 1.0 + cos.squeeze()
    return loss_intra, loss_inter


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = DEVICE
    model = CPCLNet(NUM_FG_DESCRIPTORS, NUM_BG_DESCRIPTORS, FEATURE_DIM).to(device)


    best_pth = os.path.join(MODEL_DIR, "best_model.pth")
    if os.path.exists(best_pth):
        model.load_state_dict(torch.load(best_pth, map_location=device))
        print(f"[Info] Loaded checkpoint: {best_pth}")

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)

    criterion_seg = nn.BCEWithLogitsLoss()

    best_dice = 0.0
    history = {"iter": [], "loss": [], "dice": []}


    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_dices = []
        pbar = tqdm(range(ITERATIONS_PER_EPOCH),
                    desc=f"Epoch {epoch}/{NUM_EPOCHS}",
                    leave=False)

        for _ in pbar:
            optimizer.zero_grad()

            support_imgs, support_masks, query_imgs, query_masks = get_episode_loader()

            pred, fg_desc, bg_desc, q_feat, s_feat, align_loss = model(
                support_imgs.to(device),
                support_masks.to(device),
                query_imgs.to(device)
            )

            query_masks = query_masks.to(device)
            if query_masks.dim() == 3:
                query_masks = query_masks.unsqueeze(0)

            if pred.shape[2:] != query_masks.shape[2:]:
                query_masks = F.interpolate(query_masks,
                                            size=pred.shape[2:],
                                            mode='nearest')

            loss_seg = criterion_seg(pred[:, 1:2], query_masks)


            loss_intra, loss_inter = intra_inter_loss(fg_desc, bg_desc)




            loss = (LOSS_WEIGHTS["lambda_seg"]   * loss_seg +
                    LOSS_WEIGHTS["lambda_intra"] * loss_intra +
                    LOSS_WEIGHTS["lambda_inter"] * loss_inter +
                    LOSS_WEIGHTS["lambda_align"] * align_loss)

            loss.backward()
            optimizer.step()

            dice = calculate_dice(pred[:, 1:2], query_masks)
            epoch_dices.append(dice)

    
            history["iter"].append(len(history["iter"]) + 1)
            history["loss"].append(loss.item())
            history["dice"].append(dice)

            pbar.set_postfix(loss=loss.item(), dice=f"{dice:.4f}")

        scheduler.step()


        valid = [d for d in epoch_dices if d >= 0.5]
        epoch_dice = sum(valid) / len(valid) if valid else 0.0
        print(f"\n>>> Epoch {epoch}  Avg Dice (≥0.5): {epoch_dice:.4f}")


        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(model.state_dict(), best_pth)
            print(f"    [Best] Saved model with Dice {best_dice:.4f}")

 
        model.eval()
        with torch.no_grad():
            s_img, s_mask, q_img, q_mask = get_episode_loader()
            s_img, s_mask, q_img, q_mask = (s_img.to(device), s_mask.to(device),
                                            q_img.to(device), q_mask.to(device))

            pred_v, _, _, _, _, _ = model(s_img, s_mask, q_img)

            pred_mask = torch.sigmoid(pred_v[:, 1, :, :]).cpu().numpy()[0]

   
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
            q_vis = (q_img * std + mean).clamp(0,1).cpu().numpy()[0]
            q_vis = q_vis.transpose(1,2,0)


            gt = q_mask.cpu().numpy()[0,0]
            gt_smooth = binary_dilation(gt, iterations=1)
            gt_smooth = binary_erosion(gt_smooth, iterations=1)

            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1); plt.imshow(q_vis); plt.title("Query Image"); plt.axis('off')
            plt.subplot(1,3,2); plt.imshow(gt_smooth, cmap='gray'); plt.title("GT (smooth)"); plt.axis('off')
            plt.subplot(1,3,3); plt.imshow(pred_mask, cmap='gray'); plt.title("Pred"); plt.axis('off')
            plt.tight_layout()
            viz_path = os.path.join(VIS_DIR, f"epoch_{epoch:03d}.png")
            plt.savefig(viz_path); plt.close()

        model.train()


    df = pd.DataFrame(history)
    csv_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTraining finished. Log saved to {csv_path}")

if __name__ == "__main__":

    train()
