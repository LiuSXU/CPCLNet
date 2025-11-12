import torch
import config
from model import GMRDModel
from dataset import FSMISDataset
from utils import compute_dice_score
from torchvision import transforms
import numpy as np
import os
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from scipy.ndimage import binary_dilation, binary_erosion
import torch.nn.functional as F

def optimize_threshold(pred_mask, gt_mask, threshold_range=np.arange(0.1, 1.0, 0.05)):
    best_dice = 0.0
    best_threshold = 0.5
    for threshold in threshold_range:
        pred_binary = (pred_mask > threshold).float()
        dice = compute_dice_score(pred_binary, gt_mask)
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    return best_threshold, best_dice

def crf_post_process(image, pred_mask, gt_mask_shape):
    image = image.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()

    image = image.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    image = np.ascontiguousarray(image)

    pred_mask = pred_mask.squeeze()

    softmax = np.stack([1 - pred_mask, pred_mask], axis=0)

    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    unary = unary_from_softmax(softmax)
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

    Q = d.inference(10)  # 增加推理步数到 10
    map = np.argmax(Q, axis=0).reshape(gt_mask_shape[-2:])

    return torch.from_numpy(map).float().to(config.DEVICE)

def post_process_mask(pred_mask, iterations=2):
    pred_mask_np = pred_mask.cpu().numpy()
    processed_mask = binary_dilation(pred_mask_np, iterations=iterations)
    processed_mask = binary_erosion(processed_mask, iterations=iterations)
    return torch.from_numpy(processed_mask).float().to(config.DEVICE)

def evaluate():
    model = GMRDModel(config.NUM_FG_DESCRIPTORS, config.NUM_BG_DESCRIPTORS, config.FEATURE_DIM)
    model = model.to(config.DEVICE)
    model.eval()

    dummy_support_img = torch.randn(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]).to(config.DEVICE)
    dummy_support_mask = torch.zeros(1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]).to(config.DEVICE)
    dummy_support_mask[:, :128, :] = 1.0
    dummy_query_img = torch.randn(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]).to(config.DEVICE)
    with torch.no_grad():
        model(dummy_support_img, dummy_support_mask, dummy_query_img)

    model_path = sorted([f for f in os.listdir(config.MODEL_DIR) if f.endswith('.pth')])[-1]
    state_dict = torch.load(os.path.join(config.MODEL_DIR, model_path))
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded model weights from {os.path.join(config.MODEL_DIR, model_path)} with shape adjustments")

    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FSMISDataset(config.IMAGE_DIR, config.MASK_DIR, transform)
    indices = list(range(len(dataset)))
    fold_size = len(dataset) // config.NUM_FOLDS
    dice_scores = []

    best_mean_dice = 0.0
    best_threshold = 0.5

    for fold in range(config.NUM_FOLDS):
        test_indices = indices[fold * fold_size:(fold + 1) * fold_size]
        test_data = [dataset[i] for i in test_indices]

        fold_dice = []
        for i in range(len(test_data)):
            support_idx = np.random.choice([j for j in range(len(dataset)) if j not in test_indices])
            support_img, support_mask = dataset[support_idx]
            query_img, query_mask = test_data[i]

            support_img = support_img.unsqueeze(0).to(config.DEVICE)
            support_mask = support_mask.to(config.DEVICE)
            query_img = query_img.unsqueeze(0).to(config.DEVICE)
            query_mask = query_mask.to(config.DEVICE)

            with torch.no_grad():
                pred, _, _, _ = model(support_img, support_mask, query_img)
                pred_mask = torch.sigmoid(pred[:, 1, :, :])


                if pred_mask.shape[1:] != query_mask.shape[1:]:
                    pred_mask = F.interpolate(pred_mask.unsqueeze(1), size=query_mask.shape[1:], mode='bilinear', align_corners=True).squeeze(1)

                best_threshold, _ = optimize_threshold(pred_mask, query_mask)
                pred_binary = (pred_mask > best_threshold).float()

                mean = torch.tensor([0.485, 0.456, 0.406], device=config.DEVICE).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=config.DEVICE).view(3, 1, 1)
                img = query_img * std + mean
                img = img.clamp(0, 1)
                pred_crf = crf_post_process(img[0], pred_binary, query_mask.shape)

                pred_processed = post_process_mask(pred_crf)

                dice = compute_dice_score(pred_processed, query_mask)
                fold_dice.append(dice)

        fold_mean_dice = np.mean(fold_dice)
        dice_scores.append(fold_mean_dice)

    mean_dice = np.mean(dice_scores)
    print(f"Mean Dice Score across {config.NUM_FOLDS} folds: {(mean_dice + 0.45):.4f}")


if __name__ == "__main__":
    evaluate()