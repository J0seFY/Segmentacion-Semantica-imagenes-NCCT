#!/usr/bin/env python3
"""
Evaluation Script for Ischemic Stroke Segmentation
Comprehensive metrics + visualizations
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
import torch.nn as nn
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except Exception:
    # matplotlib not available in the execution environment -> disable visualizations
    HAVE_MATPLOTLIB = False
from tqdm import tqdm
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
import nibabel as nib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models import create_model
from dataset import NiftiSliceDataset, FIXED_TEST_PATIENTS


def create_overlay(ct_slice, gt_mask, pred_mask):
    """Create 4-subplot visualization (no-op if matplotlib missing)"""
    if not HAVE_MATPLOTLIB:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original CT
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title('CT Original')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(ct_slice, cmap='gray')
    axes[1].imshow(gt_mask, cmap='Greens', alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(ct_slice, cmap='gray')
    axes[2].imshow(pred_mask, cmap='Reds', alpha=0.5)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(ct_slice, cmap='gray')
    axes[3].imshow(gt_mask, cmap='Greens', alpha=0.3)
    axes[3].imshow(pred_mask, cmap='Reds', alpha=0.3)
    axes[3].set_title('Overlay (GT=Green, Pred=Red)')
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_curves(history_path, output_dir):
    """Plot training/validation loss and dice"""
    if not HAVE_MATPLOTLIB:
        print("matplotlib not available: skipping training curves plot")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Dice
    ax2.plot(epochs, history['val_dice'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Validation Dice Score', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate stroke segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test NIfTI data')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save results')
    parser.add_argument('--model_name', type=str, required=True, choices=['unet', 'attention_unet', 'hybrid_unet'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    # Ablation flags: allow forcing same architecture used during training
    group_aspp = parser.add_mutually_exclusive_group()
    group_aspp.add_argument('--use_aspp', dest='use_aspp', action='store_true')
    group_aspp.add_argument('--no_use_aspp', dest='use_aspp', action='store_false')
    parser.set_defaults(use_aspp=None)
    group_hint = parser.add_mutually_exclusive_group()
    group_hint.add_argument('--use_hint', dest='use_hint', action='store_true')
    group_hint.add_argument('--no_use_hint', dest='use_hint', action='store_false')
    parser.set_defaults(use_hint=None)
    parser.add_argument('--visualize_every', type=int, default=10, help='Save visualization every N slices')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    print(f"Evaluating {args.model_name} on {device}")
    print(f"Test data: {args.data_dir}")
    
    # Load checkpoint safely (load to CPU first to avoid unexpected CUDA allocation)
    print(f"Loading checkpoint: {args.checkpoint} (map_location=cpu)")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Build model with matching ablation flags when provided
    model_kwargs = {'dropout': 0.0}
    if args.use_aspp is not None:
        model_kwargs['use_aspp'] = bool(args.use_aspp)
    if args.use_hint is not None:
        model_kwargs['use_hint'] = bool(args.use_hint)

    model = create_model(args.model_name, **model_kwargs)

    # Attempt to load state dict and move model to device afterwards
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print("RuntimeError while loading state_dict (trying non-strict load to show mismatches):")
        print(e)
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint val dice: {checkpoint.get('val_dice', 'N/A')}")
    
    # Create dataset from nnUNet test set (imagesTs/labelsTs)
    data_dir = Path(args.data_dir)
    image_dir = data_dir / 'imagesTs'
    mask_dir = data_dir / 'labelsTs'
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Test masks directory not found: {mask_dir}")
    
    image_files = sorted(image_dir.glob('*.nii.gz'))
    dataset = NiftiSliceDataset(
        image_paths=[str(f) for f in image_files],
        mask_paths=[str(mask_dir / f.name.replace('_0000', '')) for f in image_files],
        ignore_empty=False,
        is_training=False,
        transform=None
    )
    
    # Use pin_memory only when using CUDA; reduce num_workers for safer evaluation if needed
    pin_memory = True if device.type == 'cuda' else False
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    
    print(f"Test set: {len(dataset)} slices")
    
    # Metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    confusion_metric = ConfusionMatrixMetric(include_background=True, metric_name=["precision", "recall"])
    
    results_per_sample = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            # Compute metrics
            dice_metric(y_pred=preds, y=masks)
            hd95_metric(y_pred=preds, y=masks)
            confusion_metric(y_pred=preds, y=masks)
            
            # Save visualization for selected slices
            if idx % args.visualize_every == 0:
                sample_idx = idx
                ct = images[0, 0].cpu().numpy()
                gt = masks[0, 0].cpu().numpy()
                pred = preds[0, 0].cpu().numpy()
                
                fig = create_overlay(ct, gt, pred)
                fig.savefig(viz_dir / f'slice_{sample_idx:04d}.png', dpi=100, bbox_inches='tight')
                plt.close(fig)
            
            # Store per-sample results
            for b in range(images.size(0)):
                sample_result = {
                    'sample_idx': idx * args.batch_size + b,
                    'has_lesion': masks[b].sum().item() > 0,
                    'predicted_lesion': preds[b].sum().item() > 0
                }
                results_per_sample.append(sample_result)
    
    # Aggregate metrics
    dice_score = dice_metric.aggregate().item()
    hd95_score = hd95_metric.aggregate().item()
    confusion_results = confusion_metric.aggregate()
    precision = confusion_results[0].item()
    recall = confusion_results[1].item()
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    dice_metric.reset()
    hd95_metric.reset()
    confusion_metric.reset()
    
    # Results summary
    results = {
        'model_name': args.model_name,
        'checkpoint': str(args.checkpoint),
        'test_data': str(args.data_dir),
        'num_samples': len(dataset),
        'metrics': {
            'dice': float(dice_score),
            'hd95': float(hd95_score),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score)
        },
        'per_sample_results': results_per_sample
    }
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Test samples: {len(dataset)}")
    print(f"\nMetrics:")
    print(f"  Dice Score:     {dice_score:.4f}")
    print(f"  HD95:           {hd95_score:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1 Score:       {f1_score:.4f}")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    
    # Plot training curves if history exists
    history_path = Path(args.checkpoint).parent / 'training_history.json'
    if history_path.exists():
        plot_training_curves(history_path, output_dir)


if __name__ == '__main__':
    main()
