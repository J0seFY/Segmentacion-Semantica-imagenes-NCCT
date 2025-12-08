#!/usr/bin/env python3
"""
Training Script - FIXED VERSION
Usa split correcto por paciente y respeta test set fijo
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
import numpy as np
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models import create_model
from dataset import split_by_patient_ids, create_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    use_amp = scaler is not None
    
    for batch in progress:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        
        # Convert to channels_last if using CUDA
        if torch.cuda.is_available():
            images = images.to(memory_format=torch.channels_last)
            masks = masks.to(memory_format=torch.channels_last)
        
        # Faster gradient zeroing
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Backward pass with gradient scaling if using AMP
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, dice_metric, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    dice_metric.reset()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            # Convert to channels_last if using CUDA
            if torch.cuda.is_available():
                images = images.to(memory_format=torch.channels_last)
                masks = masks.to(memory_format=torch.channels_last)
            
            # Use AMP for inference too
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            
            preds = torch.sigmoid(outputs) > 0.5
            dice_metric(y_pred=preds, y=masks)
    
    val_loss = running_loss / len(loader.dataset)
    val_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    return val_loss, val_dice


def main():
    parser = argparse.ArgumentParser(description='Train stroke segmentation model - FIXED VERSION')
    parser.add_argument('--model_name', type=str, required=True, choices=['unet', 'attention_unet', 'hybrid_unet'])
    parser.add_argument('--data_dir', type=str, required=True, help='Path to nnUNet dataset root (e.g. nnUNet_raw/Dataset001_AISD)')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save results')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (hybrid_unet only)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement to reset patience')
    parser.add_argument('--poly_power', type=float, default=0.9, help='Poly learning rate decay power')
    parser.add_argument('--no_compile', action='store_true', default=False,
                        help='Disable torch.compile() to avoid Inductor compile crashes')
    parser.add_argument('--use_aspp', action='store_true', default=True,
                        help='Enable ASPP module (for hybrid_unet). Use --no-use_aspp to disable.')
    parser.add_argument('--no_use_aspp', dest='use_aspp', action='store_false',
                        help='Disable ASPP module (for hybrid_unet)')
    parser.add_argument('--use_hint', action='store_true', default=True,
                        help='Enable Hint module (for hybrid_unet). Use --no-use_hint to disable.')
    parser.add_argument('--no_use_hint', dest='use_hint', action='store_false',
                        help='Disable Hint module (for hybrid_unet)')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # CUDA optimizations (enabled by default)
    if torch.cuda.is_available():
        # cuDNN autotuner: finds optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for A100+ GPUs (faster with minimal precision loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"CUDA optimizations enabled: cuDNN benchmark, TF32")
    
    print(f"Training {args.model_name} on {device}")
    print(f"Dataset root (nnUNet): {args.data_dir}")
    print(f"Preprocessing: HU window [15,40] + min-max [0,1]")
    print(f"Results: {save_dir}")
    
    # Save config
    config = vars(args)
    config['device'] = str(device)
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Data - SPLIT CORRECTO POR PACIENTE
    data_root = Path(args.data_dir)
    images_tr_dir = data_root / 'imagesTr'
    labels_tr_dir = data_root / 'labelsTr'
    images_ts_dir = data_root / 'imagesTs'
    labels_ts_dir = data_root / 'labelsTs'

    # Directorio sanity checks
    missing = [p for p in [images_tr_dir, labels_tr_dir, images_ts_dir, labels_ts_dir] if not p.exists()]
    if missing:
        print("\nERROR: Faltan subdirectorios nnUNet esperados:")
        for m in missing:
            print(f" - {m}")
        print("Asegura que --data_dir apunta al root con imagesTr/labelsTr/imagesTs/labelsTs.")
        sys.exit(1)

    print("\nSplitting dataset (nnUNet structure) por patient IDs (incluye test fijo)...")
    split_data = split_by_patient_ids(
        str(images_tr_dir),
        str(labels_tr_dir),
        str(images_ts_dir),
        str(labels_ts_dir),
        val_ratio=args.val_split,
        seed=args.seed
    )
    
    train_loader, val_loader = create_dataloaders(
        split_data['train_images'],
        split_data['train_masks'],
        split_data['val_images'],
        split_data['val_masks'],
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"\nTrain: {len(split_data['train_images'])} volumes, {len(train_loader.dataset)} slices")
    print(f"Val: {len(split_data['val_images'])} volumes, {len(val_loader.dataset)} slices")
    print(f"Test: {len(split_data['test_images'])} volumes (fijo paper, no usado en entrenamiento)")
    
    # Model with CUDA optimizations
    # Pass ASPP/HINT flags to model factory (used for hybrid_unet ablations)
    model = create_model(args.model_name, dropout=args.dropout, use_aspp=getattr(args, 'use_aspp', True), use_hint=getattr(args, 'use_hint', True)).to(device)
    
    # Channels-last memory format for better GPU performance
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
        print("Model converted to channels_last memory format")
    
    # Optionally compile model with torch.compile for PyTorch 2.x
    if not args.no_compile:
        try:
            model = torch.compile(model, mode='max-autotune')
            print("Model compiled with torch.compile (max-autotune mode)")
        except Exception as e:
            print(f"torch.compile not available or failed: {e}")
            print("Continuing without compilation (PyTorch 2.0+ required)")
    else:
        print("Skipping torch.compile() because --no_compile was set")
    
    # Loss and optimizer
    criterion = DiceCELoss(sigmoid=True, include_background=True, to_onehot_y=False, 
                           lambda_dice=0.6, lambda_ce=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5, fused=True if torch.cuda.is_available() else False)
    
    # Mixed precision training (AMP) for faster training with lower memory
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    if scaler:
        print("Automatic Mixed Precision (AMP) enabled")
    
    # Poly learning rate policy: lr = lr_initial * (1 - epoch/max_epochs)^power
    def poly_lr_lambda(epoch):
        return (1 - epoch / args.epochs) ** args.poly_power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    # Training loop
    best_dice = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'lr': []}
    log_file = save_dir / f'{args.model_name}_training.log'
    
    with open(log_file, 'w') as f:
        f.write(f"Epoch,TrainLoss,ValLoss,ValDice,LR,Time\n")
    
    print(f"\nStarting training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=scaler)
        val_loss, val_dice = validate(model, val_loader, criterion, dice_metric, device, use_amp=(scaler is not None))
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()  # Poly LR update
        
        epoch_time = time.time() - epoch_start
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(current_lr)
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_dice:.6f},{current_lr:.8f},{epoch_time:.2f}\n")
        
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.2e}")
        
        # Save best model and early stopping check
        if val_dice > best_dice + args.min_delta:
            best_dice = val_dice
            patience_counter = 0
            # Save original model (unwrap if compiled)
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'config': config
            }, save_dir / 'best_model.pth')
            print(f"  Best model saved (Dice: {val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs)")
            break
        
        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Final model
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'epoch': epoch if patience_counter >= args.patience else args.epochs,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
        'config': config
    }, save_dir / 'final_model.pth')
    
    # Save history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Best validation Dice: {best_dice:.4f}")
    if patience_counter >= args.patience:
        print(f"Stopped early at epoch {epoch}/{args.epochs}")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
