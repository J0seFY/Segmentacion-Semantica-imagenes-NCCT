

import argparse
import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.metrics import DiceMetric, ConfusionMatrixMetric, compute_hausdorff_distance

# Configuración de Matplotlib para servidores sin pantalla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importar modelos
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models import create_model
from dataset import NiftiSliceDataset


def preprocess_volume(nifti_img):
    """
    Replica preprocesamiento de entrenamiento:
    HU window [15,40] + min-max [0,1]
    """
    data = nifti_img.get_fdata()
    min_hu, max_hu = 15, 40
    data = np.clip(data, min_hu, max_hu)
    
    if (max_hu - min_hu) != 0:
        data = (data - min_hu) / (max_hu - min_hu)
    else:
        data = np.zeros_like(data)
        
    return data.astype(np.float32)


def save_best_slice_overlay(ct_vol, gt_vol, pred_vol, patient_id, metrics, save_path):
    """
    Visualiza el slice con mayor área de lesión en GT.
    Incluye métricas del paciente en el título.
    """
    z_indices = np.where(np.sum(gt_vol, axis=(0, 1)) > 0)[0]
    
    if len(z_indices) > 0:
        lesion_areas = [np.sum(gt_vol[:, :, z]) for z in z_indices]
        best_z = z_indices[np.argmax(lesion_areas)]
    else:
        pred_indices = np.where(np.sum(pred_vol, axis=(0, 1)) > 0)[0]
        if len(pred_indices) > 0:
            best_z = pred_indices[len(pred_indices)//2]
        else:
            best_z = ct_vol.shape[2] // 2

    ct_slice = np.rot90(ct_vol[:, :, best_z])
    gt_slice = np.rot90(gt_vol[:, :, best_z])
    pred_slice = np.rot90(pred_vol[:, :, best_z])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Título con métricas
    title = f"ID: {patient_id} | Slice: {best_z}\n"
    title += f"Dice: {metrics['dice']:.3f} | HD95: {metrics['hd95']:.1f}mm | "
    title += f"Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f}"
    
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title(title, fontsize=9)
    axes[0].axis('off')
    
    axes[1].imshow(ct_slice, cmap='gray')
    axes[1].imshow(gt_slice, cmap='Greens', alpha=0.5)
    axes[1].set_title('Ground Truth', fontsize=10)
    axes[1].axis('off')
    
    axes[2].imshow(ct_slice, cmap='gray')
    axes[2].imshow(pred_slice, cmap='Reds', alpha=0.5)
    axes[2].set_title('Prediction', fontsize=10)
    axes[2].axis('off')
    
    axes[3].imshow(ct_slice, cmap='gray')
    axes[3].imshow(gt_slice, cmap='Greens', alpha=0.4)
    axes[3].imshow(pred_slice, cmap='Reds', alpha=0.4)
    axes[3].set_title('Overlay (Yellow=TP)', fontsize=10)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history_path, output_dir):
    """Grafica curvas de entrenamiento."""
    if not os.path.exists(history_path):
        print("  [INFO] No se encontró historial de entrenamiento.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['val_dice'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Validation Dice', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


def evaluate_patient_unified(model, img_path, lbl_path, device):
    """
    Evaluación unificada por paciente:
    - Métricas slice-wise: Dice, Precision, Recall (agregadas por paciente)
    - Métrica volumétrica: HD95 en mm
    
    Returns:
        dict: Métricas del paciente + arrays para distribución de slices
    """
    # Cargar volumen
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)
    spacing = [float(x) for x in img_nii.header.get_zooms()[:3]]
    
    vol_data = preprocess_volume(img_nii)
    gt_data = lbl_nii.get_fdata()
    gt_data = (gt_data > 0).astype(np.uint8)
    
    h, w, d = vol_data.shape
    pred_vol = np.zeros_like(gt_data, dtype=np.uint8)
    
    # Métricas MONAI para slices
    dice_metric = DiceMetric(include_background=True, reduction="none")  # Sin agregación
    confusion_metric = ConfusionMatrixMetric(
        include_background=True, 
        metric_name=["precision", "recall"],
        reduction="none"
    )
    
    batch_size = 8
    model.eval()
    
    # Inferencia slice-by-slice con métricas acumuladas
    with torch.no_grad():
        for i in range(0, d, batch_size):
            end_i = min(i + batch_size, d)
            batch_data = vol_data[:, :, i:end_i].transpose(2, 0, 1)
            
            tensor_input = torch.from_numpy(batch_data).float().unsqueeze(1).to(device)
            tensor_gt = torch.from_numpy(gt_data[:, :, i:end_i].transpose(2, 0, 1)).unsqueeze(1).to(device)
            
            outputs = model(tensor_input)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).to(torch.uint8)
            
            # Calcular métricas slice-wise
            dice_metric(y_pred=preds, y=tensor_gt)
            confusion_metric(y_pred=preds, y=tensor_gt)
            
            # Guardar predicciones
            pred_vol[:, :, i:end_i] = preds.cpu().numpy().squeeze(1).transpose(1, 2, 0)
            
            del tensor_input, tensor_gt, outputs, probs, preds
            torch.cuda.empty_cache()
    
    # Agregar métricas slice-wise
    dice_per_slice = dice_metric.aggregate().cpu().numpy()  # Shape: (num_slices,)
    confusion_results = confusion_metric.aggregate()  # Shape: (2, num_slices)
    precision_per_slice = confusion_results[0].cpu().numpy()
    recall_per_slice = confusion_results[1].cpu().numpy()
    
    dice_metric.reset()
    confusion_metric.reset()
    
    # Promedios del paciente (ignorando NaNs de slices vacíos)
    dice_patient = np.nanmean(dice_per_slice)
    precision_patient = np.nanmean(precision_per_slice)
    recall_patient = np.nanmean(recall_per_slice)
    f1_patient = 2 * (precision_patient * recall_patient) / (precision_patient + recall_patient + 1e-8)
    
    # HD95 volumétrico en mm
    tensor_pred_5d = torch.from_numpy(pred_vol).unsqueeze(0).unsqueeze(0)
    tensor_gt_5d = torch.from_numpy(gt_data).unsqueeze(0).unsqueeze(0)
    
    hd95 = np.nan
    if np.sum(pred_vol) > 0 and np.sum(gt_data) > 0:
        hd95_metric = compute_hausdorff_distance(
            y_pred=tensor_pred_5d, 
            y=tensor_gt_5d, 
            include_background=False, 
            percentile=95, 
            spacing=spacing
        )
        hd95 = hd95_metric.item()
    elif np.sum(gt_data) == 0 and np.sum(pred_vol) == 0:
        hd95 = 0.0
    else:
        hd95 = 100.0  # Penalización por desajuste total
    
    metrics = {
        'dice': float(dice_patient),
        'hd95': float(hd95),
        'precision': float(precision_patient),
        'recall': float(recall_patient),
        'f1': float(f1_patient),
        # Distribuciones para análisis detallado
        'dice_per_slice': dice_per_slice.tolist(),
        'precision_per_slice': precision_per_slice.tolist(),
        'recall_per_slice': recall_per_slice.tolist(),
        'num_slices': d,
        'has_lesion': bool(np.sum(gt_data) > 0)
    }
    
    return metrics, vol_data, gt_data, pred_vol


def generate_boxplot_data(results, output_dir):
    """
    Genera archivo JSON con estructura optimizada para boxplots.
    Formato: { 'dice': [val1, val2, ...], 'hd95': [...], ... }
    """
    boxplot_data = {
        'dice': [],
        'hd95': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for patient in results:
        for metric in boxplot_data.keys():
            if metric in patient and not np.isnan(patient[metric]):
                boxplot_data[metric].append(patient[metric])
    
    with open(output_dir / 'boxplot_data.json', 'w') as f:
        json.dump(boxplot_data, f, indent=2)
    
    print(f"  [SAVED] Boxplot data → {output_dir / 'boxplot_data.json'}")


def main():
    parser = argparse.ArgumentParser(description='Unified Evaluation - Slice + Volumetric Metrics')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True, choices=['unet', 'attention_unet', 'hybrid_unet'])
    parser.add_argument('--use_aspp', action='store_true', default=True)
    parser.add_argument('--no_use_aspp', dest='use_aspp', action='store_false')
    parser.add_argument('--use_hint', action='store_true', default=True)
    parser.add_argument('--no_use_hint', dest='use_hint', action='store_false')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("UNIFIED EVALUATION - Slice-wise + Volumetric Metrics")
    print("="*70)
    print(f"Model:      {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {args.data_dir}")
    print(f"Device:     {device}")
    
    # Cargar modelo
    model = create_model(args.model_name, use_aspp=args.use_aspp, use_hint=args.use_hint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[10:] if k.startswith('_orig_mod.') else k] = v
    
    try:
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"  [WARN] Loading weights: {e}")
        model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    # Cargar test set
    ts_img_dir = Path(args.data_dir) / 'imagesTs'
    ts_lbl_dir = Path(args.data_dir) / 'labelsTs'
    
    if not ts_img_dir.exists():
        raise FileNotFoundError(f"Test images not found: {ts_img_dir}")
    
    image_files = sorted(list(ts_img_dir.glob('*.nii.gz')))
    print(f"Test patients: {len(image_files)}")
    print("="*70 + "\n")
    
    patient_results = []
    
    # Evaluar cada paciente
    for img_path in tqdm(image_files, desc="Evaluating Patients"):
        pid = img_path.name.replace('.nii.gz', '').replace('_0000', '')
        lbl_name = img_path.name.replace('_0000.nii.gz', '.nii.gz')
        lbl_path = ts_lbl_dir / lbl_name
        
        if not lbl_path.exists():
            lbl_path = ts_lbl_dir / img_path.name
        
        if not lbl_path.exists():
            print(f"  [SKIP] {pid}: No label found")
            continue
        
        try:
            metrics, ct, gt, pred = evaluate_patient_unified(
                model, str(img_path), str(lbl_path), device
            )
            metrics['patient_id'] = pid
            patient_results.append(metrics)
            
            # Visualización
            save_best_slice_overlay(ct, gt, pred, pid, metrics, viz_dir / f"{pid}.png")
            
        except Exception as e:
            print(f"  [ERROR] {pid}: {e}")
            import traceback
            traceback.print_exc()
    
    if not patient_results:
        print("\n[ERROR] No patients evaluated.")
        return
    
    # Estadísticas globales
    print("\n" + "="*70)
    print("GLOBAL STATISTICS (Mean ± Std)")
    print("="*70)
    
    summary_stats = {}
    metrics_keys = ['dice', 'hd95', 'precision', 'recall', 'f1']
    
    for metric in metrics_keys:
        vals = [p[metric] for p in patient_results if not np.isnan(p[metric])]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        median_val = np.median(vals)
        min_val = np.min(vals)
        max_val = np.max(vals)
        
        summary_stats[metric] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'median': float(median_val),
            'min': float(min_val),
            'max': float(max_val),
            'n_samples': len(vals)
        }
        
        unit = 'mm' if metric == 'hd95' else ''
        print(f"{metric.upper():<12}: {mean_val:.4f} ± {std_val:.4f} {unit}")
        print(f"              (median: {median_val:.4f}, range: [{min_val:.4f}, {max_val:.4f}])")
    
    print("="*70)
    
    # Guardar resultados completos
    final_report = {
        'model_config': {
            'model_name': args.model_name,
            'use_aspp': args.use_aspp,
            'use_hint': args.use_hint,
            'checkpoint': args.checkpoint
        },
        'summary_statistics': summary_stats,
        'patient_details': patient_results,
        'metadata': {
            'num_patients': len(patient_results),
            'test_data_dir': str(args.data_dir)
        }
    }
    
    with open(output_dir / 'unified_evaluation_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n[SAVED] Full report → {output_dir / 'unified_evaluation_report.json'}")
    
    # Generar datos para boxplot
    generate_boxplot_data(patient_results, output_dir)
    
    # Curvas de entrenamiento
    history_path = Path(args.checkpoint).parent / 'training_history.json'
    if history_path.exists():
        plot_training_curves(history_path, output_dir)
        print(f"[SAVED] Training curves → {output_dir / 'training_curves.png'}")
    
    print(f"\n[DONE] Evaluation complete. Results: {output_dir}\n")


if __name__ == '__main__':
    main()
