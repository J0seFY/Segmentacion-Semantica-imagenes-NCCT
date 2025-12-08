#!/usr/bin/env python3
"""
Thesis Evaluation Script - 3D Volumetric Metrics
Author: José Fuentes Yáñez
Validates segmentation models with clinical metrics (Dice, HD95 in mm) per patient.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance, compute_dice

# Configuración de Matplotlib para servidores sin pantalla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importar modelos
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models import create_model

def preprocess_volume(nifti_img):
    """
    REPLICA EL PREPROCESAMIENTO DE ENTRENAMIENTO.
    Basado en tu log: 'HU window [15,40] + min-max [0,1]'
    """
    data = nifti_img.get_fdata()
    
    # --- VENTANEO (WINDOWING) ---
    # Según tu script de entrenamiento, usas ventana [15, 40]
    # Esto es muy estrecho (alto contraste), verifica si es correcto.
    # Si en dataset.py usas otra cosa, CAMBIA ESTO.
    min_hu = 15
    max_hu = 40
    
    data = np.clip(data, min_hu, max_hu)
    
    # --- NORMALIZACIÓN MIN-MAX [0, 1] ---
    # Evitar división por cero si el volumen es plano
    if (max_hu - min_hu) != 0:
        data = (data - min_hu) / (max_hu - min_hu)
    else:
        data = np.zeros_like(data)
        
    return data

def save_clinical_overlay(ct_vol, gt_vol, pred_vol, patient_id, save_path):
    """
    Genera una visualización de la 'mejor' slice (la que tiene más lesión).
    Si no hay lesión en GT, busca la que tiene más predicción o el centro.
    """
    # Encontrar índices donde hay lesión en el GT
    z_indices = np.where(np.sum(gt_vol, axis=(0, 1)) > 0)[0]
    
    if len(z_indices) > 0:
        # Elegir la slice con mayor área de lesión
        lesion_areas = [np.sum(gt_vol[:, :, z]) for z in z_indices]
        best_z = z_indices[np.argmax(lesion_areas)]
    else:
        # Si no hay GT (caso negativo), buscar si hay predicción
        pred_indices = np.where(np.sum(pred_vol, axis=(0, 1)) > 0)[0]
        if len(pred_indices) > 0:
            best_z = pred_indices[len(pred_indices)//2]
        else:
            # Centro del volumen
            best_z = ct_vol.shape[2] // 2

    # Extraer slices 2D
    ct_slice = ct_vol[:, :, best_z]
    gt_slice = gt_vol[:, :, best_z]
    pred_slice = pred_vol[:, :, best_z]

    # Rotación estándar para visualización (depende de la orientación NIfTI)
    # Usualmente rot90 ayuda a verlo "de frente"
    ct_slice = np.rot90(ct_slice)
    gt_slice = np.rot90(gt_slice)
    pred_slice = np.rot90(pred_slice)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. CT Original
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title(f'ID: {patient_id} | Slice: {best_z}', fontsize=10)
    axes[0].axis('off')
    
    # 2. Ground Truth
    axes[1].imshow(ct_slice, cmap='gray')
    axes[1].imshow(gt_slice, cmap='Greens', alpha=0.5)
    axes[1].set_title('Ground Truth (Verde)', fontsize=10)
    axes[1].axis('off')
    
    # 3. Prediction
    axes[2].imshow(ct_slice, cmap='gray')
    axes[2].imshow(pred_slice, cmap='Reds', alpha=0.5)
    axes[2].set_title('Predicción (Rojo)', fontsize=10)
    axes[2].axis('off')
    
    # 4. Overlay
    axes[3].imshow(ct_slice, cmap='gray')
    axes[3].imshow(gt_slice, cmap='Greens', alpha=0.4)
    axes[3].imshow(pred_slice, cmap='Reds', alpha=0.4)
    axes[3].set_title('Superposición (Amarillo=TP)', fontsize=10)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_curves(history_path, output_dir):
    """Grafica las curvas de entrenamiento desde el JSON."""
    if not os.path.exists(history_path):
        print("Aviso: No se encontró historial de entrenamiento para graficar.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Curvas de Pérdida (Loss)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dice
    ax2.plot(epochs, history['val_dice'], label='Val Dice', color='green')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validación Dice Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()

def evaluate_patient(model, img_path, lbl_path, device):
    """
    Procesa UN paciente:
    1. Carga volumen completo.
    2. Preprocesa.
    3. Inferencia por batch (para no saturar VRAM).
    4. Reconstruye volumen 3D.
    5. Calcula métricas 3D (con unidades físicas mm).
    """
    # Cargar NIfTI
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)
    
    # Obtener espaciado físico (mm) para HD95
    spacing = [float(x) for x in img_nii.header.get_zooms()[:3]]
    
    # Preprocesar
    vol_data = preprocess_volume(img_nii)
    gt_data = lbl_nii.get_fdata()
    gt_data = (gt_data > 0).astype(np.uint8) # Binarizar GT
    
    # Dimensiones: (H, W, D) -> Asumimos slices en el eje Z (índice 2)
    h, w, d = vol_data.shape
    
    # Contenedor para predicción 3D
    pred_vol = np.zeros_like(gt_data, dtype=np.uint8)
    
    # Inferencia (Slice por slice o batch pequeño)
    # Reducido de 16 a 4 para modelos pesados (Hybrid UNet con full-scale connections)
    batch_size = 8
    model.eval()
    
    with torch.no_grad():
        for i in range(0, d, batch_size):
            end_i = min(i + batch_size, d)
            
            # Extraer batch de slices y transponer a (B, H, W)
            # Nota: NIfTI es (H, W, D), necesitamos (D, H, W) para PyTorch
            batch_data = vol_data[:, :, i:end_i].transpose(2, 0, 1)
            
            # Tensorizar: (B, 1, H, W)
            tensor_input = torch.from_numpy(batch_data).float().unsqueeze(1).to(device)
            
            # Inferencia
            # Usar AMP si se usó en entrenamiento para consistencia, aunque en inferencia afecta menos
            outputs = model(tensor_input)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).cpu().numpy().squeeze(1) # (B, H, W)
            
            # Guardar en volumen reconstruido (transponer de vuelta a H,W,D)
            pred_vol[:, :, i:end_i] = preds.transpose(1, 2, 0)
            
            # Liberar memoria GPU explícitamente para modelos pesados
            del tensor_input, outputs, probs
            torch.cuda.empty_cache()

    # --- CÁLCULO DE MÉTRICAS 3D ---
    
    # Aplanar para métricas simples (Prec, Recall, Acc)
    flat_gt = gt_data.flatten()
    flat_pred = pred_vol.flatten()
    
    tp = np.sum((flat_gt == 1) & (flat_pred == 1))
    tn = np.sum((flat_gt == 0) & (flat_pred == 0))
    fp = np.sum((flat_gt == 0) & (flat_pred == 1))
    fn = np.sum((flat_gt == 1) & (flat_pred == 0))
    
    smooth = 1e-6
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth) # Sensibilidad
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    dice_score = (2 * tp) / (2 * tp + fp + fn + smooth)

    # HD95 (Métrica Espacial usando MONAI)
    # MONAI requiere formato (B, C, spatial...) -> (1, 1, H, W, D)
    tensor_pred_5d = torch.from_numpy(pred_vol).unsqueeze(0).unsqueeze(0)
    tensor_gt_5d = torch.from_numpy(gt_data).unsqueeze(0).unsqueeze(0)
    
    hd95 = np.nan
    # HD95 falla si uno de los volúmenes está vacío
    if np.sum(pred_vol) > 0 and np.sum(gt_data) > 0:
        hd95_metric = compute_hausdorff_distance(
            y_pred=tensor_pred_5d, 
            y=tensor_gt_5d, 
            include_background=False, 
            percentile=95, 
            spacing=spacing # ¡CRÍTICO: Unidades reales!
        )
        hd95 = hd95_metric.item()
    elif np.sum(gt_data) == 0 and np.sum(pred_vol) == 0:
        hd95 = 0.0 # Perfecto (ambos vacíos)
    else:
        # Penalización por fallo catastrófico (uno vacío y el otro no)
        hd95 = 100.0 # Valor alto arbitrario (mm) para indicar fallo
        
    metrics = {
        'dice': dice_score,
        'hd95': hd95,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1
    }
    
    return metrics, vol_data, gt_data, pred_vol

def main():
    parser = argparse.ArgumentParser(description='Thesis Evaluation - 3D Patient-wise')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to nnUNet raw dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Folder to save results')
    parser.add_argument('--model_name', type=str, required=True, choices=['unet', 'attention_unet', 'hybrid_unet'])
    # Flags para replicar arquitectura exacta
    parser.add_argument('--use_aspp', action='store_true', default=True)
    parser.add_argument('--no_use_aspp', dest='use_aspp', action='store_false')
    parser.add_argument('--use_hint', action='store_true', default=True)
    parser.add_argument('--no_use_hint', dest='use_hint', action='store_false')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- EVALUACIÓN DE TESIS (Modo Clínico 3D) ---")
    print(f"Modelo: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Datos: {args.data_dir}")
    
    # 1. Cargar Modelo
    # Pasar argumentos de arquitectura para asegurar coincidencia
    model = create_model(args.model_name, use_aspp=args.use_aspp, use_hint=args.use_hint)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Manejo de 'state_dict' compilado (prefijo _orig_mod)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v # Quitar prefijo
        else:
            new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"\nAdvertencia de carga de pesos: {e}")
        print("Intentando carga no estricta...")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.to(device)
    model.eval()
    
    # 2. Configurar Datos (Test Set nnUNet)
    # Buscamos en imagesTs (Test set fijo)
    ts_img_dir = Path(args.data_dir) / 'imagesTs'
    ts_lbl_dir = Path(args.data_dir) / 'labelsTs'
    
    if not ts_img_dir.exists():
        raise FileNotFoundError(f"No se encontró carpeta de test: {ts_img_dir}")

    image_files = sorted(list(ts_img_dir.glob('*.nii.gz')))
    print(f"Pacientes encontrados en Test Set: {len(image_files)}")
    
    patient_results = []
    
    # 3. Bucle de Evaluación
    for img_path in tqdm(image_files, desc="Evaluando Pacientes"):
        pid = img_path.name.replace('.nii.gz', '').replace('_0000', '')
        
        # Buscar label correspondiente
        # nnUnet convención: imagen=Case_001_0000.nii.gz, label=Case_001.nii.gz
        lbl_name = img_path.name.replace('_0000.nii.gz', '.nii.gz')
        lbl_path = ts_lbl_dir / lbl_name
        
        if not lbl_path.exists():
            # Intento alternativo (mismo nombre)
            lbl_path = ts_lbl_dir / img_path.name
        
        if not lbl_path.exists():
            print(f"Skipping {pid}: No ground truth label found.")
            continue
            
        try:
            metrics, ct, gt, pred = evaluate_patient(model, str(img_path), str(lbl_path), device)
            metrics['patient_id'] = pid
            patient_results.append(metrics)
            
            # Guardar visualización
            save_clinical_overlay(ct, gt, pred, pid, viz_dir / f"{pid}_eval.png")
            
        except Exception as e:
            print(f"Error evaluando paciente {pid}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Resumen Estadístico
    if not patient_results:
        print("No se evaluaron pacientes.")
        return

    print("\n" + "="*50)
    print("RESULTADOS GLOBALES (Media ± Desviación Estándar)")
    print("="*50)
    
    summary_stats = {}
    keys = ['dice', 'hd95', 'precision', 'recall', 'accuracy']
    
    for k in keys:
        # Filtrar NaNs (casos fallidos) para el promedio
        vals = [p[k] for p in patient_results if not np.isnan(p[k])]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        summary_stats[k] = {'mean': mean_val, 'std': std_val}
        
        print(f"{k.upper():<10}: {mean_val:.4f} ± {std_val:.4f}")
        
    # 5. Guardar Reporte
    report = {
        'model_config': vars(args),
        'summary': summary_stats,
        'patient_details': patient_results
    }
    
    with open(output_dir / 'final_metrics_report.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    # 6. Graficar Entrenamiento
    history_path = Path(args.checkpoint).parent / 'training_history.json'
    if history_path.exists():
        plot_training_curves(history_path, output_dir)
        print(f"\nCurvas de entrenamiento guardadas.")
    
    print(f"\nEvaluación completa. Resultados en: {output_dir}")

if __name__ == '__main__':
    main()