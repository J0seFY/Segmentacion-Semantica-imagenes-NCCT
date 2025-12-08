#!/usr/bin/env python3
"""
Thesis Evaluation Script - nnU-Net Post-Processing
Author: José Fuentes Yáñez (Refactored)
Compares existing nnU-Net predictions against Ground Truth using MONAI metrics.
"""

import argparse
import os
from pathlib import Path
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from monai.metrics import compute_hausdorff_distance, compute_dice

# Configuración de Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_clinical_overlay(ct_path, gt_path, pred_path, patient_id, save_path):
    """
    Visualización 2D de la slice más relevante.
    Requiere la imagen CT original para el fondo.
    """
    # Cargar volúmenes
    ct_vol = nib.load(ct_path).get_fdata()
    gt_vol = nib.load(gt_path).get_fdata()
    pred_vol = nib.load(pred_path).get_fdata()

    # Normalización simple para visualización (Windowing aproximado de Stroke)
    min_hu, max_hu = 15, 40
    ct_vol = np.clip(ct_vol, min_hu, max_hu)
    ct_vol = (ct_vol - min_hu) / (max_hu - min_hu)

    # Encontrar slice con más lesión en GT (o Pred si no hay GT)
    z_indices = np.where(np.sum(gt_vol, axis=(0, 1)) > 0)[0]
    if len(z_indices) > 0:
        # Slice con mayor área de lesión
        lesion_areas = [np.sum(gt_vol[:, :, z]) for z in z_indices]
        best_z = z_indices[np.argmax(lesion_areas)]
    else:
        pred_indices = np.where(np.sum(pred_vol, axis=(0, 1)) > 0)[0]
        best_z = pred_indices[len(pred_indices)//2] if len(pred_indices) > 0 else ct_vol.shape[2] // 2

    # Extraer y rotar slices
    slices = [np.rot90(vol[:, :, best_z]) for vol in [ct_vol, gt_vol, pred_vol]]
    ct_s, gt_s, pred_s = slices

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = [f'ID: {patient_id} | Z:{best_z}', 'Ground Truth', 'Predicción', 'Overlay']
    
    # 1. CT
    axes[0].imshow(ct_s, cmap='gray')
    
    # 2. GT
    axes[1].imshow(ct_s, cmap='gray')
    axes[1].imshow(gt_s, cmap='Greens', alpha=0.5)
    
    # 3. Pred
    axes[2].imshow(ct_s, cmap='gray')
    axes[2].imshow(pred_s, cmap='Reds', alpha=0.5)
    
    # 4. Overlay
    axes[3].imshow(ct_s, cmap='gray')
    axes[3].imshow(gt_s, cmap='Greens', alpha=0.4) # GT = Verde
    axes[3].imshow(pred_s, cmap='Reds', alpha=0.4)   # Pred = Rojo
    
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_folder(model_name, pred_dir, gt_dir, img_dir, output_dir, device):
    """
    Evalúa una carpeta específica (ej: 2d o ensemble)
    """
    print(f"\n>>> Evaluando: {model_name.upper()}")
    print(f"    Predicciones: {pred_dir}")
    
    viz_dir = output_dir / model_name / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Buscar archivos en la carpeta de predicciones
    pred_files = sorted(list(Path(pred_dir).glob('*.nii.gz')))
    if not pred_files:
        print(f"ERROR: No se encontraron predicciones en {pred_dir}")
        return None

    results = []

    for pred_path in tqdm(pred_files, desc=f"Procesando {model_name}"):
        filename = pred_path.name
        # nnU-Net output suele ser "Case_001.nii.gz". 
        # Asegúrate que GT tenga el mismo nombre.
        gt_path = Path(gt_dir) / filename
        
        # Para la imagen original, nnUNet suele añadir _0000
        # Intentamos adivinar el nombre de la imagen original para visualización
        img_name_candidate = filename.replace('.nii.gz', '_0000.nii.gz')
        img_path = Path(img_dir) / img_name_candidate

        if not gt_path.exists():
            print(f"FALLO EN: {gt_path}")
            print(f"--> Verifica si esta ruta existe exactamente.")
            continue

        # Cargar imágenes
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)
        
        # Obtener espaciado (Zoom) para HD95 en mm
        spacing = [float(x) for x in gt_nii.header.get_zooms()[:3]]
        
        
        # Datos a tensores (B, C, H, W, D) para MONAI
        # MONAI espera One-Hot o máscaras con canal. Aquí usaremos máscaras binarias.
        gt_data = torch.from_numpy(gt_nii.get_fdata()).float().to(device)
        pred_data = torch.from_numpy(pred_nii.get_fdata()).float().to(device)
        
        # Binarizar (por seguridad)
        gt_data = (gt_data > 0).float().unsqueeze(0).unsqueeze(0)   # (1, 1, H, W, D)
        pred_data = (pred_data > 0).float().unsqueeze(0).unsqueeze(0)

        # Métricas MONAI
        dice = compute_dice(y_pred=pred_data, y=gt_data, include_background=False).item()
        
        # HD95
        # Manejo de casos vacíos para evitar crash
        if torch.sum(gt_data) == 0 and torch.sum(pred_data) == 0:
             hd95 = 0.0
        elif torch.sum(gt_data) == 0 or torch.sum(pred_data) == 0:
             hd95 = np.nan # O penalización máx, ej. 100.0
        else:
            hd95 = compute_hausdorff_distance(
                y_pred=pred_data, 
                y=gt_data, 
                include_background=False, 
                percentile=95,
                spacing=spacing
            ).item()

        # Guardar resultado paciente
        pid = filename.replace('.nii.gz', '')
        results.append({
            'id': pid,
            'dice': dice,
            'hd95': hd95
        })

        # Generar Visualización (solo si existe la imagen original)
        if img_path.exists():
            save_clinical_overlay(img_path, gt_path, pred_path, pid, viz_dir / f"{pid}.png")

    # Resumen del modelo
    if results:
        dices = [r['dice'] for r in results]
        hds = [r['hd95'] for r in results if not np.isnan(r['hd95'])]
        
        summary = {
            'model': model_name,
            'mean_dice': np.mean(dices),
            'std_dice': np.std(dices),
            'mean_hd95': np.mean(hds) if hds else 0,
            'std_hd95': np.std(hds) if hds else 0,
            'details': results
        }
        return summary
    return None

def main():
    parser = argparse.ArgumentParser()
    # Rutas base de nnU-Net raw
    parser.add_argument('--gt_dir', required=True, help='Carpeta labelsTs')
    parser.add_argument('--img_dir', required=True, help='Carpeta imagesTs (para visualización)')
    parser.add_argument('--output_dir', required=True, help='Donde guardar reporte')
    
    # Rutas específicas de predicciones (Argumentos opcionales)
    parser.add_argument('--pred_2d', help='Carpeta predicciones 2D')
    parser.add_argument('--pred_3d', help='Carpeta predicciones 3D Fullres')
    parser.add_argument('--pred_ens', help='Carpeta predicciones Ensamble')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_eval = {}
    if args.pred_2d: models_to_eval['2d'] = args.pred_2d
    if args.pred_3d: models_to_eval['3d_fullres'] = args.pred_3d
    if args.pred_ens: models_to_eval['ensemble'] = args.pred_ens

    final_report = {}

    print(f"--- INICIANDO EVALUACIÓN MULTI-MODELO ---")
    
    for name, path in models_to_eval.items():
        summary = evaluate_folder(name, path, args.gt_dir, args.img_dir, output_dir, device)
        if summary:
            final_report[name] = summary
            print(f"Resultados {name}: Dice={summary['mean_dice']:.4f} | HD95={summary['mean_hd95']:.4f} mm")

    # Guardar JSON final
    with open(output_dir / 'comparison_report.json', 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print(f"\nEvaluación completa. Reporte guardado en {output_dir}")

if __name__ == '__main__':
    main()