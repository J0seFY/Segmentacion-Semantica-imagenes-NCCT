#!/usr/bin/env python3
"""
Extractor de Puntos Reales desde Máscaras NIfTI

Este script cuenta la cantidad REAL de puntos (píxeles activos) en las máscaras
de predicción y ground truth, para luego usarlo en análisis de escalabilidad.

Entrada: Carpetas con predicciones y GT (.nii.gz)
Salida: CSV con información real de puntos por paciente

Uso:
    python extract_real_points.py \
      --pred predictions_dir \
      --gt ground_truth_dir \
      --output output/real_points.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def load_nifti(filepath: str) -> np.ndarray:
    """Carga archivo NIfTI y retorna datos como array numpy"""
    try:
        nii = nib.load(filepath)
        return nii.get_fdata()
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return None


def binarize_volume(volume: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binariza volumen: valores > threshold -> 1, resto -> 0"""
    return (volume > threshold).astype(np.uint8)


def count_points_per_slice(volume: np.ndarray) -> list:
    """
    Cuenta puntos activos por slice en volumen 3D.
    
    Returns:
        Lista con número de puntos en cada slice Z
    """
    points_per_slice = []
    for z in range(volume.shape[2]):
        slice_2d = volume[:, :, z]
        num_points = np.sum(slice_2d > 0)
        points_per_slice.append(num_points)
    
    return points_per_slice


def extract_patient_id(filename: str) -> str:
    """Extrae ID del paciente desde nombre de archivo"""
    # Soporta formatos: Case_XXXXX_0000.nii.gz o XXXXX.nii.gz
    stem = Path(filename).stem.replace('.nii', '')
    
    # Intentar formato Case_XXXXX_0000
    if 'Case_' in stem:
        parts = stem.split('_')
        if len(parts) >= 2:
            return parts[1]
    
    # Formato directo XXXXX
    parts = stem.split('_')
    return parts[0]


def process_patient(pred_file: Path, gt_file: Path, patient_id: str) -> dict:
    """
    Procesa un paciente y extrae información real de puntos.
    
    Returns:
        Diccionario con información detallada
    """
    # Cargar volúmenes
    pred_vol = load_nifti(str(pred_file))
    gt_vol = load_nifti(str(gt_file))
    
    if pred_vol is None or gt_vol is None:
        return None
    
    # Binarizar
    pred_vol = binarize_volume(pred_vol)
    gt_vol = binarize_volume(gt_vol)
    
    # Contar puntos por slice
    pred_points_per_slice = count_points_per_slice(pred_vol)
    gt_points_per_slice = count_points_per_slice(gt_vol)
    
    # Identificar slices con lesión (en ambos)
    slices_with_lesion = []
    pred_points_in_lesion_slices = []
    gt_points_in_lesion_slices = []
    
    for z in range(len(pred_points_per_slice)):
        if pred_points_per_slice[z] > 0 and gt_points_per_slice[z] > 0:
            slices_with_lesion.append(z)
            pred_points_in_lesion_slices.append(pred_points_per_slice[z])
            gt_points_in_lesion_slices.append(gt_points_per_slice[z])
    
    # Estadísticas
    total_pred_points = sum(pred_points_per_slice)
    total_gt_points = sum(gt_points_per_slice)
    
    avg_pred_points = np.mean(pred_points_in_lesion_slices) if pred_points_in_lesion_slices else 0
    avg_gt_points = np.mean(gt_points_in_lesion_slices) if gt_points_in_lesion_slices else 0
    
    # Promedio de puntos por slice con lesión (para escalabilidad)
    avg_points_per_slice = (avg_pred_points + avg_gt_points) / 2
    
    # Total de puntos en slices con lesión
    total_points_in_lesion_slices = sum(pred_points_in_lesion_slices) + sum(gt_points_in_lesion_slices)
    
    return {
        'patient_id': patient_id,
        'total_slices': pred_vol.shape[2],
        'slices_with_lesion': len(slices_with_lesion),
        'total_pred_points': total_pred_points,
        'total_gt_points': total_gt_points,
        'avg_pred_points_per_slice': avg_pred_points,
        'avg_gt_points_per_slice': avg_gt_points,
        'avg_points_per_slice': avg_points_per_slice,
        'total_points_in_lesion_slices': total_points_in_lesion_slices,
        'min_pred_points': min(pred_points_in_lesion_slices) if pred_points_in_lesion_slices else 0,
        'max_pred_points': max(pred_points_in_lesion_slices) if pred_points_in_lesion_slices else 0,
        'min_gt_points': min(gt_points_in_lesion_slices) if gt_points_in_lesion_slices else 0,
        'max_gt_points': max(gt_points_in_lesion_slices) if gt_points_in_lesion_slices else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extrae información real de puntos desde máscaras NIfTI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python extract_real_points.py \\
    --pred predictions_3d_fullres/ \\
    --gt labelsTs/ \\
    --output output/real_points_data.csv
        """
    )
    
    parser.add_argument('--pred', type=str, required=True,
                       help='Directorio con predicciones .nii.gz')
    parser.add_argument('--gt', type=str, required=True,
                       help='Directorio con Ground Truths .nii.gz')
    parser.add_argument('--output', '-o', type=str, default='real_points_data.csv',
                       help='Archivo CSV de salida')
    
    args = parser.parse_args()
    
    pred_dir = Path(args.pred)
    gt_dir = Path(args.gt)
    
    if not pred_dir.exists():
        print(f"Error: No existe {pred_dir}")
        return False
    
    if not gt_dir.exists():
        print(f"Error: No existe {gt_dir}")
        return False
    
    print(f"\n{'='*80}")
    print(f"EXTRACCIÓN DE PUNTOS REALES")
    print(f"{'='*80}\n")
    
    # Obtener lista de archivos
    pred_files = list(pred_dir.glob("*.nii.gz"))
    
    # Extraer IDs de pacientes
    patient_ids = set()
    for pred_file in pred_files:
        patient_id = extract_patient_id(pred_file.name)
        patient_ids.add(patient_id)
    
    patient_ids = sorted(list(patient_ids))
    
    print(f"Pacientes encontrados: {len(patient_ids)}\n")
    
    # Procesar cada paciente
    results = []
    
    pbar = tqdm(total=len(patient_ids), desc="Procesando pacientes")
    
    for patient_id in patient_ids:
        # Buscar archivos
        pred_files_patient = list(pred_dir.glob(f"*{patient_id}*.nii.gz"))
        gt_files_patient = list(gt_dir.glob(f"*{patient_id}*.nii.gz"))
        
        if not pred_files_patient or not gt_files_patient:
            pbar.update(1)
            continue
        
        # Procesar
        result = process_patient(pred_files_patient[0], gt_files_patient[0], patient_id)
        
        if result:
            results.append(result)
        
        pbar.update(1)
    
    pbar.close()
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    # Guardar CSV
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Datos extraídos exitosamente")
    print(f"✓ Archivo guardado: {output_file}")
    print(f"✓ Total de pacientes: {len(df)}")
    
    # Mostrar estadísticas
    print(f"\n{'='*80}")
    print(f"ESTADÍSTICAS DE PUNTOS")
    print(f"{'='*80}\n")
    
    print(f"Promedio de puntos por slice con lesión:")
    print(f"  Media: {df['avg_points_per_slice'].mean():.1f}")
    print(f"  Mediana: {df['avg_points_per_slice'].median():.1f}")
    print(f"  Min: {df['avg_points_per_slice'].min():.1f}")
    print(f"  Max: {df['avg_points_per_slice'].max():.1f}")
    
    print(f"\nTotal de puntos en slices con lesión:")
    print(f"  Media: {df['total_points_in_lesion_slices'].mean():.1f}")
    print(f"  Mediana: {df['total_points_in_lesion_slices'].median():.1f}")
    print(f"  Min: {df['total_points_in_lesion_slices'].min():.1f}")
    print(f"  Max: {df['total_points_in_lesion_slices'].max():.1f}")
    
    print(f"\n{'='*80}\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
