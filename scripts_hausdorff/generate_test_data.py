#!/usr/bin/env python3
"""
Generador de Datos de Prueba Sintéticos para Benchmark de Hausdorff

Crea volúmenes 3D sintéticos (predicciones y Ground Truths) con lesiones
realistas para validar el script de benchmark.

Uso:
    python generate_test_data.py --output test_data_dir --num_patients 5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm


def create_synthetic_lesion(volume: np.ndarray, z_start: int, z_end: int,
                           lesion_type: str = 'circle') -> np.ndarray:
    """
    Crea lesión sintética en volumen.
    
    Args:
        volume: Volumen 3D (512, 512, Z)
        z_start: Slice inicial
        z_end: Slice final
        lesion_type: 'circle', 'ellipse', 'irregular'
    
    Returns:
        Volumen con lesión
    """
    h, w, d = volume.shape
    center_h, center_w = h // 2, w // 2
    
    for z in range(z_start, z_end):
        if z < d:
            if lesion_type == 'circle':
                # Círculo con ruido
                radius = np.random.randint(20, 40)
                for i in range(h):
                    for j in range(w):
                        dist = np.sqrt((i - center_h)**2 + (j - center_w)**2)
                        if dist <= radius + np.random.randint(-5, 5):
                            volume[i, j, z] = 255
            
            elif lesion_type == 'ellipse':
                # Elipse
                a, b = np.random.randint(25, 45), np.random.randint(15, 30)
                for i in range(h):
                    for j in range(w):
                        if ((i - center_h)**2 / a**2 + (j - center_w)**2 / b**2) <= 1:
                            if np.random.random() > 0.1:  # 90% fill
                                volume[i, j, z] = 255
            
            elif lesion_type == 'irregular':
                # Forma irregular (suma de círculos)
                num_circles = np.random.randint(2, 4)
                for _ in range(num_circles):
                    r_center_h = center_h + np.random.randint(-30, 30)
                    r_center_w = center_w + np.random.randint(-30, 30)
                    radius = np.random.randint(15, 35)
                    
                    for i in range(h):
                        for j in range(w):
                            dist = np.sqrt((i - r_center_h)**2 + (j - r_center_w)**2)
                            if dist <= radius:
                                volume[i, j, z] = 255
    
    return volume


def create_synthetic_patient(output_dir: Path, patient_id: str, 
                            num_slices: int = 30):
    """
    Crea par (predicción, GT) sintético para un paciente.
    
    Args:
        output_dir: Directorio de salida
        patient_id: ID del paciente
        num_slices: Número de slices en el volumen
    """
    # Crear directorios
    pred_dir = output_dir / 'predictions'
    gt_dir = output_dir / 'ground_truth'
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Dimensiones del volumen (HU típicamente 512x512)
    h, w = 512, 512
    
    # Crear volúmenes vacíos
    pred_volume = np.zeros((h, w, num_slices), dtype=np.float32)
    gt_volume = np.zeros((h, w, num_slices), dtype=np.float32)
    
    # Crear Ground Truth (más "limpio")
    z_start_gt = np.random.randint(5, 10)
    z_end_gt = z_start_gt + np.random.randint(8, 15)
    lesion_type_gt = np.random.choice(['circle', 'ellipse'])
    gt_volume = create_synthetic_lesion(gt_volume, z_start_gt, z_end_gt, lesion_type_gt)
    
    # Crear predicción (similar pero con pequeñas variaciones)
    z_start_pred = z_start_gt + np.random.randint(-2, 3)
    z_end_pred = z_end_gt + np.random.randint(-2, 3)
    lesion_type_pred = lesion_type_gt
    pred_volume = create_synthetic_lesion(pred_volume, z_start_pred, z_end_pred, lesion_type_pred)
    
    # Agregar ruido (para simular errores de predicción)
    noise = np.random.normal(0, 0.1, pred_volume.shape)
    pred_volume = np.clip(pred_volume + noise * 50, 0, 255)
    
    # Normalizar a [0, 1] (como hacen las redes neuronales)
    gt_volume = gt_volume / 255.0
    pred_volume = pred_volume / 255.0
    
    # Crear objetos NIfTI
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0  # 1mm spacing
    
    gt_nifti = nib.Nifti1Image(gt_volume, affine)
    pred_nifti = nib.Nifti1Image(pred_volume, affine)
    
    # Guardar
    gt_file = gt_dir / f'Case_{patient_id}.nii.gz'
    pred_file = pred_dir / f'Case_{patient_id}_0000.nii.gz'
    
    nib.save(gt_nifti, str(gt_file))
    nib.save(pred_nifti, str(pred_file))
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Genera datos de prueba sintéticos para benchmark de Hausdorff'
    )
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Directorio de salida')
    parser.add_argument('--num_patients', '-n', type=int, default=5,
                       help='Número de pacientes sintéticos a generar')
    parser.add_argument('--slices_per_patient', type=int, default=30,
                       help='Número de slices por paciente')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generador de Datos de Prueba Sintéticos")
    print(f"{'='*60}\n")
    
    print(f"Directorio de salida: {output_dir}")
    print(f"Número de pacientes: {args.num_patients}")
    print(f"Slices por paciente: {args.slices_per_patient}\n")
    
    pbar = tqdm(total=args.num_patients, desc="Generando pacientes")
    
    for i in range(args.num_patients):
        patient_id = f"{i+1:05d}"
        try:
            create_synthetic_patient(output_dir, patient_id, args.slices_per_patient)
            pbar.update(1)
        except Exception as e:
            print(f"Error generando paciente {patient_id}: {e}")
    
    pbar.close()
    
    print(f"\n✓ Datos sintéticos generados exitosamente en: {output_dir}")
    print(f"✓ Predicciones: {output_dir}/predictions/")
    print(f"✓ Ground Truths: {output_dir}/ground_truth/\n")


if __name__ == '__main__':
    main()
