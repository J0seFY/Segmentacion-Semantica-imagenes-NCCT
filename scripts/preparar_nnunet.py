#!/usr/bin/env python3
"""
Script automatizado para preparar, entrenar y evaluar un modelo nnU-Net v2
para segmentación de ACV isquémico.

Autor: Pipeline automatizado para tesis
Fecha: 2025
"""

import os
import argparse
import subprocess
import json
import glob
import shutil
import numpy as np
import torch
import nibabel as nib
import cv2
import pydicom
from tqdm import tqdm
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

DATASET_DICOM_ROOT = "/home/jfuentes/Segmentacion-semantica-en-imagenes-biomedicas-de-ACV-isquemico-con-arquitecturas-U-Net/Dataset_dicom"
# Intentar ambas variantes de la variable de entorno (con mayúsculas/minúsculas)
NNUNET_RAW_PATH = os.environ.get('nnUNet_raw') or os.environ.get('nnUNET_raw')
DATASET_ID = "001"
DATASET_NAME = "AISD"

# Etiquetas de lesión a binarizar en máscaras PNG
LESION_LABELS = [1, 2, 3, 5]

# Lista fija de pacientes de prueba
FIXED_TEST_PATIENTS = [
    '0073410', '0072723', '0226290', '0537908', '0538058', '0091415',
    '0538780', '0073540', '0226188', '0226258', '0226314', '0091507',
    '0226298', '0538975', '0226257', '0226142', '0072681', '0091538',
    '0538983', '0537961', '0091646', '0072765', '0226137', '0091621',
    '0091458', '0021822', '0538319', '0226133', '0091657', '0537925',
    '0073489', '0538502', '0091476', '0226136', '0538532', '0073312',
    '0539025', '0226309', '0226307', '0091383', '0021092', '0537990',
    '0226299', '0073060', '0538505', '0073424', '0091534', '0226125',
    '0072691', '0538425', '0226199', '0226261'
]

# ============================================================================
# VERIFICACIONES INICIALES
# ============================================================================

def verify_environment():
    """Verifica que el entorno esté configurado correctamente."""
    print("🔍 Verificando configuración del entorno...")
    
    # Verificar/crear variable de entorno nnUNet_raw
    global NNUNET_RAW_PATH
    if NNUNET_RAW_PATH is None:
        # Intentar leer en tiempo de ejecución
        raw_from_env = os.environ.get('nnUNet_raw') or os.environ.get('nnUNET_raw')
        if raw_from_env is None:
            # Fallback: crear en la raíz del proyecto, junto a este script
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            NNUNET_RAW_PATH = os.path.join(project_root, 'nnUNet_raw')
            os.makedirs(NNUNET_RAW_PATH, exist_ok=True)
            os.environ['nnUNet_raw'] = NNUNET_RAW_PATH
            print(f"   ⚠️  nnUNet_raw no estaba definido. Configurado automáticamente a: {NNUNET_RAW_PATH}")
        else:
            NNUNET_RAW_PATH = raw_from_env
    print(f"   ✅ nnUNet_raw: {NNUNET_RAW_PATH}")

    # Asegurar que el directorio base de nnUNet_raw exista
    os.makedirs(NNUNET_RAW_PATH, exist_ok=True)

    # Verificar/crear nnUNet_preprocessed
    nnunet_preprocessed = os.environ.get('nnUNet_preprocessed') or os.environ.get('nnUNET_preprocessed')
    if nnunet_preprocessed is None:
        base_dir = os.path.dirname(NNUNET_RAW_PATH.rstrip('/')) or os.getcwd()
        nnunet_preprocessed = os.path.join(base_dir, 'nnUNet_preprocessed')
        os.makedirs(nnunet_preprocessed, exist_ok=True)
        os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
        print(f"   ⚠️  nnUNet_preprocessed no estaba definido. Configurado automáticamente a: {nnunet_preprocessed}")
    else:
        # asegurar que exista
        os.makedirs(nnunet_preprocessed, exist_ok=True)
    print(f"   ✅ nnUNet_preprocessed: {nnunet_preprocessed}")

    # Verificar/crear nnUNet_results
    nnunet_results = os.environ.get('nnUNet_results') or os.environ.get('nnUNET_results')
    if nnunet_results is None:
        base_dir = os.path.dirname(NNUNET_RAW_PATH.rstrip('/')) or os.getcwd()
        nnunet_results = os.path.join(base_dir, 'nnUNet_results')
        os.makedirs(nnunet_results, exist_ok=True)
        os.environ['nnUNet_results'] = nnunet_results
        print(f"   ⚠️  nnUNet_results no estaba definido. Configurado automáticamente a: {nnunet_results}")
    else:
        os.makedirs(nnunet_results, exist_ok=True)
    print(f"   ✅ nnUNet_results: {nnunet_results}")
    
    # Verificar que pydicom esté instalado
    try:
        import pydicom
        print("   ✅ pydicom está instalado")
    except ImportError:
        raise EnvironmentError(
            "❌ pydicom no está instalado.\n"
            "Instálalo con: pip install pydicom"
        )
    
    # Verificar que el dataset DICOM existe
    if not os.path.exists(DATASET_DICOM_ROOT):
        raise FileNotFoundError(
            f"❌ El dataset DICOM no existe en: {DATASET_DICOM_ROOT}\n"
            f"Por favor, verifica la ruta DATASET_DICOM_ROOT en el script."
        )
    print(f"   ✅ Dataset DICOM encontrado: {DATASET_DICOM_ROOT}")
    
    print("✅ Todas las verificaciones pasaron correctamente.\n")


# ============================================================================
# FASE 1: PREPARACIÓN DE DATOS
# ============================================================================

def setup_directories():
    """Crea la estructura de carpetas para nnU-Net."""
    print("📁 Creando estructura de directorios para nnU-Net...")
    
    dataset_path = os.path.join(NNUNET_RAW_PATH, f"Dataset{DATASET_ID}_{DATASET_NAME}")
    
    dirs_to_create = [
        os.path.join(dataset_path, "imagesTr"),
        os.path.join(dataset_path, "labelsTr"),
        os.path.join(dataset_path, "imagesTs"),
        os.path.join(dataset_path, "labelsTs"),
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    print(f"✅ Estructura creada en: {dataset_path}\n")
    return dataset_path


def get_patient_lists():
    """Obtiene las listas de pacientes de entrenamiento y prueba."""
    print("👥 Escaneando pacientes en el dataset DICOM...")
    
    image_dir = os.path.join(DATASET_DICOM_ROOT, "image")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"❌ No se encuentra la carpeta: {image_dir}")
    
    # Obtener todos los pacientes
    all_patients = [
        p for p in os.listdir(image_dir) 
        if os.path.isdir(os.path.join(image_dir, p))
    ]
    
    print(f"   📊 Total de pacientes encontrados: {len(all_patients)}")
    
    # Separar en train y test
    test_patients = [p for p in all_patients if p in FIXED_TEST_PATIENTS]
    train_patients = [p for p in all_patients if p not in FIXED_TEST_PATIENTS]
    
    print(f"   🎯 Pacientes de entrenamiento: {len(train_patients)}")
    print(f"   🧪 Pacientes de prueba: {len(test_patients)}")
    
    if len(test_patients) != len(FIXED_TEST_PATIENTS):
        missing = set(FIXED_TEST_PATIENTS) - set(test_patients)
        if missing:
            print(f"   ⚠️  Advertencia: {len(missing)} pacientes de test no encontrados: {missing}")
    
    print()
    return train_patients, test_patients


def dicom_to_nifti(dicom_dir, output_path):
    """
    Convierte una serie DICOM a formato NIfTI usando pydicom.
    
    Args:
        dicom_dir: Directorio con archivos DICOM
        output_path: Ruta de salida para el archivo .nii.gz
    
    Returns:
        nib.Nifti1Image o None si falla
    """
    # Obtener todos los archivos DICOM
    dicom_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    
    if not dicom_files:
        return None
    
    # Leer todos los DICOM
    slices = []
    for dcm_file in dicom_files:
        try:
            ds = pydicom.dcmread(dcm_file, force=True)
            if hasattr(ds, 'PixelData'):
                slices.append(ds)
        except Exception:
            continue
    
    if not slices:
        return None
    
    # Ordenar por ImagePositionPatient (posición Z)
    def get_z_position(dcm):
        if hasattr(dcm, 'ImagePositionPatient'):
            return float(dcm.ImagePositionPatient[2])
        elif hasattr(dcm, 'SliceLocation'):
            return float(dcm.SliceLocation)
        elif hasattr(dcm, 'InstanceNumber'):
            return float(dcm.InstanceNumber)
        return 0.0
    
    slices = sorted(slices, key=get_z_position)
    
    # Extraer arrays de píxeles
    pixel_arrays = []
    for ds in slices:
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Aplicar RescaleSlope y RescaleIntercept si existen
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        
        pixel_arrays.append(pixel_array)
    
    # Stack en volumen 3D: (H, W, D) -> transponer a (W, H, D) para NIfTI
    volume = np.stack(pixel_arrays, axis=-1)
    
    # Construir matriz affine
    first_ds = slices[0]
    
    # Valores por defecto
    affine = np.eye(4)
    
    if hasattr(first_ds, 'ImageOrientationPatient') and hasattr(first_ds, 'ImagePositionPatient') and hasattr(first_ds, 'PixelSpacing'):
        # Reconstrucción completa del affine
        iop = np.array(first_ds.ImageOrientationPatient).reshape(2, 3)
        row_cosines = iop[0]
        col_cosines = iop[1]
        slice_normal = np.cross(row_cosines, col_cosines)
        
        pixel_spacing = [float(x) for x in first_ds.PixelSpacing]
        row_spacing = pixel_spacing[0]
        col_spacing = pixel_spacing[1]
        
        # Calcular spacing entre slices
        if len(slices) > 1 and hasattr(slices[1], 'ImagePositionPatient'):
            pos0 = np.array(first_ds.ImagePositionPatient)
            pos1 = np.array(slices[1].ImagePositionPatient)
            slice_spacing = np.abs(np.dot(pos1 - pos0, slice_normal))
            if slice_spacing == 0:
                slice_spacing = float(getattr(first_ds, 'SliceThickness', 1.0))
        else:
            slice_spacing = float(getattr(first_ds, 'SliceThickness', 1.0))
        
        # Construir matriz de rotación
        rotation_matrix = np.column_stack([
            col_cosines * col_spacing,
            row_cosines * row_spacing,
            slice_normal * slice_spacing
        ])
        
        affine[:3, :3] = rotation_matrix
        affine[:3, 3] = np.array(first_ds.ImagePositionPatient)
    else:
        # Affine simplificado con spacing
        pixel_spacing = getattr(first_ds, 'PixelSpacing', [1.0, 1.0])
        slice_thickness = getattr(first_ds, 'SliceThickness', 1.0)
        
        affine[0, 0] = float(pixel_spacing[1])
        affine[1, 1] = float(pixel_spacing[0])
        affine[2, 2] = float(slice_thickness)
    
    # Crear imagen NIfTI
    nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine)
    
    # Guardar
    nib.save(nifti_img, output_path)
    
    return nifti_img


def process_patient(patient_id, split, dataset_path):
    """
    Procesa un paciente: convierte DICOM a NIfTI, crea máscara alineada.
    
    Args:
        patient_id: ID del paciente
        split: 'Tr' para training, 'Ts' para test
        dataset_path: Ruta base del dataset nnU-Net
    """
    # Rutas de origen
    dicom_dir = os.path.join(DATASET_DICOM_ROOT, "image", patient_id, "CT")
    mask_dir = os.path.join(DATASET_DICOM_ROOT, "mask", patient_id)
    
    # Rutas de destino
    images_output = os.path.join(dataset_path, f"images{split}")
    labels_output = os.path.join(dataset_path, f"labels{split}")
    
    # Verificar que existen las carpetas de origen
    if not os.path.exists(dicom_dir):
        print(f"   ⚠️  DICOM no encontrado para {patient_id}, saltando...")
        return False
    
    if not os.path.exists(mask_dir):
        print(f"   ⚠️  Máscara no encontrada para {patient_id}, saltando...")
        return False
    
    try:
        # ====================================================================
        # PASO A-C: CONVERTIR DICOM A NIFTI CON PYDICOM
        # ====================================================================
        
        # Ruta de salida para la imagen
        target_nifti = os.path.join(images_output, f"{patient_id}_0000.nii.gz")
        
        # Convertir DICOM a NIfTI usando pydicom
        nifti_img = dicom_to_nifti(dicom_dir, target_nifti)
        
        if nifti_img is None:
            print(f"   ❌ No se pudo convertir DICOM para {patient_id}")
            return False
        
        # ====================================================================
        # PASO D-F: CARGAR Y APILAR MÁSCARAS PNG
        # ====================================================================
        
        # Encontrar todos los PNG
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        
        if not mask_files:
            print(f"   ⚠️  No se encontraron máscaras PNG para {patient_id}")
            return False
        
        # Ordenar numéricamente por nombre de archivo (000.png, 001.png, ...)
        mask_files = sorted(mask_files, key=lambda x: int(Path(x).stem))
        
        # Leer y apilar máscaras
        mask_slices = []
        for mask_file in mask_files:
            img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"   ⚠️  No se pudo leer {mask_file}")
                continue
            
            # Binarizar: 0 (fondo) y 1 (lesión) usando etiquetas específicas
            binary_mask = np.isin(img, LESION_LABELS).astype(np.uint8)
            mask_slices.append(binary_mask)
        
        if not mask_slices:
            print(f"   ❌ No se pudieron cargar máscaras para {patient_id}")
            return False
        
        # Stack en 3D: (H, W, D)
        mask_volume = np.stack(mask_slices, axis=-1)
        
        # ====================================================================
        # PASO G-I: ALINEAR MÁSCARA CON IMAGEN Y GUARDAR
        # ====================================================================
        
        # Cargar la imagen NIfTI para obtener affine y header
        img_nifti = nib.load(target_nifti)
        img_affine = img_nifti.affine
        img_header = img_nifti.header
        
        # Verificar dimensiones
        img_shape = img_nifti.shape
        mask_shape = mask_volume.shape
        
        # Si las dimensiones no coinciden, ajustar la máscara
        if img_shape != mask_shape:
            print(f"   ⚠️  Dimensiones no coinciden para {patient_id}:")
            print(f"      Imagen: {img_shape}, Máscara: {mask_shape}")
            
            # Intentar redimensionar o rellenar con ceros
            # Opción 1: Si la máscara tiene menos slices, rellenar con ceros
            if mask_shape[2] < img_shape[2]:
                padding = img_shape[2] - mask_shape[2]
                zeros_pad = np.zeros((mask_shape[0], mask_shape[1], padding), dtype=np.uint8)
                mask_volume = np.concatenate([mask_volume, zeros_pad], axis=2)
                print(f"      ✅ Máscara rellenada con {padding} slices de ceros")
            
            # Opción 2: Si la máscara tiene más slices, recortar
            elif mask_shape[2] > img_shape[2]:
                mask_volume = mask_volume[:, :, :img_shape[2]]
                print(f"      ✅ Máscara recortada a {img_shape[2]} slices")
            
            # Verificar dimensiones H y W
            if mask_shape[0] != img_shape[0] or mask_shape[1] != img_shape[1]:
                from scipy.ndimage import zoom
                zoom_factors = (
                    img_shape[0] / mask_volume.shape[0],
                    img_shape[1] / mask_volume.shape[1],
                    1.0  # No cambiar profundidad
                )
                mask_volume = zoom(mask_volume, zoom_factors, order=0)  # order=0 para nearest neighbor
                print(f"      ✅ Máscara redimensionada a {img_shape}")
        
        # Crear NIfTI de máscara con el mismo affine y header
        mask_nifti = nib.Nifti1Image(mask_volume, affine=img_affine, header=img_header)
        
        # Guardar máscara alineada
        mask_output_path = os.path.join(labels_output, f"{patient_id}.nii.gz")
        nib.save(mask_nifti, mask_output_path)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error procesando {patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def create_dataset_json(dataset_path, num_train, num_test):
    """Crea el archivo dataset.json con la metadata del dataset."""
    print("📝 Creando dataset.json...")
    
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": num_train,
        "numTest": num_test,
        "file_ending": ".nii.gz",
        "dataset_name": DATASET_NAME,
        "dataset_description": "Ischemic stroke lesion segmentation from CT scans"
    }
    
    json_path = os.path.join(dataset_path, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"   ✅ dataset.json creado: {json_path}")
    print(f"   📊 Entrenamiento: {num_train} casos")
    print(f"   🧪 Prueba: {num_test} casos\n")


# ============================================================================
# FASE 2: VERIFICACIÓN MANUAL
# ============================================================================

def pause_for_verification(dataset_path):
    """Sección informativa (verificación manual previamente realizada, no bloquea)."""
    print("=" * 80)
    print("ℹ️  VERIFICACIÓN MANUAL (OMITIDA) ")
    print("=" * 80)
    print()
    print("La verificación manual del alineamiento ya fue realizada anteriormente.")
    print("Este paso ahora es no interactivo para permitir ejecución en 2º plano.")
    print()
    print(f"📁 Imágenes de entrenamiento: {os.path.join(dataset_path, 'imagesTr')}")
    print(f"📁 Máscaras de entrenamiento: {os.path.join(dataset_path, 'labelsTr')}")
    print()
    print("Si necesitas revisar nuevamente, abre un visor (ITK-SNAP / 3D Slicer) manualmente.")
    print("Continuando automáticamente con el pipeline...\n")


# ============================================================================
# FASES 3 Y 4: EJECUCIÓN DE NNUNET
# ============================================================================

def run_command(command_list, description=""):
    """Ejecuta un comando de shell y verifica errores."""
    if description:
        print(f"\n{'=' * 80}")
        print(f"🚀 {description}")
        print(f"{'=' * 80}")
        print(f"Comando: {' '.join(command_list)}\n")
    
    try:
        result = subprocess.run(
            command_list,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        print(result.stdout)
        print(f"\n✅ {description} - COMPLETADO\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error ejecutando: {description}")
        print(f"Código de salida: {e.returncode}")
        print(f"Salida:\n{e.stdout}")
        return False


def run_nnunet_pipeline(dataset_path):
    """Ejecuta el pipeline completo de nnU-Net."""
    
    images_ts_path = os.path.join(dataset_path, "imagesTs")
    labels_ts_path = os.path.join(dataset_path, "labelsTs")
    predict_output_path = os.path.join(dataset_path, "predictions")
    eval_output_path = os.path.join(dataset_path, "evaluation")
    
    os.makedirs(predict_output_path, exist_ok=True)
    os.makedirs(eval_output_path, exist_ok=True)
    
    # ========================================================================
    # COMANDO 1: Plan and Preprocess
    # ========================================================================
    success = run_command(
        ['nnUNetv2_plan_and_preprocess', '-d', DATASET_ID, '--verify_dataset_integrity'],
        "PASO 1/4: Planificación y preprocesamiento"
    )
    
    if not success:
        print("❌ Fallo en la planificación. Abortando pipeline.")
        return False
    
    # ========================================================================
    # COMANDO 2: Training
    # ========================================================================
    success = run_command(
        ['nnUNetv2_train', DATASET_ID, '3d_fullres', 'all'],
        "PASO 2/4: Entrenamiento del modelo"
    )
    
    if not success:
        print("❌ Fallo en el entrenamiento. Abortando pipeline.")
        return False
    
    # ========================================================================
    # COMANDO 3: Prediction
    # ========================================================================
    success = run_command(
        ['nnUNetv2_predict', 
         '-i', images_ts_path,
         '-o', predict_output_path,
         '-d', DATASET_ID,
         '-c', '3d_fullres',
         '-f', 'all'],
        "PASO 3/4: Predicción en set de prueba"
    )
    
    if not success:
        print("❌ Fallo en la predicción. Abortando pipeline.")
        return False
    
    # ========================================================================
    # COMANDO 4: Evaluation (delegado a función dedicada)
    # ========================================================================
    success = run_evaluation(dataset_path)
    if not success:
        print("❌ Fallo en la evaluación.")
        return False

    return True


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def _case_id_from_path(p: str) -> str:
    name = Path(p).name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return Path(p).stem


def _compute_binary_metrics(gt: np.ndarray, pred: np.ndarray):
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    gt_sum = int(gt.sum())
    pred_sum = int(pred.sum())

    # Ambos vacíos: perfecto por convención
    if gt_sum == 0 and pred_sum == 0:
        return dict(dice=1.0, f1=1.0, precision=1.0, recall=1.0,
                    tp=0, fp=0, fn=0, gt_voxels=0, pred_voxels=0)

    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())

    denom_dice = 2 * tp + fp + fn
    dice = (2 * tp / denom_dice) if denom_dice > 0 else 0.0

    prec_denom = tp + fp
    if prec_denom > 0:
        precision = tp / prec_denom
    else:
        precision = 1.0 if pred_sum == 0 and gt_sum == 0 else 0.0

    rec_denom = tp + fn
    if rec_denom > 0:
        recall = tp / rec_denom
    else:
        recall = 1.0 if gt_sum == 0 else 0.0

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return dict(dice=float(dice), f1=float(f1), precision=float(precision), recall=float(recall),
                tp=tp, fp=fp, fn=fn, gt_voxels=gt_sum, pred_voxels=pred_sum)


def compute_and_save_metrics(labels_dir: str, preds_dir: str, out_dir: str, config: str = "default"):
    print("\n🧮 Calculando métricas adicionales (Dice, F1, Recall, Precision)...")
    os.makedirs(out_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.nii*')))
    per_case = []

    for gt_path in label_files:
        cid = _case_id_from_path(gt_path)
        # Predicción esperada: mismo id que label
        pred_path = os.path.join(preds_dir, f"{cid}.nii.gz")
        if not os.path.exists(pred_path):
            # fallback: .nii
            alt_path = os.path.join(preds_dir, f"{cid}.nii")
            if os.path.exists(alt_path):
                pred_path = alt_path
            else:
                # último recurso: glob por prefijo
                candidates = sorted(glob.glob(os.path.join(preds_dir, f"{cid}*.nii*")))
                if candidates:
                    pred_path = candidates[0]
                else:
                    print(f"   ⚠️  Predicción no encontrada para {cid}, se omite")
                    continue

        gt_img = nib.load(gt_path)
        pred_img = nib.load(pred_path)
        gt_arr = np.asanyarray(gt_img.dataobj)
        pred_arr = np.asanyarray(pred_img.dataobj)

        # Alinear shape si difiere (raro; nnU-Net debería coincidir)
        if gt_arr.shape != pred_arr.shape:
            print(f"   ⚠️  Shape mismatch {cid}: gt {gt_arr.shape} vs pred {pred_arr.shape}. Ajustando...")
            # Recortar o pad a tamaño de gt
            pad_z = gt_arr.shape[2] - pred_arr.shape[2]
            arr = pred_arr
            if pad_z > 0:
                arr = np.pad(arr, ((0,0),(0,0),(0,pad_z)), mode='constant')
            elif pad_z < 0:
                arr = arr[:, :, :gt_arr.shape[2]]
            pred_arr = arr
            # Ajuste H/W si necesario
            if gt_arr.shape[:2] != pred_arr.shape[:2]:
                from scipy.ndimage import zoom
                zoom_factors = (gt_arr.shape[0]/pred_arr.shape[0], gt_arr.shape[1]/pred_arr.shape[1], 1.0)
                pred_arr = zoom(pred_arr, zoom_factors, order=0)

        m = _compute_binary_metrics(gt_arr, pred_arr)
        m['case'] = cid
        per_case.append(m)

    if not per_case:
        print("   ⚠️  No se calcularon métricas (no hay pares gt/pred).")
        return

    # Resumen
    mean_dice = float(np.mean([x['dice'] for x in per_case]))
    mean_f1 = float(np.mean([x['f1'] for x in per_case]))
    mean_prec = float(np.mean([x['precision'] for x in per_case]))
    mean_rec = float(np.mean([x['recall'] for x in per_case]))

    summary = dict(
        mean_dice=mean_dice,
        mean_f1=mean_f1,
        mean_precision=mean_prec,
        mean_recall=mean_rec,
        num_cases=len(per_case)
    )

    # Guardar JSON
    out_json = os.path.join(out_dir, f'custom_metrics_{config}.json')
    with open(out_json, 'w') as f:
        json.dump({'summary': summary, 'per_case': per_case}, f, indent=2)
    print(f"   ✅ Métricas guardadas en {out_json}")

    # Guardar CSV por caso
    out_csv = os.path.join(out_dir, f'custom_metrics_per_case_{config}.csv')
    try:
        import csv as _csv
        keys = ['case','dice','f1','precision','recall','tp','fp','fn','gt_voxels','pred_voxels']
        with open(out_csv, 'w', newline='') as cf:
            w = _csv.DictWriter(cf, fieldnames=keys)
            w.writeheader()
            for row in per_case:
                w.writerow({k: row.get(k, '') for k in keys})
        print(f"   ✅ CSV por caso guardado en {out_csv}")
    except Exception as e:
        print(f"   ⚠️  No se pudo escribir CSV: {e}")
    
    return summary


def run_evaluation(dataset_path: str) -> bool:
    """Ejecuta la evaluación de nnU-Net (oficial) y luego calcula métricas pedidas."""
    labels_ts_path = os.path.join(dataset_path, "labelsTs")
    predict_output_path = os.path.join(dataset_path, "predictions")
    eval_output_path = os.path.join(dataset_path, "evaluation")
    return run_evaluation_for_config(dataset_path, predict_output_path, eval_output_path, "3d_fullres")


def run_evaluation_for_config(dataset_path: str, predict_output_path: str, eval_output_path: str, config: str) -> bool:
    """Ejecuta evaluación para una configuración específica."""
    labels_ts_path = os.path.join(dataset_path, "labelsTs")
    os.makedirs(eval_output_path, exist_ok=True)

    # Rutas necesarias para la evaluación en nnU-Net v2
    dataset_json_path = os.path.join(dataset_path, 'dataset.json')
    nnunet_preprocessed = os.environ.get('nnUNet_preprocessed') or os.environ.get('nnUNET_preprocessed')
    plans_file_path = os.path.join(
        nnunet_preprocessed,
        f"Dataset{DATASET_ID}_{DATASET_NAME}",
        'nnUNetPlans.json'
    )

    # nnUNetv2_evaluate_folder requiere: gt_folder pred_folder -djfile -pfile [-o]
    # nnUNetv2_evaluate_folder espera un archivo JSON como salida (-o <file.json>)
    summary_json_path = os.path.join(eval_output_path, f'nnunet_eval_summary_{config}.json')
    ok = run_command(
        ['nnUNetv2_evaluate_folder',
         labels_ts_path,
         predict_output_path,
         '-djfile', dataset_json_path,
         '-pfile', plans_file_path,
         '-o', summary_json_path],
        f"PASO 4/4: Evaluación de resultados ({config})"
    )
    if not ok:
        return False

    # Métricas personalizadas (Dice, F1, Precision, Recall)
    try:
        # Intentar MONAI primero
        monai_summary = None
        try:
            monai_summary = compute_and_save_metrics_monai(labels_ts_path, predict_output_path, eval_output_path, config)
        except ImportError as _e:
            print(f"   ℹ️  MONAI no disponible, usando métricas internas. Detalle: {_e}")
        except Exception as _e:
            print(f"   ⚠️  Error con métricas MONAI, usando fallback interno. Detalle: {_e}")

        if monai_summary:
            print(f"\n📌 Resumen de métricas ({config} - MONAI):")
            print(f"   Dice medio:      {monai_summary['mean_dice']:.4f}")
            print(f"   F1 medio:        {monai_summary['mean_f1']:.4f}")
            print(f"   Precision media: {monai_summary['mean_precision']:.4f}")
            print(f"   Recall medio:    {monai_summary['mean_recall']:.4f}\n")
        else:
            summary = compute_and_save_metrics(labels_ts_path, predict_output_path, eval_output_path, config)
            if summary:
                print(f"\n📌 Resumen de métricas ({config} - custom):")
                print(f"   Dice medio:      {summary['mean_dice']:.4f}")
                print(f"   F1 medio:        {summary['mean_f1']:.4f}")
                print(f"   Precision media: {summary['mean_precision']:.4f}")
                print(f"   Recall medio:    {summary['mean_recall']:.4f}\n")
    except Exception as e:
        print(f"⚠️  Error calculando métricas adicionales: {e}")
        import traceback; traceback.print_exc()

    return True


def compute_and_save_metrics_monai(labels_dir: str, preds_dir: str, out_dir: str, config: str = "default"):
    """Calcula Dice, Precision, Recall y F1 usando MONAI (solo binario) por caso y promedio.
    Corrige el error de canal esperado en MONAI utilizando:
      - DiceMetric con tensors one-hot
      - Cálculo manual TP/FP/FN para precision/recall/F1
    Devuelve summary dict o None si no hay pares.
    """
    try:
        from monai.metrics import DiceMetric
        from monai.networks.utils import one_hot
    except Exception as e:
        raise ImportError(f"MONAI no disponible para métricas: {e}")

    print(f"\n🧮 Calculando métricas MONAI (Dice, Precision, Recall, F1) (modo binario - {config})...")
    os.makedirs(out_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.nii*')))
    per_case = []

    dice_metric = DiceMetric(include_background=False, reduction="none")

    for gt_path in label_files:
        cid = _case_id_from_path(gt_path)
        pred_path = os.path.join(preds_dir, f"{cid}.nii.gz")
        if not os.path.exists(pred_path):
            alt_path = os.path.join(preds_dir, f"{cid}.nii")
            if os.path.exists(alt_path):
                pred_path = alt_path
            else:
                candidates = sorted(glob.glob(os.path.join(preds_dir, f"{cid}*.nii*")))
                if candidates:
                    pred_path = candidates[0]
                else:
                    print(f"   ⚠️  Predicción no encontrada para {cid}, se omite")
                    continue

        gt_arr = np.asanyarray(nib.load(gt_path).dataobj).astype(np.int64)
        pred_arr = np.asanyarray(nib.load(pred_path).dataobj).astype(np.int64)

        # Ajuste simple de profundidad si difiere
        if gt_arr.shape != pred_arr.shape:
            pad_z = gt_arr.shape[2] - pred_arr.shape[2]
            arr = pred_arr
            if pad_z > 0:
                arr = np.pad(arr, ((0,0),(0,0),(0,pad_z)), mode='constant')
            elif pad_z < 0:
                arr = arr[:, :, :gt_arr.shape[2]]
            pred_arr = arr

        # Binarizar explícitamente
        gt_bin = (gt_arr > 0).astype(np.int64)
        pred_bin = (pred_arr > 0).astype(np.int64)

        # Preparar para Dice (one-hot)
        gt_t = torch.from_numpy(gt_bin)[None]  # [1, H, W, D]
        pred_t = torch.from_numpy(pred_bin)[None]
        gt_oh = one_hot(gt_t, num_classes=2).float()  # [1,2,...]
        pred_oh = one_hot(pred_t, num_classes=2).float()

        d = dice_metric(pred_oh, gt_oh)  # por clase (solo lesión porque include_background=False)
        dice_val = float(d.mean().item())

        # TP, FP, FN para precision/recall/f1
        tp = int(((pred_bin == 1) & (gt_bin == 1)).sum())
        fp = int(((pred_bin == 1) & (gt_bin == 0)).sum())
        fn = int(((pred_bin == 0) & (gt_bin == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_case.append({
            'case': cid,
            'dice': dice_val,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': tp,
            'fp': fp,
            'fn': fn
        })

    if not per_case:
        print("   ⚠️  No se calcularon métricas MONAI (no hay pares gt/pred).")
        return None

    mean_dice = float(np.mean([x['dice'] for x in per_case]))
    mean_f1 = float(np.mean([x['f1'] for x in per_case]))
    mean_prec = float(np.mean([x['precision'] for x in per_case]))
    mean_rec = float(np.mean([x['recall'] for x in per_case]))

    summary = dict(
        mean_dice=mean_dice,
        mean_f1=mean_f1,
        mean_precision=mean_prec,
        mean_recall=mean_rec,
        num_cases=len(per_case),
        backend="monai"
    )

    out_json = os.path.join(out_dir, f'monai_metrics_{config}.json')
    with open(out_json, 'w') as f:
        json.dump({'summary': summary, 'per_case': per_case}, f, indent=2)
    print(f"   ✅ Métricas MONAI guardadas en {out_json}")

    return summary


def main():
    """Orquesta el pipeline con selección de etapas."""
    parser = argparse.ArgumentParser(description='Pipeline nnU-Net v2 (AISD) con etapas seleccionables')
    parser.add_argument('--start-stage', type=int, default=0, choices=[0,1,2,3,4],
                        help='Etapa inicial: 0=prep datos, 1=plan, 2=train, 3=predict, 4=evaluate')
    parser.add_argument('--end-stage', type=int, default=4, choices=[0,1,2,3,4],
                        help='Etapa final (inclusive)')
    parser.add_argument('--configurations', type=str, nargs='+', default=['3d_fullres'],
                        choices=['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres', 'ensemble'],
                        help='Configuraciones a entrenar/predecir/evaluar (ej: 2d 3d_fullres ensemble)')
    args = parser.parse_args()

    start_stage = args.start_stage
    end_stage = args.end_stage
    configurations = args.configurations
    if end_stage < start_stage:
        end_stage = start_stage

    print("\n" + "=" * 80)
    print("🧠 PIPELINE AUTOMATIZADO nnU-Net v2 - SEGMENTACIÓN DE ACV ISQUÉMICO")
    print(f"Configuraciones: {', '.join(configurations)}")
    print("=" * 80 + "\n")

    # Verificar entorno
    verify_environment()

    # dataset_path fijo a partir de nnUNet_raw
    dataset_path = os.path.join(NNUNET_RAW_PATH, f"Dataset{DATASET_ID}_{DATASET_NAME}")

    # Etapa 0: Preparación de datos
    if start_stage <= 0 <= end_stage:
        # Crear directorios
        dataset_path = setup_directories()

        # Obtener listas de pacientes
        train_patients, test_patients = get_patient_lists()

        # Procesar pacientes de entrenamiento
        print("🔄 Procesando pacientes de ENTRENAMIENTO...")
        train_success = 0
        for patient_id in tqdm(train_patients, desc="Entrenamiento"):
            if process_patient(patient_id, "Tr", dataset_path):
                train_success += 1

        print(f"\n✅ Entrenamiento: {train_success}/{len(train_patients)} pacientes procesados correctamente\n")

        # Procesar pacientes de prueba
        print("🔄 Procesando pacientes de PRUEBA...")
        test_success = 0
        for patient_id in tqdm(test_patients, desc="Prueba"):
            if process_patient(patient_id, "Ts", dataset_path):
                test_success += 1

        print(f"\n✅ Prueba: {test_success}/{len(test_patients)} pacientes procesados correctamente\n")

        # Crear dataset.json
        create_dataset_json(dataset_path, train_success, test_success)

        # Pausa informativa
        pause_for_verification(dataset_path)

    # Etapa 1: Plan and Preprocess
    if start_stage <= 1 <= end_stage:
        ok = run_command(
            ['nnUNetv2_plan_and_preprocess', '-d', DATASET_ID, '--verify_dataset_integrity'],
            "PASO 1/4: Planificación y preprocesamiento"
        )
        if not ok:
            print("❌ Fallo en la planificación. Abortando.")
            return

    # Etapa 2: Training
    if start_stage <= 2 <= end_stage:
        # Filtrar 'ensemble' ya que no se entrena, solo se crea a partir de otros modelos
        train_configs = [c for c in configurations if c != 'ensemble']
        for config in train_configs:
            print(f"\n🎯 Entrenando configuración: {config}")
            ok = run_command(
                ['nnUNetv2_train', DATASET_ID, config, 'all'],
                f"PASO 2/4: Entrenamiento del modelo ({config})"
            )
            if not ok:
                print(f"❌ Fallo en el entrenamiento de {config}. Abortando.")
                return

    # Etapa 3: Prediction
    if start_stage <= 3 <= end_stage:
        images_ts_path = os.path.join(dataset_path, "imagesTs")
        
        # Predicciones individuales
        predict_configs = [c for c in configurations if c != 'ensemble']
        for config in predict_configs:
            print(f"\n🔮 Prediciendo con configuración: {config}")
            predict_output_path = os.path.join(dataset_path, f"predictions_{config}")
            os.makedirs(predict_output_path, exist_ok=True)
            ok = run_command(
                ['nnUNetv2_predict', '-i', images_ts_path, '-o', predict_output_path, '-d', DATASET_ID, '-c', config, '-f', 'all'],
                f"PASO 3/4: Predicción en set de prueba ({config})"
            )
            if not ok:
                print(f"❌ Fallo en la predicción de {config}. Abortando.")
                return
        
        # Si se pidió ensemble, crearlo combinando las predicciones disponibles
        if 'ensemble' in configurations:
            print(f"\n🔮 Creando predicciones de ensemble...")
            predict_output_path = os.path.join(dataset_path, "predictions_ensemble")
            os.makedirs(predict_output_path, exist_ok=True)

            # Buscar todas las carpetas de predicciones disponibles (excepto ensemble)
            available_predictions = []
            for config in ['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres']:
                pred_path = os.path.join(dataset_path, f"predictions_{config}")
                if os.path.exists(pred_path) and os.listdir(pred_path):
                    available_predictions.append(pred_path)

            # Compatibilidad con nombre legado: 'predictions' == 'predictions_3d_fullres'
            legacy_pred = os.path.join(dataset_path, "predictions")
            if os.path.exists(legacy_pred) and os.listdir(legacy_pred):
                available_predictions.append(legacy_pred)

            # Quitar duplicados preservando orden
            seen = set()
            available_predictions = [p for p in available_predictions if not (p in seen or seen.add(p))]

            if len(available_predictions) < 2:
                # Con una sola fuente, copiar como predicción de ensemble para permitir evaluación
                print(f"⚠️  Advertencia: Se encontraron solo {len(available_predictions)} predicciones. El ensemble real requiere al menos 2.")
                print(f"    Carpetas encontradas: {available_predictions}")
                if len(available_predictions) == 1:
                    print("    Creando 'predictions_ensemble' como copia de la única fuente disponible para poder evaluar...")
                    src = available_predictions[0]
                    # Copiar/actualizar solo archivos .nii.gz
                    for f in glob.glob(os.path.join(src, '*.nii.gz')):
                        dst = os.path.join(predict_output_path, os.path.basename(f))
                        if not os.path.exists(dst):
                            shutil.copy(f, dst)
                else:
                    print("    No hay predicciones disponibles para crear ensemble.")
            else:
                print(f"   Combinando predicciones de: {[os.path.basename(p) for p in available_predictions]}")
                # nnUNetv2_ensemble acepta múltiples directorios de entrada y crea el promedio
                ensemble_cmd = ['nnUNetv2_ensemble'] + ['-i'] + available_predictions + ['-o', predict_output_path]
                ok = run_command(
                    ensemble_cmd,
                    f"PASO 3/4: Creación de ensemble"
                )
                if not ok:
                    print(f"❌ Fallo en la creación del ensemble. Continuando...")

            # Completar predicciones faltantes respecto a labelsTs usando fuentes individuales (si existen)
            labels_ts_path = os.path.join(dataset_path, "labelsTs")
            gt_files = [os.path.basename(p) for p in glob.glob(os.path.join(labels_ts_path, '*.nii.gz'))]
            pred_files = [os.path.basename(p) for p in glob.glob(os.path.join(predict_output_path, '*.nii.gz'))]
            missing = sorted(set(gt_files) - set(pred_files))
            if missing:
                print(f"   Completando {len(missing)} casos faltantes en predictions_ensemble a partir de fuentes individuales...")
                source_dirs = available_predictions
                filled = 0
                for fname in missing:
                    placed = False
                    for src in source_dirs:
                        candidate = os.path.join(src, fname)
                        if os.path.exists(candidate):
                            shutil.copy(candidate, os.path.join(predict_output_path, fname))
                            filled += 1
                            placed = True
                            break
                    if not placed:
                        print(f"      ⚠️  No se encontró '{fname}' en fuentes: {[os.path.basename(s) for s in source_dirs]}")
                print(f"   Casos completados: {filled}")

    # Etapa 4: Evaluation + métricas
    if start_stage <= 4 <= end_stage:
        for config in configurations:
            print(f"\n📊 Evaluando configuración: {config}")
            predict_output_path = os.path.join(dataset_path, f"predictions_{config}")
            eval_output_path = os.path.join(dataset_path, f"evaluation_{config}")
            os.makedirs(eval_output_path, exist_ok=True)
            
            # Crear una función de evaluación que acepte paths custom
            ok = run_evaluation_for_config(dataset_path, predict_output_path, eval_output_path, config)
            if not ok:
                print(f"❌ Fallo en la evaluación de {config}.")
                # No abortar, continuar con otras configs

    # Mensaje final
    print("\n" + "=" * 80)
    print(f"🎉 PIPELINE COMPLETADO (etapas ejecutadas: {start_stage}-{end_stage})")
    print(f"Configuraciones procesadas: {', '.join(configurations)}")
    print("=" * 80)
    print(f"\n📊 Resultados de evaluación en:")
    for config in configurations:
        eval_path = os.path.join(dataset_path, f'evaluation_{config}')
        if os.path.exists(eval_path):
            print(f"   [{config}]: {eval_path}")
    print(f"\n📁 Predicciones:")
    for config in configurations:
        pred_path = os.path.join(dataset_path, f'predictions_{config}')
        if os.path.exists(pred_path):
            print(f"   [{config}]: {pred_path}")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
