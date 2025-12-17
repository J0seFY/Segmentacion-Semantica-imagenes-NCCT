#!/usr/bin/env python3
"""
Generación de volúmenes NIfTI y máscaras alineadas.
Produce la estructura Dataset<ID>_<NAME> y dataset.json para pipelines de
segmentación (compatible con nnU-Net u otros modelos).
"""

import os
import argparse
import glob
import json
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pydicom
from tqdm import tqdm

# ============================================================================
# CONFIGURACIÓN GLOBAL (puedes adaptar estas constantes)
# ============================================================================

DATASET_DICOM_ROOT = \
    "/home/jfuentes/Segmentacion-semantica-en-imagenes-biomedicas-de-ACV-isquemico-con-arquitecturas-U-Net/Dataset_dicom"
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
    """Verifica/crea las rutas nnUNet_* y comprueba dependencias básicas."""
    print("Verificando configuración del entorno...")

    nnunet_raw_path = os.environ.get('nnUNet_raw') or os.environ.get('nnUNET_raw')
    if nnunet_raw_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        nnunet_raw_path = os.path.join(project_root, 'nnUNet_raw')
        os.makedirs(nnunet_raw_path, exist_ok=True)
        os.environ['nnUNet_raw'] = nnunet_raw_path
        print(f"   nnUNet_raw no estaba definido. Configurado automáticamente a: {nnunet_raw_path}")
    else:
        os.makedirs(nnunet_raw_path, exist_ok=True)
    print(f"   nnUNet_raw: {nnunet_raw_path}")

    nnunet_preprocessed = os.environ.get('nnUNet_preprocessed') or os.environ.get('nnUNET_preprocessed')
    if nnunet_preprocessed is None:
        base_dir = os.path.dirname(nnunet_raw_path.rstrip('/')) or os.getcwd()
        nnunet_preprocessed = os.path.join(base_dir, 'nnUNet_preprocessed')
        os.makedirs(nnunet_preprocessed, exist_ok=True)
        os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
        print(f"   nnUNet_preprocessed no estaba definido. Configurado automáticamente a: {nnunet_preprocessed}")
    else:
        os.makedirs(nnunet_preprocessed, exist_ok=True)
    print(f"   nnUNet_preprocessed: {nnunet_preprocessed}")

    nnunet_results = os.environ.get('nnUNet_results') or os.environ.get('nnUNET_results')
    if nnunet_results is None:
        base_dir = os.path.dirname(nnunet_raw_path.rstrip('/')) or os.getcwd()
        nnunet_results = os.path.join(base_dir, 'nnUNet_results')
        os.makedirs(nnunet_results, exist_ok=True)
        os.environ['nnUNet_results'] = nnunet_results
        print(f"   nnUNet_results no estaba definido. Configurado automáticamente a: {nnunet_results}")
    else:
        os.makedirs(nnunet_results, exist_ok=True)
    print(f"   nnUNet_results: {nnunet_results}")

    try:
        import pydicom as _  # noqa: F401
        print("   pydicom está instalado")
    except ImportError:
        raise EnvironmentError("pydicom no está instalado.\nInstálalo con: pip install pydicom")

    if not os.path.exists(DATASET_DICOM_ROOT):
        raise FileNotFoundError(
            f"El dataset DICOM no existe en: {DATASET_DICOM_ROOT}\n"
            f"Por favor, verifica la ruta DATASET_DICOM_ROOT en el script."
        )
    print(f"   Dataset DICOM encontrado: {DATASET_DICOM_ROOT}")

    print("Todas las verificaciones pasaron correctamente.\n")

    return dict(
        nnunet_raw_path=nnunet_raw_path,
        nnunet_preprocessed=nnunet_preprocessed,
        nnunet_results=nnunet_results,
    )


# ============================================================================
# PREPARACIÓN DE DATOS
# ============================================================================

def setup_directories(nnunet_raw_path: str, dataset_id: str, dataset_name: str):
    """Crea la estructura de carpetas para nnU-Net y devuelve dataset_path."""
    print("Creando estructura de directorios para volúmenes...")

    dataset_path = os.path.join(nnunet_raw_path, f"Dataset{dataset_id}_{dataset_name}")
    dirs_to_create = [
        os.path.join(dataset_path, "imagesTr"),
        os.path.join(dataset_path, "labelsTr"),
        os.path.join(dataset_path, "imagesTs"),
        os.path.join(dataset_path, "labelsTs"),
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   creado: {dir_path}")

    print(f"Estructura creada en: {dataset_path}\n")
    return dataset_path


def get_patient_lists(dataset_dicom_root: str, fixed_test_patients):
    """Obtiene listas de pacientes de entrenamiento y prueba."""
    print("Escaneando pacientes en el dataset DICOM...")

    image_dir = os.path.join(dataset_dicom_root, "image")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"No se encuentra la carpeta: {image_dir}")

    all_patients = [p for p in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, p))]
    print(f"   Total de pacientes encontrados: {len(all_patients)}")

    test_patients = [p for p in all_patients if p in fixed_test_patients]
    train_patients = [p for p in all_patients if p not in fixed_test_patients]

    print(f"   Pacientes de entrenamiento: {len(train_patients)}")
    print(f"   Pacientes de prueba: {len(test_patients)}")

    if len(test_patients) != len(fixed_test_patients):
        missing = set(fixed_test_patients) - set(test_patients)
        if missing:
            print(f"   Advertencia: {len(missing)} pacientes de test no encontrados: {missing}")

    print()
    return train_patients, test_patients


def dicom_to_nifti(dicom_dir, output_path):
    """Convierte una serie DICOM a NIfTI usando pydicom."""
    dicom_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    if not dicom_files:
        return None

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

    def get_z_position(dcm):
        if hasattr(dcm, 'ImagePositionPatient'):
            return float(dcm.ImagePositionPatient[2])
        if hasattr(dcm, 'SliceLocation'):
            return float(dcm.SliceLocation)
        if hasattr(dcm, 'InstanceNumber'):
            return float(dcm.InstanceNumber)
        return 0.0

    slices = sorted(slices, key=get_z_position)

    pixel_arrays = []
    for ds in slices:
        pixel_array = ds.pixel_array.astype(np.float32)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        pixel_arrays.append(pixel_array)

    volume = np.stack(pixel_arrays, axis=-1)
    first_ds = slices[0]
    affine = np.eye(4)

    if hasattr(first_ds, 'ImageOrientationPatient') and hasattr(first_ds, 'ImagePositionPatient') and hasattr(first_ds, 'PixelSpacing'):
        iop = np.array(first_ds.ImageOrientationPatient).reshape(2, 3)
        row_cosines = iop[0]
        col_cosines = iop[1]
        slice_normal = np.cross(row_cosines, col_cosines)

        pixel_spacing = [float(x) for x in first_ds.PixelSpacing]
        row_spacing, col_spacing = pixel_spacing[0], pixel_spacing[1]

        if len(slices) > 1 and hasattr(slices[1], 'ImagePositionPatient'):
            pos0 = np.array(first_ds.ImagePositionPatient)
            pos1 = np.array(slices[1].ImagePositionPatient)
            slice_spacing = np.abs(np.dot(pos1 - pos0, slice_normal))
            if slice_spacing == 0:
                slice_spacing = float(getattr(first_ds, 'SliceThickness', 1.0))
        else:
            slice_spacing = float(getattr(first_ds, 'SliceThickness', 1.0))

        rotation_matrix = np.column_stack([
            col_cosines * col_spacing,
            row_cosines * row_spacing,
            slice_normal * slice_spacing
        ])

        affine[:3, :3] = rotation_matrix
        affine[:3, 3] = np.array(first_ds.ImagePositionPatient)
    else:
        pixel_spacing = getattr(first_ds, 'PixelSpacing', [1.0, 1.0])
        slice_thickness = getattr(first_ds, 'SliceThickness', 1.0)
        affine[0, 0] = float(pixel_spacing[1])
        affine[1, 1] = float(pixel_spacing[0])
        affine[2, 2] = float(slice_thickness)

    nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nifti_img, output_path)
    return nifti_img


def process_patient(patient_id: str, split: str, dataset_path: str, dataset_dicom_root: str, lesion_labels):
    """Convierte DICOM+PNG a NIfTI alineado para un paciente."""
    dicom_dir = os.path.join(dataset_dicom_root, "image", patient_id, "CT")
    mask_dir = os.path.join(dataset_dicom_root, "mask", patient_id)

    images_output = os.path.join(dataset_path, f"images{split}")
    labels_output = os.path.join(dataset_path, f"labels{split}")

    if not os.path.exists(dicom_dir):
        print(f"   DICOM no encontrado para {patient_id}, saltando...")
        return False
    if not os.path.exists(mask_dir):
        print(f"   Máscara no encontrada para {patient_id}, saltando...")
        return False

    try:
        target_nifti = os.path.join(images_output, f"{patient_id}_0000.nii.gz")
        nifti_img = dicom_to_nifti(dicom_dir, target_nifti)
        if nifti_img is None:
            print(f"   No se pudo convertir DICOM para {patient_id}")
            return False

        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        if not mask_files:
            print(f"   No se encontraron máscaras PNG para {patient_id}")
            return False

        mask_files = sorted(mask_files, key=lambda x: int(Path(x).stem))
        mask_slices = []
        for mask_file in mask_files:
            img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"   No se pudo leer {mask_file}")
                continue
            binary_mask = np.isin(img, lesion_labels).astype(np.uint8)
            mask_slices.append(binary_mask)

        if not mask_slices:
            print(f"   No se pudieron cargar máscaras para {patient_id}")
            return False

        mask_volume = np.stack(mask_slices, axis=-1)
        img_nifti = nib.load(target_nifti)
        img_affine = img_nifti.affine
        img_header = img_nifti.header

        img_shape = img_nifti.shape
        mask_shape = mask_volume.shape

        if img_shape != mask_shape:
            print(f"   Dimensiones no coinciden para {patient_id}:")
            print(f"      Imagen: {img_shape}, Máscara: {mask_shape}")

            if mask_shape[2] < img_shape[2]:
                padding = img_shape[2] - mask_shape[2]
                zeros_pad = np.zeros((mask_shape[0], mask_shape[1], padding), dtype=np.uint8)
                mask_volume = np.concatenate([mask_volume, zeros_pad], axis=2)
                print(f"      Máscara rellenada con {padding} slices de ceros")
            elif mask_shape[2] > img_shape[2]:
                mask_volume = mask_volume[:, :, :img_shape[2]]
                print(f"      Máscara recortada a {img_shape[2]} slices")

            if mask_volume.shape[0] != img_shape[0] or mask_volume.shape[1] != img_shape[1]:
                from scipy.ndimage import zoom
                zoom_factors = (
                    img_shape[0] / mask_volume.shape[0],
                    img_shape[1] / mask_volume.shape[1],
                    1.0
                )
                mask_volume = zoom(mask_volume, zoom_factors, order=0)
                print(f"      Máscara redimensionada a {img_shape}")

        mask_nifti = nib.Nifti1Image(mask_volume, affine=img_affine, header=img_header)
        mask_output_path = os.path.join(labels_output, f"{patient_id}.nii.gz")
        nib.save(mask_nifti, mask_output_path)
        return True

    except Exception as e:  # pragma: no cover - logging de error
        print(f"   Error procesando {patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def create_dataset_json(dataset_path: str, dataset_id: str, dataset_name: str, num_train: int, num_test: int):
    """Crea dataset.json con la metadata del dataset."""
    print("Creando dataset.json...")

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": num_train,
        "numTest": num_test,
        "file_ending": ".nii.gz",
        "dataset_name": dataset_name,
        "dataset_description": "Ischemic stroke lesion segmentation from CT scans",
    }

    json_path = os.path.join(dataset_path, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)

    print(f"   dataset.json creado: {json_path}")
    print(f"   Entrenamiento: {num_train} casos")
    print(f"   Prueba: {num_test} casos\n")


def pause_for_verification(dataset_path):
    """Muestra rutas para revisión manual (no interactivo)."""
    print("=" * 80)
    print("VERIFICACIÓN MANUAL (OMITIDA)")
    print("=" * 80)
    print()
    print("La verificación manual del alineamiento ya fue realizada anteriormente.")
    print("Este paso ahora es no interactivo para permitir ejecución en 2º plano.")
    print()
    print(f"Imágenes de entrenamiento: {os.path.join(dataset_path, 'imagesTr')}")
    print(f"Máscaras de entrenamiento: {os.path.join(dataset_path, 'labelsTr')}")
    print()
    print("Si necesitas revisar nuevamente, abre un visor (ITK-SNAP / 3D Slicer) manualmente.")
    print("Continuando automáticamente con el pipeline...\n")


# ============================================================================
# ORQUESTACIÓN DE PREPROCESAMIENTO
# ============================================================================

def run_preprocessing(
    dataset_id: str = DATASET_ID,
    dataset_name: str = DATASET_NAME,
    dataset_dicom_root: str = DATASET_DICOM_ROOT,
    lesion_labels=LESION_LABELS,
    fixed_test_patients=FIXED_TEST_PATIENTS,
    env_paths=None,
):
    """Ejecuta el flujo de preprocesamiento completo y devuelve dataset_path."""
    env_paths = env_paths or verify_environment()
    dataset_path = setup_directories(env_paths["nnunet_raw_path"], dataset_id, dataset_name)

    train_patients, test_patients = get_patient_lists(dataset_dicom_root, fixed_test_patients)

    print("Procesando pacientes de ENTRENAMIENTO...")
    train_success = 0
    for patient_id in tqdm(train_patients, desc="Entrenamiento"):
        if process_patient(patient_id, "Tr", dataset_path, dataset_dicom_root, lesion_labels):
            train_success += 1

    print(f"\nEntrenamiento: {train_success}/{len(train_patients)} pacientes procesados correctamente\n")

    print("Procesando pacientes de PRUEBA...")
    test_success = 0
    for patient_id in tqdm(test_patients, desc="Prueba"):
        if process_patient(patient_id, "Ts", dataset_path, dataset_dicom_root, lesion_labels):
            test_success += 1

    print(f"\nPrueba: {test_success}/{len(test_patients)} pacientes procesados correctamente\n")

    create_dataset_json(dataset_path, dataset_id, dataset_name, train_success, test_success)
    pause_for_verification(dataset_path)
    return dataset_path


# ============================================================================
# CLI
# ============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(description='Preprocesamiento nnU-Net v2 (AISD)')
    parser.add_argument('--dataset-id', default=DATASET_ID, help='ID del dataset (nnU-Net)')
    parser.add_argument('--dataset-name', default=DATASET_NAME, help='Nombre del dataset')
    parser.add_argument('--dicom-root', default=DATASET_DICOM_ROOT, help='Raíz del dataset DICOM (image/mask)')
    return parser.parse_args()


def main():
    args = _parse_args()
    run_preprocessing(
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        dataset_dicom_root=args.dicom_root,
    )


if __name__ == '__main__':
    main()
