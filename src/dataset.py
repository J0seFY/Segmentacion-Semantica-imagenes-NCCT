"""
Dataset utilities for AISD thesis experiments.

Funciones clave:
- NiftiSliceDataset: carga volúmenes 3D y expone slices 2D.
- get_patient_id: normaliza el ID de paciente desde nombres nnUNet.
- split_by_patient_ids: construye split train/val/test sin leakage usando estructura nnUNet (imagesTr/imagesTs).
"""

import os
import glob
from typing import List, Tuple, Dict

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, RandFlipd, RandRotated, RandZoomd,
    RandGaussianNoised, EnsureTyped
)

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


def get_patient_id(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith('.nii.gz'):
        base = base[:-7]
    if base.endswith('_0000'):
        base = base[:-5]
    return base


class NiftiSliceDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        ignore_empty: bool = False,
        is_training: bool = False,
        transform=None
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.ignore_empty = ignore_empty
        self.is_training = is_training
        self.transform = transform
        self.slice_index = self._build_slice_index()
        print(f"Dataset creado: {len(self.slice_index)} slices de {len(self.image_paths)} volúmenes")

    def _build_slice_index(self) -> List[Tuple[int, int]]:
        index = []
        for v_idx, (img_p, mask_p) in enumerate(zip(self.image_paths, self.mask_paths)):
            img_nii = nib.load(img_p)
            num_slices = img_nii.shape[2]
            if self.ignore_empty:
                mask_nii = nib.load(mask_p)
                mask_data = np.asanyarray(mask_nii.dataobj)
                for s in range(num_slices):
                    if mask_data[:, :, s].sum() > 0:
                        index.append((v_idx, s))
            else:
                for s in range(num_slices):
                    index.append((v_idx, s))
        return index

    def _preprocess(self, slice_img: np.ndarray) -> np.ndarray:
        """
        Preprocessing pipeline:
        1. HU windowing [15, 40] (acute stroke detection range)
        2. Min-max normalization [0, 1]
        """
        # Step 1: HU windowing
        lo, hi = 15, 40
        windowed = np.clip(slice_img, lo, hi)
        
        # Step 2: Min-max normalization
        normalized = (windowed - lo) / (hi - lo)
        
        return normalized.astype(np.float32)

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx, slice_idx = self.slice_index[idx]
        img_nii = nib.load(self.image_paths[vol_idx])
        mask_nii = nib.load(self.mask_paths[vol_idx])
        img_vol = np.asanyarray(img_nii.dataobj).astype(np.float32)
        mask_vol = np.asanyarray(mask_nii.dataobj).astype(np.uint8)
        img_slice = img_vol[:, :, slice_idx]
        mask_slice = mask_vol[:, :, slice_idx]
        
        # Apply preprocessing: HU window [15,40] + min-max [0,1]
        img_slice = self._preprocess(img_slice)
        mask_slice = (mask_slice > 0).astype(np.uint8)
        img_slice = img_slice[None, ...]
        mask_slice = mask_slice[None, ...]
        if self.transform and self.is_training:
            data = {'image': img_slice, 'mask': mask_slice}
            data = self.transform(data)
            img_t = data['image']; mask_t = data['mask']
        else:
            img_t = torch.from_numpy(img_slice)
            mask_t = torch.from_numpy(mask_slice)
        return {'image': img_t, 'mask': mask_t, 'volume_idx': vol_idx, 'slice_idx': slice_idx}


def get_augmentation_transforms(is_training: bool = True):
    if not is_training:
        return None
    return Compose([
        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1),
        RandRotated(keys=['image', 'mask'], prob=0.5, range_x=0.35, mode=['bilinear', 'nearest']),
        RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.9, max_zoom=1.1, mode=['trilinear', 'nearest']),
        RandGaussianNoised(keys=['image'], prob=0.15, mean=0.0, std=0.01),
        EnsureTyped(keys=['image', 'mask'])
    ])


def split_by_patient_ids(
    images_tr_dir: str,
    labels_tr_dir: str,
    images_ts_dir: str,
    labels_ts_dir: str,
    test_patient_ids: List[str] = FIXED_TEST_PATIENTS,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[str]]:
    np.random.seed(seed)
    # Test set
    test_imgs = sorted(glob.glob(os.path.join(images_ts_dir, '*.nii.gz')))
    test_msks = sorted(glob.glob(os.path.join(labels_ts_dir, '*.nii.gz')))
    test_img_map = {get_patient_id(f): f for f in test_imgs}
    test_msk_map = {get_patient_id(f): f for f in test_msks}
    missing_test = [p for p in test_patient_ids if p not in test_img_map]
    if missing_test:
        print(f"⚠️  Faltan {len(missing_test)} pacientes test en imagesTs: {missing_test[:20]} ...")
    else:
        print("✅ Todos los pacientes test presentes en imagesTs")
    test_images, test_masks = [], []
    for pid in test_patient_ids:
        img = test_img_map.get(pid); msk = test_msk_map.get(pid)
        if img and msk:
            test_images.append(img); test_masks.append(msk)
        elif img and not msk:
            print(f"⚠️  Máscara faltante para test {pid}")
    # Train/Val
    tr_imgs = sorted(glob.glob(os.path.join(images_tr_dir, '*.nii.gz')))
    tr_msks = sorted(glob.glob(os.path.join(labels_tr_dir, '*.nii.gz')))
    tr_img_map = {get_patient_id(f): f for f in tr_imgs}
    tr_msk_map = {get_patient_id(f): f for f in tr_msks}
    trainval_patients = [p for p in tr_img_map.keys() if p not in test_patient_ids]
    print(f"Pacientes candidatos train+val: {len(trainval_patients)}")
    missing_masks_trainval = [p for p in trainval_patients if p not in tr_msk_map]
    if missing_masks_trainval:
        print(f"⚠️  {len(missing_masks_trainval)} pacientes sin máscara (ejemplo): {missing_masks_trainval[:20]}")
    np.random.shuffle(trainval_patients)
    split_idx = int(len(trainval_patients) * (1 - val_ratio))
    train_patients = trainval_patients[:split_idx]
    val_patients = trainval_patients[split_idx:]
    train_images = [tr_img_map[p] for p in train_patients if p in tr_img_map]
    train_masks = [tr_msk_map[p] for p in train_patients if p in tr_msk_map]
    val_images = [tr_img_map[p] for p in val_patients if p in tr_img_map]
    val_masks = [tr_msk_map[p] for p in val_patients if p in tr_msk_map]
    print(f"Split -> Train:{len(train_patients)} Val:{len(val_patients)} Test:{len(test_images)}")
    return {
        'train_images': train_images,
        'train_masks': train_masks,
        'val_images': val_images,
        'val_masks': val_masks,
        'test_images': test_images,
        'test_masks': test_masks
    }


def create_dataloaders(
    train_images: List[str],
    train_masks: List[str],
    val_images: List[str],
    val_masks: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    ignore_empty: bool = False
) -> Tuple[DataLoader, DataLoader]:
    tr_tf = get_augmentation_transforms(is_training=True)
    train_ds = NiftiSliceDataset(train_images, train_masks, ignore_empty, True, tr_tf)
    val_ds = NiftiSliceDataset(val_images, val_masks, ignore_empty, False, None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader


if __name__ == '__main__':
    print('dataset.py cargado')
    print(f'Pacientes test configurados: {len(FIXED_TEST_PATIENTS)}')
