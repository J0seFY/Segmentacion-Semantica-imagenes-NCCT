#!/usr/bin/env python3
"""
Script automatizado para entrenar y evaluar nnU-Net v2.
La generación de volúmenes (DICOM->NIfTI, máscaras, dataset.json) vive en scripts/generacion_volumenes_nifti.py.
"""

import os
import argparse
import subprocess
import json
import glob
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from generacion_volumenes_nifti import (
    DATASET_DICOM_ROOT,
    DATASET_ID,
    DATASET_NAME,
    FIXED_TEST_PATIENTS,
    LESION_LABELS,
    run_preprocessing,
    verify_environment,
)


# ============================================================================
# UTILIDADES NNUNET
# ============================================================================

def run_command(command_list, description=""):
    """Ejecuta un comando de shell y verifica errores."""
    if description:
        print(f"\n{'=' * 80}")
        print(f"{description}")
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
        print(f"\n{description} - COMPLETADO\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError ejecutando: {description}")
        print(f"Código de salida: {e.returncode}")
        print(f"Salida:\n{e.stdout}")
        return False


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
    print("\nCalculando métricas adicionales (Dice, F1, Recall, Precision)...")
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
                    print(f"   Predicción no encontrada para {cid}, se omite")
                    continue

        gt_img = nib.load(gt_path)
        pred_img = nib.load(pred_path)
        gt_arr = np.asanyarray(gt_img.dataobj)
        pred_arr = np.asanyarray(pred_img.dataobj)

        # Alinear shape si difiere (raro; nnU-Net debería coincidir)
        if gt_arr.shape != pred_arr.shape:
            print(f"   Shape mismatch {cid}: gt {gt_arr.shape} vs pred {pred_arr.shape}. Ajustando...")
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
        print("   No se calcularon métricas (no hay pares gt/pred).")
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
    print(f"   Métricas guardadas en {out_json}")

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
        print(f"   CSV por caso guardado en {out_csv}")
    except Exception as e:
        print(f"   No se pudo escribir CSV: {e}")
    
    return summary


def run_evaluation(dataset_path: str) -> bool:
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
            print(f"   MONAI no disponible, usando métricas internas. Detalle: {_e}")
        except Exception as _e:
            print(f"   Error con métricas MONAI, usando fallback interno. Detalle: {_e}")

        if monai_summary:
            print(f"\nResumen de métricas ({config} - MONAI):")
            print(f"   Dice medio:      {monai_summary['mean_dice']:.4f}")
            print(f"   F1 medio:        {monai_summary['mean_f1']:.4f}")
            print(f"   Precision media: {monai_summary['mean_precision']:.4f}")
            print(f"   Recall medio:    {monai_summary['mean_recall']:.4f}\n")
        else:
            summary = compute_and_save_metrics(labels_ts_path, predict_output_path, eval_output_path, config)
            if summary:
                print(f"\nResumen de métricas ({config} - custom):")
                print(f"   Dice medio:      {summary['mean_dice']:.4f}")
                print(f"   F1 medio:        {summary['mean_f1']:.4f}")
                print(f"   Precision media: {summary['mean_precision']:.4f}")
                print(f"   Recall medio:    {summary['mean_recall']:.4f}\n")
    except Exception as e:
        print(f"Error calculando métricas adicionales: {e}")
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

    print(f"\nCalculando métricas MONAI (Dice, Precision, Recall, F1) (modo binario - {config})...")
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
                    print(f"   Predicción no encontrada para {cid}, se omite")
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
        print("   No se calcularon métricas MONAI (no hay pares gt/pred).")
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
    print(f"   Métricas MONAI guardadas en {out_json}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Pipeline nnU-Net v2 (AISD) con etapas seleccionables')
    parser.add_argument('--start-stage', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='Etapa inicial: 0=prep datos, 1=plan, 2=train, 3=predict, 4=evaluate')
    parser.add_argument('--end-stage', type=int, default=4, choices=[0, 1, 2, 3, 4],
                        help='Etapa final (inclusive)')
    parser.add_argument('--configurations', type=str, nargs='+', default=['3d_fullres'],
                        choices=['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres', 'ensemble'],
                        help='Configuraciones a entrenar/predecir/evaluar (ej: 2d 3d_fullres ensemble)')
    args = parser.parse_args()

    start_stage = args.start_stage
    end_stage = args.end_stage if args.end_stage >= args.start_stage else args.start_stage
    configurations = args.configurations

    print("\n" + "=" * 80)
    print("PIPELINE AUTOMATIZADO nnU-Net v2 - SEGMENTACIÓN DE ACV ISQUÉMICO")
    print(f"Configuraciones: {', '.join(configurations)}")
    print("=" * 80 + "\n")

    env_paths = verify_environment()
    dataset_path = os.path.join(env_paths["nnunet_raw_path"], f"Dataset{DATASET_ID}_{DATASET_NAME}")

    if start_stage <= 0 <= end_stage:
        dataset_path = run_preprocessing(
            dataset_id=DATASET_ID,
            dataset_name=DATASET_NAME,
            dataset_dicom_root=DATASET_DICOM_ROOT,
            lesion_labels=LESION_LABELS,
            fixed_test_patients=FIXED_TEST_PATIENTS,
            env_paths=env_paths,
        )

    if start_stage <= 1 <= end_stage:
        ok = run_command(
            ['nnUNetv2_plan_and_preprocess', '-d', DATASET_ID, '--verify_dataset_integrity'],
            "PASO 1/4: Planificación y preprocesamiento"
        )
        if not ok:
            print("Fallo en la planificación. Abortando.")
            return

    if start_stage <= 2 <= end_stage:
        train_configs = [c for c in configurations if c != 'ensemble']
        for config in train_configs:
            print(f"\nEntrenando configuración: {config}")
            ok = run_command(
                ['nnUNetv2_train', DATASET_ID, config, 'all'],
                f"PASO 2/4: Entrenamiento del modelo ({config})"
            )
            if not ok:
                print(f"Fallo en el entrenamiento de {config}. Abortando.")
                return

    if start_stage <= 3 <= end_stage:
        images_ts_path = os.path.join(dataset_path, "imagesTs")

        predict_configs = [c for c in configurations if c != 'ensemble']
        for config in predict_configs:
            print(f"\nPrediciendo con configuración: {config}")
            predict_output_path = os.path.join(dataset_path, f"predictions_{config}")
            os.makedirs(predict_output_path, exist_ok=True)
            ok = run_command(
                ['nnUNetv2_predict', '-i', images_ts_path, '-o', predict_output_path, '-d', DATASET_ID, '-c', config, '-f', 'all'],
                f"PASO 3/4: Predicción en set de prueba ({config})"
            )
            if not ok:
                print(f"Fallo en la predicción de {config}. Abortando.")
                return

        if 'ensemble' in configurations:
            print(f"\nCreando predicciones de ensemble...")
            predict_output_path = os.path.join(dataset_path, "predictions_ensemble")
            os.makedirs(predict_output_path, exist_ok=True)

            available_predictions = []
            for config in ['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres']:
                pred_path = os.path.join(dataset_path, f"predictions_{config}")
                if os.path.exists(pred_path) and os.listdir(pred_path):
                    available_predictions.append(pred_path)

            legacy_pred = os.path.join(dataset_path, "predictions")
            if os.path.exists(legacy_pred) and os.listdir(legacy_pred):
                available_predictions.append(legacy_pred)

            seen = set()
            available_predictions = [p for p in available_predictions if not (p in seen or seen.add(p))]

            if len(available_predictions) < 2:
                print(f"Advertencia: Se encontraron solo {len(available_predictions)} predicciones. El ensemble real requiere al menos 2.")
                print(f"    Carpetas encontradas: {available_predictions}")
                if len(available_predictions) == 1:
                    print("    Creando 'predictions_ensemble' como copia de la única fuente disponible para poder evaluar...")
                    src = available_predictions[0]
                    for f in glob.glob(os.path.join(src, '*.nii.gz')):
                        dst = os.path.join(predict_output_path, os.path.basename(f))
                        if not os.path.exists(dst):
                            shutil.copy(f, dst)
                else:
                    print("    No hay predicciones disponibles para crear ensemble.")
            else:
                print(f"   Combinando predicciones de: {[os.path.basename(p) for p in available_predictions]}")
                ensemble_cmd = ['nnUNetv2_ensemble'] + ['-i'] + available_predictions + ['-o', predict_output_path]
                ok = run_command(
                    ensemble_cmd,
                    "PASO 3/4: Creación de ensemble"
                )
                if not ok:
                    print(f"Fallo en la creación del ensemble. Continuando...")

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
                        print(f"      No se encontró '{fname}' en fuentes: {[os.path.basename(s) for s in source_dirs]}")
                print(f"   Casos completados: {filled}")

    if start_stage <= 4 <= end_stage:
        for config in configurations:
            print(f"\nEvaluando configuración: {config}")
            predict_output_path = os.path.join(dataset_path, f"predictions_{config}")
            eval_output_path = os.path.join(dataset_path, f"evaluation_{config}")
            os.makedirs(eval_output_path, exist_ok=True)

            ok = run_evaluation_for_config(dataset_path, predict_output_path, eval_output_path, config)
            if not ok:
                print(f"Fallo en la evaluación de {config}.")

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETADO (etapas ejecutadas: {start_stage}-{end_stage})")
    print(f"Configuraciones procesadas: {', '.join(configurations)}")
    print("=" * 80)
    print(f"\nResultados de evaluación en:")
    for config in configurations:
        eval_path = os.path.join(dataset_path, f'evaluation_{config}')
        if os.path.exists(eval_path):
            print(f"   [{config}]: {eval_path}")
    print(f"\nPredicciones:")
    for config in configurations:
        pred_path = os.path.join(dataset_path, f'predictions_{config}')
        if os.path.exists(pred_path):
            print(f"   [{config}]: {pred_path}")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
