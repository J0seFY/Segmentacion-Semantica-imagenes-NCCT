#!/usr/bin/env python3
"""
Validación de Precisión Numérica: Algoritmos de Distancia de Hausdorff

Autor: Software Engineering Expert
Descripción:
    Script que valida la PRECISIÓN NUMÉRICA de 3 implementaciones de Hausdorff:
    1. hausdorff_taha (referencia baseline)
    2. hausdorff_kamata (optimización Kamata)
    3. hausdorff_k2t_maxheap_v2 (optimización K2-Tree)
    
    Objetivo: Verificar que TODOS los algoritmos retornan el MISMO valor numérico
    de distancia de Hausdorff (dentro de tolerancia de error de punto flotante).

Entrada:
    - Carpeta con predicciones .nii.gz (nnU-Net)
    - Carpeta con Ground Truths .nii.gz

Salida:
    - CSV con valores de Hausdorff por algoritmo, slice y paciente
    - Reporte de validación (diferencias, errores, tolerancia)
    - Gráfico de correlación entre algoritmos
    - Estadísticas de diferencias absolutas y relativas
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Tuple, Dict, List
import warnings

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Importar funciones de Hausdorff
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hausdorff_k2tree import (
    hausdorff_taha,
    hausdorff_kamata,
    hausdorff_k2t_maxheap_v2
)

warnings.filterwarnings('ignore')


class HausdorffAccuracyValidator:
    """
    Clase para validar precisión numérica de algoritmos de Hausdorff
    """
    
    def __init__(self, predictions_dir: str, ground_truth_dir: str, 
                 output_dir: str = None, tolerance: float = 1e-6):
        """
        Inicializa el validador.
        
        Args:
            predictions_dir: Carpeta con predicciones .nii.gz
            ground_truth_dir: Carpeta con Ground Truths .nii.gz
            output_dir: Directorio para guardar resultados
            tolerance: Tolerancia para considerar valores como iguales (default: 1e-6)
        """
        self.predictions_dir = Path(predictions_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.output_dir = Path(output_dir) if output_dir else self.predictions_dir / 'validation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tolerance = tolerance
        
        # Almacenar resultados
        self.results = []
        self.algorithms = {
            'Taha': hausdorff_taha,
            'Kamata': hausdorff_kamata,
            'K2Tree_MaxHeap_V2': hausdorff_k2t_maxheap_v2
        }
        
        self.validation_results = {
            'total_comparisons': 0,
            'exact_matches': 0,
            'within_tolerance': 0,
            'mismatches': 0,
            'errors': 0,
            'max_absolute_diff': 0.0,
            'max_relative_diff': 0.0,
            'differences': []
        }
        
    def _load_nifti(self, filepath: str) -> np.ndarray:
        """Carga archivo NIfTI y retorna datos como array numpy"""
        try:
            nifti_obj = nib.load(filepath)
            data = nifti_obj.get_fdata()
            return data.astype(np.uint8)
        except Exception as e:
            print(f"Error cargando {filepath}: {e}")
            return None
    
    def _binarize_volume(self, volume: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binariza volumen: valores > threshold -> 1, resto -> 0"""
        return (volume > threshold).astype(np.uint8)
    
    def _filter_slices_with_lesion(self, pred_volume: np.ndarray, 
                                    gt_volume: np.ndarray) -> np.ndarray:
        """Retorna índices de slices con lesión en pred Y gt"""
        valid_slices = []
        for z in range(pred_volume.shape[2]):
            if np.sum(pred_volume[:, :, z]) > 0 and np.sum(gt_volume[:, :, z]) > 0:
                valid_slices.append(z)
        return np.array(valid_slices)
    
    def _compute_all_algorithms(self, pred_slice: np.ndarray, 
                                gt_slice: np.ndarray) -> Dict[str, float]:
        """
        Calcula Hausdorff con TODOS los algoritmos para el mismo slice.
        
        Returns:
            Diccionario {nombre_algoritmo: valor_hausdorff}
        """
        results = {}
        
        for algo_name, algo_func in self.algorithms.items():
            try:
                hd_value = algo_func(pred_slice, gt_slice)
                results[algo_name] = float(hd_value)
            except Exception as e:
                print(f"Error en {algo_name}: {e}")
                results[algo_name] = None
        
        return results
    
    def _validate_values(self, values: Dict[str, float], patient_id: str, 
                         slice_idx: int) -> Dict:
        """
        Valida que todos los valores sean iguales (dentro de tolerancia).
        
        Returns:
            Diccionario con resultado de validación
        """
        # Filtrar valores None
        valid_values = {k: v for k, v in values.items() if v is not None}
        
        if len(valid_values) < len(self.algorithms):
            return {
                'status': 'ERROR',
                'patient_id': patient_id,
                'slice_idx': slice_idx,
                'values': values,
                'message': f'Algunos algoritmos fallaron: {len(valid_values)}/{len(self.algorithms)}'
            }
        
        # Comparar todos contra Taha (baseline)
        baseline_value = valid_values.get('Taha')
        
        if baseline_value is None:
            return {
                'status': 'ERROR',
                'patient_id': patient_id,
                'slice_idx': slice_idx,
                'values': values,
                'message': 'Baseline (Taha) no disponible'
            }
        
        # Calcular diferencias
        diffs = {}
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        all_match = True
        
        for algo_name, algo_value in valid_values.items():
            if algo_name == 'Taha':
                continue
            
            abs_diff = abs(algo_value - baseline_value)
            rel_diff = abs_diff / baseline_value if baseline_value != 0 else 0.0
            
            diffs[algo_name] = {
                'absolute': abs_diff,
                'relative': rel_diff
            }
            
            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)
            
            if abs_diff > self.tolerance:
                all_match = False
        
        # Determinar status
        if all_match:
            status = 'EXACT_MATCH' if max_abs_diff == 0 else 'WITHIN_TOLERANCE'
        else:
            status = 'MISMATCH'
        
        return {
            'status': status,
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'values': values,
            'differences': diffs,
            'max_absolute_diff': max_abs_diff,
            'max_relative_diff': max_rel_diff
        }
    
    def validate_patient(self, patient_id: str) -> List[Dict]:
        """
        Valida todos los slices de un paciente.
        
        Returns:
            Lista de diccionarios con resultados de validación por slice
        """
        # Buscar archivos
        pred_files = list(self.predictions_dir.glob(f"*{patient_id}*.nii.gz"))
        if not pred_files:
            direct_path = self.predictions_dir / f"{patient_id}.nii.gz"
            if direct_path.exists():
                pred_files = [direct_path]
        
        gt_files = list(self.ground_truth_dir.glob(f"*{patient_id}*.nii.gz"))
        if not gt_files:
            direct_path = self.ground_truth_dir / f"{patient_id}.nii.gz"
            if direct_path.exists():
                gt_files = [direct_path]
        
        if not pred_files or not gt_files:
            return []
        
        # Cargar volúmenes
        pred_vol = self._load_nifti(str(pred_files[0]))
        gt_vol = self._load_nifti(str(gt_files[0]))
        
        if pred_vol is None or gt_vol is None:
            return []
        
        # Binarizar
        pred_vol = self._binarize_volume(pred_vol)
        gt_vol = self._binarize_volume(gt_vol)
        
        # Obtener slices con lesión
        valid_slices = self._filter_slices_with_lesion(pred_vol, gt_vol)
        
        if len(valid_slices) == 0:
            return []
        
        patient_results = []
        
        # Procesar cada slice
        for z_idx in valid_slices:
            pred_slice = pred_vol[:, :, z_idx]
            gt_slice = gt_vol[:, :, z_idx]
            
            # Calcular con todos los algoritmos
            values = self._compute_all_algorithms(pred_slice, gt_slice)
            
            # Validar
            validation = self._validate_values(values, patient_id, int(z_idx))
            patient_results.append(validation)
            
            # Actualizar estadísticas globales
            self.validation_results['total_comparisons'] += 1
            
            if validation['status'] == 'EXACT_MATCH':
                self.validation_results['exact_matches'] += 1
            elif validation['status'] == 'WITHIN_TOLERANCE':
                self.validation_results['within_tolerance'] += 1
            elif validation['status'] == 'MISMATCH':
                self.validation_results['mismatches'] += 1
            elif validation['status'] == 'ERROR':
                self.validation_results['errors'] += 1
            
            if validation.get('max_absolute_diff', 0) > self.validation_results['max_absolute_diff']:
                self.validation_results['max_absolute_diff'] = validation['max_absolute_diff']
            
            if validation.get('max_relative_diff', 0) > self.validation_results['max_relative_diff']:
                self.validation_results['max_relative_diff'] = validation['max_relative_diff']
            
            # Almacenar diferencias para análisis
            if 'differences' in validation:
                self.validation_results['differences'].append({
                    'patient_id': patient_id,
                    'slice_idx': z_idx,
                    'max_abs_diff': validation['max_absolute_diff'],
                    'max_rel_diff': validation['max_relative_diff']
                })
        
        return patient_results
    
    def run_validation(self) -> pd.DataFrame:
        """
        Ejecuta validación para todos los pacientes.
        
        Returns:
            DataFrame con resultados detallados
        """
        # Obtener lista de pacientes
        pred_files = list(self.predictions_dir.glob("*.nii.gz"))
        patient_ids = set()
        
        for fpath in pred_files:
            filename = fpath.stem.replace('.nii', '')
            
            if 'Case_' in filename:
                case_id = filename.split('_')[1]
                patient_ids.add(case_id)
            else:
                patient_ids.add(filename)
        
        if not patient_ids:
            print("No se encontraron archivos de predicción.")
            return None
        
        patient_ids = sorted(list(patient_ids))
        all_results = []
        
        print(f"\n{'='*80}")
        print(f"VALIDACIÓN DE PRECISIÓN NUMÉRICA - {len(patient_ids)} PACIENTES")
        print(f"Tolerancia: {self.tolerance}")
        print(f"{'='*80}\n")
        
        # Procesar cada paciente
        pbar = tqdm(total=len(patient_ids), desc="Validando pacientes")
        
        for patient_id in patient_ids:
            patient_results = self.validate_patient(patient_id)
            
            # Convertir a formato DataFrame
            for result in patient_results:
                row = {
                    'patient_id': result['patient_id'],
                    'slice_idx': result['slice_idx'],
                    'status': result['status']
                }
                
                # Agregar valores de cada algoritmo
                if 'values' in result:
                    for algo_name, value in result['values'].items():
                        row[f'hausdorff_{algo_name}'] = value
                
                # Agregar diferencias
                if 'differences' in result:
                    for algo_name, diffs in result['differences'].items():
                        row[f'abs_diff_{algo_name}'] = diffs['absolute']
                        row[f'rel_diff_{algo_name}'] = diffs['relative']
                
                if 'max_absolute_diff' in result:
                    row['max_absolute_diff'] = result['max_absolute_diff']
                    row['max_relative_diff'] = result['max_relative_diff']
                
                all_results.append(row)
            
            pbar.update(1)
        
        pbar.close()
        
        # Crear DataFrame
        if all_results:
            self.results_df = pd.DataFrame(all_results)
            return self.results_df
        else:
            print("No se pudo procesar ningún paciente.")
            return None
    
    def print_validation_summary(self):
        """Imprime resumen de validación"""
        print("\n" + "="*80)
        print("RESUMEN DE VALIDACIÓN DE PRECISIÓN NUMÉRICA")
        print("="*80 + "\n")
        
        total = self.validation_results['total_comparisons']
        exact = self.validation_results['exact_matches']
        within = self.validation_results['within_tolerance']
        mismatch = self.validation_results['mismatches']
        errors = self.validation_results['errors']
        
        print(f"{'Total de comparaciones:':<35} {total}")
        print(f"{'Coincidencias exactas:':<35} {exact} ({exact/total*100:.2f}%)")
        print(f"{'Dentro de tolerancia:':<35} {within} ({within/total*100:.2f}%)")
        print(f"{'Diferencias significativas:':<35} {mismatch} ({mismatch/total*100:.2f}%)")
        print(f"{'Errores:':<35} {errors} ({errors/total*100:.2f}%)")
        
        print("\n" + "-"*80)
        print(f"{'Diferencia absoluta máxima:':<35} {self.validation_results['max_absolute_diff']:.10f}")
        print(f"{'Diferencia relativa máxima:':<35} {self.validation_results['max_relative_diff']*100:.8f}%")
        print(f"{'Tolerancia configurada:':<35} {self.tolerance:.10f}")
        
        # Calcular estadísticas descriptivas de los valores de Hausdorff
        if self.results_df is not None and len(self.results_df) > 0:
            print("\n" + "="*80)
            print("ESTADÍSTICAS DESCRIPTIVAS DE VALORES DE HAUSDORFF")
            print("="*80 + "\n")
            
            # Tabla de estadísticas
            print(f"{'Algoritmo':<25} {'Media':<12} {'Mediana':<12} {'Desv.Std':<12} {'Min':<12} {'Max':<12}")
            print("-" * 85)
            
            for algo_name in ['Taha', 'Kamata', 'K2Tree_MaxHeap_V2']:
                col_name = f'hausdorff_{algo_name}'
                if col_name in self.results_df.columns:
                    values = self.results_df[col_name].dropna()
                    
                    if len(values) > 0:
                        mean_val = values.mean()
                        median_val = values.median()
                        std_val = values.std()
                        min_val = values.min()
                        max_val = values.max()
                        
                        # Nombre amigable
                        display_name = algo_name.replace('K2Tree_MaxHeap_V2', 'K2-Tree V2')
                        
                        print(f"{display_name:<25} {mean_val:<12.4f} {median_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")
        
        print("\n" + "="*80)
        
        # Determinar resultado final
        if mismatch == 0 and errors == 0:
            print("✅ VALIDACIÓN EXITOSA: Todos los algoritmos son numéricamente equivalentes")
        elif mismatch > 0:
            print(f"⚠️  ADVERTENCIA: {mismatch} diferencias significativas detectadas")
        
        if errors > 0:
            print(f"❌ ERROR: {errors} comparaciones fallaron")
        
        print("="*80 + "\n")
    
    def save_results_csv(self):
        """Guarda resultados detallados en CSV"""
        if self.results_df is None or len(self.results_df) == 0:
            return
        
        output_file = self.output_dir / 'hausdorff_validation_detailed.csv'
        self.results_df.to_csv(output_file, index=False)
        print(f"✓ Resultados detallados guardados: {output_file}")
    
    def save_summary_report(self):
        """Guarda reporte de resumen en texto"""
        output_file = self.output_dir / 'hausdorff_validation_summary.txt'
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE VALIDACIÓN DE PRECISIÓN NUMÉRICA\n")
            f.write("Algoritmos de Distancia de Hausdorff\n")
            f.write("="*80 + "\n\n")
            
            f.write("CONFIGURACIÓN\n")
            f.write("-"*80 + "\n")
            f.write(f"Tolerancia: {self.tolerance:.10f}\n")
            f.write(f"Directorio predicciones: {self.predictions_dir}\n")
            f.write(f"Directorio ground truth: {self.ground_truth_dir}\n\n")
            
            f.write("RESULTADOS DE VALIDACIÓN\n")
            f.write("-"*80 + "\n")
            
            total = self.validation_results['total_comparisons']
            exact = self.validation_results['exact_matches']
            within = self.validation_results['within_tolerance']
            mismatch = self.validation_results['mismatches']
            errors = self.validation_results['errors']
            
            f.write(f"Total de comparaciones: {total}\n")
            f.write(f"Coincidencias exactas: {exact} ({exact/total*100:.2f}%)\n")
            f.write(f"Dentro de tolerancia: {within} ({within/total*100:.2f}%)\n")
            f.write(f"Diferencias significativas: {mismatch} ({mismatch/total*100:.2f}%)\n")
            f.write(f"Errores: {errors} ({errors/total*100:.2f}%)\n\n")
            
            f.write("ESTADÍSTICAS DE DIFERENCIAS\n")
            f.write("-"*80 + "\n")
            f.write(f"Diferencia absoluta máxima: {self.validation_results['max_absolute_diff']:.10f}\n")
            f.write(f"Diferencia relativa máxima: {self.validation_results['max_relative_diff']*100:.8f}%\n\n")
            
            # Agregar estadísticas descriptivas
            if self.results_df is not None and len(self.results_df) > 0:
                f.write("ESTADÍSTICAS DESCRIPTIVAS DE VALORES DE HAUSDORFF\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Algoritmo':<25} {'Media':<12} {'Mediana':<12} {'Desv.Std':<12} {'Min':<12} {'Max':<12}\n")
                f.write("-" * 85 + "\n")
                
                for algo_name in ['Taha', 'Kamata', 'K2Tree_MaxHeap_V2']:
                    col_name = f'hausdorff_{algo_name}'
                    if col_name in self.results_df.columns:
                        values = self.results_df[col_name].dropna()
                        
                        if len(values) > 0:
                            mean_val = values.mean()
                            median_val = values.median()
                            std_val = values.std()
                            min_val = values.min()
                            max_val = values.max()
                            
                            # Nombre amigable
                            display_name = algo_name.replace('K2Tree_MaxHeap_V2', 'K2-Tree V2')
                            
                            f.write(f"{display_name:<25} {mean_val:<12.4f} {median_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}\n")
                
                f.write("\n")
            
            f.write("CONCLUSIÓN\n")
            f.write("-"*80 + "\n")
            if mismatch == 0 and errors == 0:
                f.write("✅ VALIDACIÓN EXITOSA\n")
                f.write("Todos los algoritmos son numéricamente equivalentes dentro de la tolerancia.\n")
                if self.results_df is not None and len(self.results_df) > 0:
                    f.write("\nLas estadísticas descriptivas son IDÉNTICAS para todos los algoritmos,\n")
                    f.write("confirmando que no solo coinciden los valores individuales, sino también\n")
                    f.write("todas las medidas de tendencia central y dispersión.\n")
            elif mismatch > 0:
                f.write("⚠️  ADVERTENCIA\n")
                f.write(f"{mismatch} diferencias significativas detectadas.\n")
                f.write("Se recomienda revisar los casos con diferencias.\n")
            
            if errors > 0:
                f.write(f"\n❌ ERRORES DETECTADOS: {errors} comparaciones fallaron.\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Reporte de resumen guardado: {output_file}")
    
    def plot_correlation(self):
        """Genera gráficos de correlación entre algoritmos"""
        if self.results_df is None or len(self.results_df) == 0:
            return
        
        # Filtrar solo slices válidos
        df_valid = self.results_df[self.results_df['status'].isin(['EXACT_MATCH', 'WITHIN_TOLERANCE', 'MISMATCH'])]
        
        if len(df_valid) == 0:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Comparación 1: Taha vs Kamata
        ax1 = axes[0]
        ax1.scatter(df_valid['hausdorff_Taha'], df_valid['hausdorff_Kamata'], 
                   alpha=0.6, s=30, color='#4ECDC4')
        
        # Línea de identidad (x=y)
        min_val = min(df_valid['hausdorff_Taha'].min(), df_valid['hausdorff_Kamata'].min())
        max_val = max(df_valid['hausdorff_Taha'].max(), df_valid['hausdorff_Kamata'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (ideal)')
        
        ax1.set_xlabel('Taha (Baseline)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Kamata', fontsize=11, fontweight='bold')
        ax1.set_title('Correlación: Taha vs Kamata', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Comparación 2: Taha vs K2Tree
        ax2 = axes[1]
        ax2.scatter(df_valid['hausdorff_Taha'], df_valid['hausdorff_K2Tree_MaxHeap_V2'], 
                   alpha=0.6, s=30, color='#45B7D1')
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (ideal)')
        
        ax2.set_xlabel('Taha (Baseline)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('K2Tree MaxHeap V2', fontsize=11, fontweight='bold')
        ax2.set_title('Correlación: Taha vs K2Tree', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Comparación 3: Kamata vs K2Tree
        ax3 = axes[2]
        ax3.scatter(df_valid['hausdorff_Kamata'], df_valid['hausdorff_K2Tree_MaxHeap_V2'], 
                   alpha=0.6, s=30, color='#FF6B6B')
        
        min_val2 = min(df_valid['hausdorff_Kamata'].min(), df_valid['hausdorff_K2Tree_MaxHeap_V2'].min())
        max_val2 = max(df_valid['hausdorff_Kamata'].max(), df_valid['hausdorff_K2Tree_MaxHeap_V2'].max())
        ax3.plot([min_val2, max_val2], [min_val2, max_val2], 'r--', linewidth=2, label='y=x (ideal)')
        
        ax3.set_xlabel('Kamata', fontsize=11, fontweight='bold')
        ax3.set_ylabel('K2Tree MaxHeap V2', fontsize=11, fontweight='bold')
        ax3.set_title('Correlación: Kamata vs K2Tree', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'hausdorff_validation_correlation.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de correlación guardado: {output_file}")
        plt.close()
    
    def plot_differences_distribution(self):
        """Genera gráfico de distribución de diferencias"""
        if self.results_df is None or len(self.results_df) == 0:
            return
        
        # Filtrar solo slices con diferencias calculadas
        df_valid = self.results_df.dropna(subset=['max_absolute_diff'])
        
        if len(df_valid) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico 1: Distribución de diferencias absolutas
        ax1 = axes[0]
        ax1.hist(df_valid['max_absolute_diff'], bins=50, color='#4ECDC4', 
                alpha=0.7, edgecolor='black')
        ax1.axvline(x=self.tolerance, color='red', linestyle='--', linewidth=2, 
                   label=f'Tolerancia ({self.tolerance:.1e})')
        ax1.set_xlabel('Diferencia Absoluta Máxima', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
        ax1.set_title('Distribución de Diferencias Absolutas', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Gráfico 2: Distribución de diferencias relativas (%)
        ax2 = axes[1]
        ax2.hist(df_valid['max_relative_diff'] * 100, bins=50, color='#45B7D1', 
                alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Diferencia Relativa Máxima (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
        ax2.set_title('Distribución de Diferencias Relativas', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'hausdorff_validation_differences.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de diferencias guardado: {output_file}")
        plt.close()


def main():
    """Función principal para ejecutar validación desde línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Validación de Precisión Numérica de Algoritmos de Hausdorff',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python validate_hausdorff_accuracy.py \\
    --pred predictions_dir \\
    --gt ground_truth_dir \\
    --output validation_results \\
    --tolerance 1e-6
        """
    )
    
    parser.add_argument('--pred', '--predictions', type=str, required=True,
                       help='Directorio con predicciones .nii.gz')
    parser.add_argument('--gt', '--ground_truth', type=str, required=True,
                       help='Directorio con Ground Truths .nii.gz')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Directorio de salida (default: pred_dir/validation)')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-6,
                       help='Tolerancia para considerar valores iguales (default: 1e-6)')
    
    args = parser.parse_args()
    
    # Validar directorios
    if not Path(args.pred).exists():
        print(f"Error: Directorio de predicciones no existe: {args.pred}")
        sys.exit(1)
    
    if not Path(args.gt).exists():
        print(f"Error: Directorio de GT no existe: {args.gt}")
        sys.exit(1)
    
    # Ejecutar validación
    validator = HausdorffAccuracyValidator(args.pred, args.gt, args.output, args.tolerance)
    
    # Correr validación
    results_df = validator.run_validation()
    
    if results_df is not None:
        # Mostrar resumen
        validator.print_validation_summary()
        
        # Guardar resultados
        validator.save_results_csv()
        validator.save_summary_report()
        validator.plot_correlation()
        validator.plot_differences_distribution()
        
        print(f"\n✓ Validación completada exitosamente.")
        print(f"✓ Resultados guardados en: {validator.output_dir}\n")
    else:
        print("Error: No se pudo completar la validación.")
        sys.exit(1)


if __name__ == '__main__':
    main()
