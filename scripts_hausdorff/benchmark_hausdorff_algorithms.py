#!/usr/bin/env python3
"""
Benchmark de Eficiencia Computacional: Comparación de 3 Algoritmos de Distancia de Hausdorff

Autor: Software Engineering Expert
Descripción:
    Script que realiza benchmark de tiempos de ejecución para 3 implementaciones
    de distancia de Hausdorff:
    1. hausdorff_taha (referencia baseline)
    2. hausdorff_kamata (optimización Kamata)
    3. hausdorff_k2t_maxheap (optimización K2-Tree con MaxHeap)
    
    Procesa predicciones 3D slice-by-slice, filtrando solo cortes con lesión,
    y mide tiempos con alta precisión usando time.perf_counter().

Entrada:
    - Carpeta con predicciones .nii.gz (nnU-Net)
    - Carpeta correspondiente con Ground Truths .nii.gz

Salida:
    - DataFrame CSV con resultados crudos
    - Estadísticas resumen (tiempo promedio, speedup)
    - Gráfico comparativo de eficiencia
"""

import argparse
import sys
import os
import time
import json
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


class HausdorffBenchmark:
    """
    Clase para realizar benchmark de algoritmos de Distancia de Hausdorff
    """
    
    def __init__(self, predictions_dir: str, ground_truth_dir: str, 
                 output_dir: str = None):
        """
        Inicializa el benchmark.
        
        Args:
            predictions_dir: Carpeta con predicciones .nii.gz
            ground_truth_dir: Carpeta con Ground Truths .nii.gz
            output_dir: Directorio para guardar resultados (default: predictions_dir)
        """
        self.predictions_dir = Path(predictions_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.output_dir = Path(output_dir) if output_dir else self.predictions_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Almacenar resultados
        self.results = []
        self.algorithms = {
            'Taha (Baseline)': hausdorff_taha,
            'Kamata': hausdorff_kamata,
            'K2-Tree MaxHeap V2': hausdorff_k2t_maxheap_v2
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
        """
        Binariza volumen: valores > threshold -> 1, resto -> 0
        """
        return (volume > threshold).astype(np.uint8)
    
    def _filter_slices_with_lesion(self, pred_volume: np.ndarray, 
                                    gt_volume: np.ndarray) -> np.ndarray:
        """
        Retorna índices de slices donde existe lesión tanto en pred como en GT
        (evita procesar slices vacíos)
        """
        valid_slices = []
        for z in range(pred_volume.shape[2]):
            if np.sum(pred_volume[:, :, z]) > 0 and np.sum(gt_volume[:, :, z]) > 0:
                valid_slices.append(z)
        return np.array(valid_slices)
    
    def _run_algorithm(self, algorithm_func, pred_slice: np.ndarray, 
                       gt_slice: np.ndarray) -> Tuple[float, float]:
        """
        Ejecuta un algoritmo y mide tiempo de ejecución.
        
        Returns:
            (tiempo_ms, valor_hausdorff)
        """
        try:
            start = time.perf_counter()
            result = algorithm_func(pred_slice, gt_slice)
            elapsed = time.perf_counter() - start
            return elapsed * 1000, result  # Convertir a ms
        except Exception as e:
            print(f"Error ejecutando algoritmo: {e}")
            return None, None
    
    def benchmark_patient(self, patient_id: str) -> Dict:
        """
        Realiza benchmark para un paciente completo (3D).
        
        Returns:
            Diccionario con resultados de tiempos y valores de Hausdorff
        """
        # Buscar archivos con múltiples patrones
        pred_files = list(self.predictions_dir.glob(f"*{patient_id}*.nii.gz"))
        
        # Si no encuentra con glob, intentar busca directa
        if not pred_files:
            direct_path = self.predictions_dir / f"{patient_id}.nii.gz"
            if direct_path.exists():
                pred_files = [direct_path]
        
        # Buscar GT también con múltiples patrones
        gt_files = list(self.ground_truth_dir.glob(f"*{patient_id}*.nii.gz"))
        if not gt_files:
            direct_path = self.ground_truth_dir / f"{patient_id}.nii.gz"
            if direct_path.exists():
                gt_files = [direct_path]
        
        if not pred_files or not gt_files:
            return None
        
        # Cargar volúmenes
        pred_vol = self._load_nifti(str(pred_files[0]))
        gt_vol = self._load_nifti(str(gt_files[0]))
        
        if pred_vol is None or gt_vol is None:
            return None
        
        # Binarizar
        pred_vol = self._binarize_volume(pred_vol)
        gt_vol = self._binarize_volume(gt_vol)
        
        # Obtener slices con lesión
        valid_slices = self._filter_slices_with_lesion(pred_vol, gt_vol)
        
        if len(valid_slices) == 0:
            return None
        
        patient_results = {
            'patient_id': patient_id,
            'total_slices': pred_vol.shape[2],
            'slices_with_lesion': len(valid_slices),
            'algorithms': {}
        }
        
        # Procesar cada algoritmo
        for algo_name, algo_func in self.algorithms.items():
            times_ms = []
            hausdorff_values = []
            
            # Iterar sobre slices con lesión
            for z_idx in valid_slices:
                pred_slice = pred_vol[:, :, z_idx]
                gt_slice = gt_vol[:, :, z_idx]
                
                time_ms, hd_value = self._run_algorithm(algo_func, pred_slice, gt_slice)
                
                if time_ms is not None:
                    times_ms.append(time_ms)
                    hausdorff_values.append(hd_value)
            
            if times_ms:
                patient_results['algorithms'][algo_name] = {
                    'times_ms': times_ms,
                    'hausdorff_values': hausdorff_values,
                    'mean_time_ms': np.mean(times_ms),
                    'std_time_ms': np.std(times_ms),
                    'min_time_ms': np.min(times_ms),
                    'max_time_ms': np.max(times_ms),
                    'num_slices_processed': len(times_ms)
                }
        
        return patient_results
    
    def run_benchmark(self) -> pd.DataFrame:
        """
        Ejecuta benchmark para todos los pacientes en el directorio.
        
        Returns:
            DataFrame con resultados
        """
        # Obtener lista de pacientes únicos
        pred_files = list(self.predictions_dir.glob("*.nii.gz"))
        patient_ids = set()
        
        for fpath in pred_files:
            # Extraer ID del paciente
            # Soporta formatos: Case_XXXXX_0000.nii.gz o XXXXX.nii.gz
            filename = fpath.stem.replace('.nii', '')
            
            if 'Case_' in filename:
                # Formato: Case_XXXXX_0000.nii.gz
                case_id = filename.split('_')[1]
                patient_ids.add(case_id)
            else:
                # Formato: XXXXX.nii.gz (nnU-Net predictions)
                patient_ids.add(filename)
        
        if not patient_ids:
            print("No se encontraron archivos de predicción en el directorio.")
            print(f"Directorio: {self.predictions_dir}")
            print(f"Archivos encontrados: {len(pred_files)}")
            if pred_files:
                print(f"Ejemplos: {[f.name for f in pred_files[:3]]}")
            return None
        
        patient_ids = sorted(list(patient_ids))
        all_results = []
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK DE HAUSDORFF - {len(patient_ids)} PACIENTES")
        print(f"{'='*80}\n")
        
        # Procesar cada paciente
        pbar = tqdm(total=len(patient_ids), desc="Procesando pacientes")
        
        for patient_id in patient_ids:
            result = self.benchmark_patient(patient_id)
            
            if result:
                # Extraer resultados por algoritmo
                for algo_name, algo_data in result['algorithms'].items():
                    row = {
                        'patient_id': patient_id,
                        'algorithm': algo_name,
                        'total_slices': result['total_slices'],
                        'slices_with_lesion': result['slices_with_lesion'],
                        'mean_time_ms': algo_data['mean_time_ms'],
                        'std_time_ms': algo_data['std_time_ms'],
                        'min_time_ms': algo_data['min_time_ms'],
                        'max_time_ms': algo_data['max_time_ms'],
                        'num_slices_processed': algo_data['num_slices_processed'],
                    }
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
    
    def compute_statistics(self) -> Dict:
        """
        Calcula estadísticas resumen: tiempo promedio, desviación estándar, speedup
        
        Returns:
            Diccionario con estadísticas
        """
        if self.results_df is None or len(self.results_df) == 0:
            return None
        
        stats = {}
        baseline_time = self.results_df[self.results_df['algorithm'] == 'Taha (Baseline)']['mean_time_ms'].mean()
        
        for algo_name in self.algorithms.keys():
            algo_data = self.results_df[self.results_df['algorithm'] == algo_name]
            
            if len(algo_data) > 0:
                mean_time = algo_data['mean_time_ms'].mean()
                std_time = algo_data['mean_time_ms'].std()
                speedup = baseline_time / mean_time if mean_time > 0 else 1.0
                
                stats[algo_name] = {
                    'mean_time_ms': mean_time,
                    'std_time_ms': std_time,
                    'speedup': speedup,
                    'samples': len(algo_data)
                }
        
        return stats
    
    def print_summary(self):
        """Imprime resumen de estadísticas en consola"""
        stats = self.compute_statistics()
        
        if stats is None:
            print("No hay datos para mostrar.")
            return
        
        print("\n" + "="*80)
        print("RESUMEN DE ESTADÍSTICAS POR ALGORITMO")
        print("="*80 + "\n")
        
        # Tabla con formato
        print(f"{'Algoritmo':<25} {'Tiempo (ms)':<20} {'Speedup':<15} {'Muestras':<10}")
        print("-" * 70)
        
        for algo_name, stats_data in stats.items():
            mean_time = stats_data['mean_time_ms']
            std_time = stats_data['std_time_ms']
            speedup = stats_data['speedup']
            samples = stats_data['samples']
            
            time_str = f"{mean_time:.4f} ± {std_time:.4f}"
            speedup_str = f"{speedup:.2f}x"
            
            print(f"{algo_name:<25} {time_str:<20} {speedup_str:<15} {samples:<10}")
        
        print("\n" + "="*80)
        print(f"{'Aceleración máxima:':<25} {max(s['speedup'] for s in stats.values()):.2f}x")
        print(f"{'Algoritmo más rápido:':<25} {max(stats, key=lambda k: stats[k]['speedup'])}")
        print("="*80 + "\n")
    
    def save_results_csv(self):
        """Guarda resultados crudos en CSV"""
        if self.results_df is None or len(self.results_df) == 0:
            return
        
        output_file = self.output_dir / 'hausdorff_benchmark_raw.csv'
        self.results_df.to_csv(output_file, index=False)
        print(f"✓ Resultados crudos guardados: {output_file}")
    
    def save_statistics_json(self):
        """Guarda estadísticas en formato JSON"""
        stats = self.compute_statistics()
        
        if stats is None:
            return
        
        # Convertir arrays a listas para JSON serialization
        stats_json = {}
        for algo_name, data in stats.items():
            stats_json[algo_name] = {
                'mean_time_ms': float(data['mean_time_ms']),
                'std_time_ms': float(data['std_time_ms']),
                'speedup': float(data['speedup']),
                'samples': int(data['samples'])
            }
        
        output_file = self.output_dir / 'hausdorff_benchmark_stats.json'
        with open(output_file, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        print(f"✓ Estadísticas guardadas: {output_file}")
    
    def plot_comparison(self):
        """Genera gráfico comparativo de tiempos y speedup"""
        stats = self.compute_statistics()
        
        if stats is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico 1: Tiempo promedio por algoritmo
        algo_names = list(stats.keys())
        times = [stats[name]['mean_time_ms'] for name in algo_names]
        stds = [stats[name]['std_time_ms'] for name in algo_names]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax1 = axes[0]
        bars1 = ax1.bar(algo_names, times, yerr=stds, capsize=5, 
                        color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Tiempo Promedio (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Tiempo de Ejecución por Algoritmo\n(barras de error = ±1 std)', 
                      fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(bottom=0)
        
        # Anotaciones de valores
        for bar, time_val, std_val in zip(bars1, times, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.4f}\n±{std_val:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Gráfico 2: Speedup vs Baseline
        speedups = [stats[name]['speedup'] for name in algo_names]
        
        ax2 = axes[1]
        bars2 = ax2.bar(algo_names, speedups, color=colors, alpha=0.8, 
                        edgecolor='black', linewidth=1.5)
        ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
        ax2.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
        ax2.set_title('Aceleración vs Taha Baseline', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(bottom=0, top=max(speedups) * 1.15)
        ax2.legend(fontsize=10)
        
        # Anotaciones de valores
        for bar, speedup_val in zip(bars2, speedups):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup_val:.2f}x',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'hausdorff_benchmark_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: {output_file}")
        plt.close()
    
    def plot_detailed_times(self):
        """Genera gráfico con distribución detallada de tiempos"""
        if self.results_df is None or len(self.results_df) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Box plot por algoritmo
        algo_names = sorted(self.results_df['algorithm'].unique())
        data_for_plot = [self.results_df[self.results_df['algorithm'] == algo]['mean_time_ms'].values 
                         for algo in algo_names]
        
        bp = ax.boxplot(data_for_plot, labels=algo_names, patch_artist=True,
                        notch=False, widths=0.6)
        
        # Colorear boxes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        # Personalizar plot
        ax.set_ylabel('Tiempo Promedio por Slice (ms)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Algoritmo', fontsize=12, fontweight='bold')
        ax.set_title('Distribución de Tiempos de Ejecución\n(Box-plot por algoritmo)', 
                     fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'hausdorff_benchmark_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de distribución guardado: {output_file}")
        plt.close()


def main():
    """
    Función principal para ejecutar benchmark desde línea de comandos
    """
    parser = argparse.ArgumentParser(
        description='Benchmark de Algoritmos de Distancia de Hausdorff (3D → 2D)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python benchmark_hausdorff_algorithms.py \\
    --pred predictions_dir \\
    --gt ground_truth_dir \\
    --output results_dir
        """
    )
    
    parser.add_argument('--pred', '--predictions', type=str, required=True,
                       help='Directorio con predicciones .nii.gz')
    parser.add_argument('--gt', '--ground_truth', type=str, required=True,
                       help='Directorio con Ground Truths .nii.gz')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Directorio de salida (default: predicciones_dir)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Modo verbose')
    
    args = parser.parse_args()
    
    # Validar directorios
    if not Path(args.pred).exists():
        print(f"Error: Directorio de predicciones no existe: {args.pred}")
        sys.exit(1)
    
    if not Path(args.gt).exists():
        print(f"Error: Directorio de GT no existe: {args.gt}")
        sys.exit(1)
    
    # Ejecutar benchmark
    benchmark = HausdorffBenchmark(args.pred, args.gt, args.output)
    
    # Correr benchmark
    results_df = benchmark.run_benchmark()
    
    if results_df is not None:
        # Mostrar resumen
        benchmark.print_summary()
        
        # Guardar resultados
        benchmark.save_results_csv()
        benchmark.save_statistics_json()
        benchmark.plot_comparison()
        benchmark.plot_detailed_times()
        
        print(f"\n✓ Benchmark completado exitosamente.")
        print(f"✓ Resultados guardados en: {benchmark.output_dir}\n")
    else:
        print("Error: No se pudo completar el benchmark.")
        sys.exit(1)


if __name__ == '__main__':
    main()
