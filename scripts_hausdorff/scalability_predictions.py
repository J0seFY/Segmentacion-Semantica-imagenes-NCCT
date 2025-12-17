#!/usr/bin/env python3
"""
Análisis Avanzado de Escalabilidad con Estimación de Puntos Reales

Script que proporciona análisis más detallado:
- Estimación de puntos basada en configuración real de CT
- Tablas de predicción de tiempo para diferentes tamaños
- Análisis de punto de equilibrio entre algoritmos
- Visualización de costos computacionales

Uso:
    python scalability_predictions.py --csv hausdorff_benchmark_raw.csv --output analysis/
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def estimate_points_realistic(slices: int, lesion_size: str = 'medium') -> int:
    """
    Estima número de puntos basado en parámetros reales de CT cerebral.
    
    Args:
        slices: Número de slices con lesión
        lesion_size: 'small' (3-5mm), 'medium' (10-20mm), 'large' (30mm+)
    
    Returns:
        Número estimado de puntos (píxeles)
    """
    # Parámetros típicos de CT cerebral:
    # - Resolución: 512x512 píxeles
    # - Espesor: 0.5-2mm por slice
    # - Lesión: círculo o irregular
    
    # Áreas típicas de lesión por tamaño (en píxeles):
    area_params = {
        'small': (150, 400),      # 3-5mm diámetro → 150-400 píxeles
        'medium': (400, 1500),    # 10-20mm diámetro → 400-1500 píxeles
        'large': (1500, 5000)     # 30mm+ → 1500-5000 píxeles
    }
    
    min_area, max_area = area_params.get(lesion_size, (400, 1500))
    
    # Promedio de puntos por slice
    avg_area = (min_area + max_area) / 2
    
    # Total de puntos: promedio_area * slices_con_lesion
    total_points = avg_area * slices
    
    return int(total_points)


def create_prediction_tables(results: dict):
    """Crea tablas de predicción de tiempo para diferentes tamaños"""
    print("\n" + "="*100)
    print("TABLAS DE PREDICCIÓN DE TIEMPOS")
    print("="*100 + "\n")
    
    # Diferentes escenarios
    scenarios = [
        ('Pequeña lesión (3-5mm)', 5, 'small'),   # 5 slices, lesión pequeña
        ('Lesión pequeña (3-5mm)', 15, 'small'),  # 15 slices
        ('Mediana lesión (10-20mm)', 10, 'medium'),  # 10 slices, lesión mediana
        ('Mediana lesión (10-20mm)', 20, 'medium'),  # 20 slices
        ('Grande lesión (30mm+)', 10, 'large'),   # 10 slices, lesión grande
        ('Grande lesión (30mm+)', 25, 'large'),   # 25 slices
    ]
    
    for scenario_name, num_slices, lesion_type in scenarios:
        print(f"\n{scenario_name}:")
        print(f"  Slices: {num_slices}, Tipo: {lesion_type}")
        print("-" * 100)
        
        # Estimar puntos
        points = estimate_points_realistic(num_slices, lesion_type)
        print(f"  Puntos estimados: {points:,}")
        print()
        
        # Mostrar tabla de tiempos predichos
        print(f"  {'Algoritmo':<30} {'Tiempo Predicho (ms)':<25} {'Nota':<45}")
        print("  " + "-" * 100)
        
        for algo_name, res in results.items():
            if 'a' in res and 'b' in res:
                predicted_time = res['a'] * (points ** res['b'])
                
                # Nota descriptiva
                if predicted_time < 1:
                    note = "Muy rápido"
                elif predicted_time < 10:
                    note = "Rápido"
                elif predicted_time < 100:
                    note = "Aceptable"
                elif predicted_time < 1000:
                    note = "Moderado"
                else:
                    note = "Lento"
                
                print(f"  {algo_name:<30} {predicted_time:>20.2f} ms     {note:<45}")


def create_breakeven_analysis(results: dict, output_dir: Path):
    """Analiza puntos de equilibrio entre algoritmos"""
    print("\n" + "="*100)
    print("ANÁLISIS DE PUNTOS DE EQUILIBRIO")
    print("="*100 + "\n")
    
    # Crear rango de puntos
    points_range = np.logspace(1, 4, 100)  # 10 a 10000 puntos
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Tiempo vs Puntos (todos)
    ax = axes[0, 0]
    
    colors = {'Taha (Baseline)': '#FF6B6B', 'Kamata': '#4ECDC4', 'K2-Tree MaxHeap V2': '#45B7D1'}
    
    for algo_name, res in results.items():
        if 'a' in res and 'b' in res:
            times = res['a'] * (points_range ** res['b'])
            ax.plot(points_range, times, linewidth=2.5, label=algo_name, 
                   color=colors.get(algo_name, '#95A5A6'))
    
    ax.set_xlabel('Número de Puntos', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiempo Predicho (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Predicción de Tiempos por Complejidad', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Gráfico 2: Speedup Kamata vs Taha
    ax = axes[0, 1]
    
    if 'Taha (Baseline)' in results and 'Kamata' in results:
        res_taha = results['Taha (Baseline)']
        res_kamata = results['Kamata']
        
        taha_times = res_taha['a'] * (points_range ** res_taha['b'])
        kamata_times = res_kamata['a'] * (points_range ** res_kamata['b'])
        speedup_kamata = taha_times / kamata_times
        
        ax.plot(points_range, speedup_kamata, linewidth=2.5, color='#4ECDC4', label='Speedup')
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
        
        ax.fill_between(points_range, 1, speedup_kamata, alpha=0.2, color='#4ECDC4')
        
        ax.set_xlabel('Número de Puntos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (Taha / Kamata)', fontsize=12, fontweight='bold')
        ax.set_title('Speedup Kamata vs Taha', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
    
    # Gráfico 3: Speedup K2-Tree vs Taha
    ax = axes[1, 0]
    
    if 'Taha (Baseline)' in results and 'K2-Tree MaxHeap V2' in results:
        res_taha = results['Taha (Baseline)']
        res_k2tree = results['K2-Tree MaxHeap V2']
        
        taha_times = res_taha['a'] * (points_range ** res_taha['b'])
        k2tree_times = res_k2tree['a'] * (points_range ** res_k2tree['b'])
        speedup_k2tree = taha_times / k2tree_times
        
        ax.plot(points_range, speedup_k2tree, linewidth=2.5, color='#45B7D1', label='Speedup')
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
        
        ax.fill_between(points_range, 1, speedup_k2tree, alpha=0.2, color='#45B7D1')
        
        ax.set_xlabel('Número de Puntos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (Taha / K2-Tree)', fontsize=12, fontweight='bold')
        ax.set_title('Speedup K2-Tree vs Taha', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
    
    # Gráfico 4: Speedup Kamata vs K2-Tree
    ax = axes[1, 1]
    
    if 'Kamata' in results and 'K2-Tree MaxHeap V2' in results:
        res_kamata = results['Kamata']
        res_k2tree = results['K2-Tree MaxHeap V2']
        
        kamata_times = res_kamata['a'] * (points_range ** res_kamata['b'])
        k2tree_times = res_k2tree['a'] * (points_range ** res_k2tree['b'])
        speedup_k2tree_over_kamata = kamata_times / k2tree_times
        
        ax.plot(points_range, speedup_k2tree_over_kamata, linewidth=2.5, color='#FF9800', label='Speedup')
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
        
        # Llenar área donde K2-Tree es mejor que Kamata
        better_indices = speedup_k2tree_over_kamata > 1
        ax.fill_between(points_range[better_indices], 1, speedup_k2tree_over_kamata[better_indices], 
                        alpha=0.2, color='#FF9800', label='K2-Tree más rápido')
        worse_indices = speedup_k2tree_over_kamata <= 1
        ax.fill_between(points_range[worse_indices], speedup_k2tree_over_kamata[worse_indices], 1,
                        alpha=0.2, color='#FF0000', label='Kamata más rápido')
        
        ax.set_xlabel('Número de Puntos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (K2-Tree / Kamata)', fontsize=12, fontweight='bold')
        ax.set_title('Comparación: K2-Tree vs Kamata', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_file = output_dir / 'breakeven_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de puntos de equilibrio guardado: {output_file}")
    plt.close()


def create_cost_comparison_table(results: dict):
    """Crea tabla comparativa de costos para diferentes tamaños"""
    print("\n" + "="*100)
    print("TABLA COMPARATIVA DE COSTOS (TIEMPOS PREDICHOS)")
    print("="*100 + "\n")
    
    # Escenarios de tamaño (puntos)
    scenarios = {
        '100 puntos': 100,
        '500 puntos': 500,
        '1,000 puntos': 1000,
        '5,000 puntos': 5000,
        '10,000 puntos': 10000,
        '50,000 puntos': 50000,
    }
    
    # Crear tabla
    data = []
    for scenario_name, num_points in scenarios.items():
        row = {'Escenario': scenario_name, 'Puntos': f"{num_points:,}"}
        
        for algo_name, res in results.items():
            if 'a' in res and 'b' in res:
                predicted_time = res['a'] * (num_points ** res['b'])
                row[algo_name] = f"{predicted_time:.3f} ms"
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    print(df.to_string(index=False))
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Predicciones avanzadas de escalabilidad',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python scalability_predictions.py \\
    --csv hausdorff_benchmark_raw.csv \\
    --output analysis/
        """
    )
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Archivo CSV del benchmark')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Validar archivo
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: No se encuentra {csv_path}")
        return False
    
    output_dir = Path(args.output) if args.output else csv_path.parent / 'scalability'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*100}")
    print(f"ANÁLISIS AVANZADO DE ESCALABILIDAD")
    print(f"{'='*100}\n")
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    df['estimated_points'] = df['slices_with_lesion'] * 800
    
    # Calcular parámetros de escalabilidad
    from scipy.stats import linregress
    
    results = {}
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo].copy()
        
        x = np.log10(algo_data['estimated_points'].values)
        y = np.log10(algo_data['mean_time_ms'].values)
        
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) >= 3:
            try:
                slope, intercept, r_value, _, _ = linregress(x, y)
                a = 10 ** intercept
                b = slope
                results[algo] = {'a': a, 'b': b, 'r_squared': r_value**2}
            except:
                pass
    
    # Crear tablas y gráficos
    create_prediction_tables(results)
    create_cost_comparison_table(results)
    create_breakeven_analysis(results, output_dir)
    
    print(f"\n{'='*100}")
    print(f"✓ Análisis avanzado completado")
    print(f"✓ Resultados guardados en: {output_dir}")
    print(f"{'='*100}\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
