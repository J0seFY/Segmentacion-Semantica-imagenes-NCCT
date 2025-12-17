#!/usr/bin/env python3
"""
Herramienta para Comparar Múltiples Benchmarks

Permite comparar resultados de benchmark ejecutados en diferentes datasets
o con diferentes configuraciones.

Uso:
    python compare_benchmarks.py \
      --benchmark1 resultados1/hausdorff_benchmark_stats.json \
      --benchmark2 resultados2/hausdorff_benchmark_stats.json \
      --output comparacion/
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_stats(json_path: str) -> dict:
    """Carga estadísticas del benchmark"""
    with open(json_path, 'r') as f:
        return json.load(f)


def compare_benchmarks(bench1: dict, bench2: dict, name1: str = "Benchmark 1", 
                       name2: str = "Benchmark 2") -> pd.DataFrame:
    """Compara dos benchmarks"""
    
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN: {name1} vs {name2}")
    print(f"{'='*80}\n")
    
    # Crear DataFrame de comparación
    comparison = []
    
    for algo in bench1.keys():
        if algo in bench2:
            b1 = bench1[algo]
            b2 = bench2[algo]
            
            time_diff = b2['mean_time_ms'] - b1['mean_time_ms']
            time_pct = (time_diff / b1['mean_time_ms']) * 100 if b1['mean_time_ms'] > 0 else 0
            speedup_diff = b2['speedup'] - b1['speedup']
            
            comparison.append({
                'algoritmo': algo,
                'tiempo_b1_ms': b1['mean_time_ms'],
                'tiempo_b2_ms': b2['mean_time_ms'],
                'diferencia_ms': time_diff,
                'diferencia_pct': time_pct,
                'speedup_b1': b1['speedup'],
                'speedup_b2': b2['speedup'],
                'speedup_diff': speedup_diff,
                'muestras_b1': b1['samples'],
                'muestras_b2': b2['samples']
            })
    
    df = pd.DataFrame(comparison)
    
    # Mostrar tabla
    print(f"{'Algoritmo':<25} {'Tiempo B1':<15} {'Tiempo B2':<15} {'Diferencia':<15}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        algo = row['algoritmo']
        t1 = f"{row['tiempo_b1_ms']:.2f} ms"
        t2 = f"{row['tiempo_b2_ms']:.2f} ms"
        diff = f"{row['diferencia_pct']:+.1f}%"
        print(f"{algo:<25} {t1:<15} {t2:<15} {diff:<15}")
    
    print("\n")
    print(f"{'Algoritmo':<25} {'Speedup B1':<15} {'Speedup B2':<15} {'Diferencia':<15}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        algo = row['algoritmo']
        s1 = f"{row['speedup_b1']:.2f}x"
        s2 = f"{row['speedup_b2']:.2f}x"
        diff = f"{row['speedup_diff']:+.2f}x"
        print(f"{algo:<25} {s1:<15} {s2:<15} {diff:<15}")
    
    return df


def plot_benchmark_comparison(bench1: dict, bench2: dict, name1: str, 
                              name2: str, output_dir: Path):
    """Genera gráfico comparativo"""
    
    algos = list(bench1.keys())
    
    # Datos
    times_b1 = [bench1[algo]['mean_time_ms'] for algo in algos]
    times_b2 = [bench2[algo]['mean_time_ms'] for algo in algos]
    
    speedups_b1 = [bench1[algo]['speedup'] for algo in algos]
    speedups_b2 = [bench2[algo]['speedup'] for algo in algos]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Tiempos
    x = np.arange(len(algos))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, times_b1, width, label=name1, alpha=0.8, color='#FF6B6B')
    bars2 = axes[0].bar(x + width/2, times_b2, width, label=name2, alpha=0.8, color='#4ECDC4')
    
    axes[0].set_ylabel('Tiempo Promedio (ms)', fontsize=11, fontweight='bold')
    axes[0].set_title('Comparación de Tiempos', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algos, rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Anotaciones
    for bar in bars1 + bars2:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Gráfico 2: Speedups
    bars3 = axes[1].bar(x - width/2, speedups_b1, width, label=name1, alpha=0.8, color='#FF6B6B')
    bars4 = axes[1].bar(x + width/2, speedups_b2, width, label=name2, alpha=0.8, color='#4ECDC4')
    
    axes[1].set_ylabel('Speedup (x)', fontsize=11, fontweight='bold')
    axes[1].set_title('Comparación de Speedup', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algos, rotation=15, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Anotaciones
    for bar in bars3 + bars4:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}x',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_file = output_dir / 'benchmark_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de comparación guardado: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Comparar múltiples benchmarks'
    )
    parser.add_argument('--benchmark1', type=str, required=True,
                       help='Archivo JSON del primer benchmark')
    parser.add_argument('--benchmark2', type=str, required=True,
                       help='Archivo JSON del segundo benchmark')
    parser.add_argument('--name1', type=str, default='Benchmark 1',
                       help='Nombre del primer benchmark')
    parser.add_argument('--name2', type=str, default='Benchmark 2',
                       help='Nombre del segundo benchmark')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Validar archivos
    if not Path(args.benchmark1).exists():
        print(f"Error: No se encuentra {args.benchmark1}")
        return False
    
    if not Path(args.benchmark2).exists():
        print(f"Error: No se encuentra {args.benchmark2}")
        return False
    
    output_dir = Path(args.output) if args.output else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar benchmarks
    print(f"Cargando benchmarks...")
    bench1 = load_benchmark_stats(args.benchmark1)
    bench2 = load_benchmark_stats(args.benchmark2)
    
    # Comparar
    df_comparison = compare_benchmarks(bench1, bench2, args.name1, args.name2)
    
    # Gráfico
    plot_benchmark_comparison(bench1, bench2, args.name1, args.name2, output_dir)
    
    # Guardar CSV
    csv_file = output_dir / 'benchmark_comparison.csv'
    df_comparison.to_csv(csv_file, index=False)
    print(f"✓ Comparación guardada: {csv_file}")
    
    print(f"\n✓ Análisis comparativo completado.")
    return True


if __name__ == '__main__':
    main()
