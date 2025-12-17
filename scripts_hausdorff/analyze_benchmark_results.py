#!/usr/bin/env python3
"""
Análisis Detallado de Resultados del Benchmark

Script complementario para explorar los resultados generados por
benchmark_hausdorff_algorithms.py con visualizaciones y análisis estadísticos.

Uso:
    python analyze_benchmark_results.py --csv hausdorff_benchmark_raw.csv --output analysis/
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_results(csv_path: str):
    """Carga resultados del CSV de benchmark"""
    df = pd.read_csv(csv_path)
    return df


def load_statistics(json_path: str):
    """Carga estadísticas del JSON"""
    with open(json_path, 'r') as f:
        stats = json.load(f)
    return stats


def analyze_per_patient(df: pd.DataFrame):
    """Analiza variabilidad por paciente"""
    print("\n" + "="*80)
    print("ANÁLISIS POR PACIENTE")
    print("="*80 + "\n")
    
    # Pivot para ver tiempos por paciente y algoritmo
    pivot = df.pivot_table(
        values='mean_time_ms',
        index='patient_id',
        columns='algorithm',
        aggfunc='first'
    )
    
    # Calcular speedup relativo por paciente
    pivot['Kamata_vs_Taha'] = pivot['Taha (Baseline)'] / pivot['Kamata']
    pivot['K2Tree_vs_Taha'] = pivot['Taha (Baseline)'] / pivot['K2-Tree MaxHeap V2']
    
    # Estadísticas
    print(f"Speedup Kamata vs Taha:")
    print(f"  Mín:   {pivot['Kamata_vs_Taha'].min():.2f}x")
    print(f"  Máx:   {pivot['Kamata_vs_Taha'].max():.2f}x")
    print(f"  Media: {pivot['Kamata_vs_Taha'].mean():.2f}x")
    print(f"  Median: {pivot['Kamata_vs_Taha'].median():.2f}x")
    
    print(f"\nSpeedup K2-Tree vs Taha:")
    print(f"  Mín:   {pivot['K2Tree_vs_Taha'].min():.2f}x")
    print(f"  Máx:   {pivot['K2Tree_vs_Taha'].max():.2f}x")
    print(f"  Media: {pivot['K2Tree_vs_Taha'].mean():.2f}x")
    print(f"  Median: {pivot['K2Tree_vs_Taha'].median():.2f}x")
    
    # Top 5 pacientes donde Kamata es más eficiente
    print(f"\nTop 5 pacientes con mayor speedup (Kamata vs Taha):")
    top5 = pivot['Kamata_vs_Taha'].nlargest(5)
    for idx, (patient, speedup) in enumerate(top5.items(), 1):
        print(f"  {idx}. Paciente {patient}: {speedup:.2f}x")
    
    return pivot


def analyze_per_slice(df: pd.DataFrame):
    """Analiza complejidad por número de slices procesados"""
    print("\n" + "="*80)
    print("ANÁLISIS POR COMPLEJIDAD (número de slices con lesión)")
    print("="*80 + "\n")
    
    # Agrupar por complejidad
    df['complexity_bin'] = pd.cut(df['slices_with_lesion'], 
                                   bins=[0, 5, 10, 15, 20, np.inf],
                                   labels=['1-5', '6-10', '11-15', '16-20', '20+'])
    
    complexity_analysis = df.groupby('complexity_bin')['mean_time_ms'].agg(['mean', 'std', 'min', 'max', 'count'])
    
    print("\nTiempos por rango de complejidad (slices con lesión):")
    print(complexity_analysis)
    
    return df


def plot_violin_by_algorithm(df: pd.DataFrame, output_dir: Path):
    """Gráfico violin mostrando distribución de tiempos"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.violinplot(data=df, x='algorithm', y='mean_time_ms', ax=ax, palette='Set2')
    ax.set_ylabel('Tiempo por Slice (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Algoritmo', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Tiempos de Ejecución\n(Violin Plot)', 
                 fontsize=13, fontweight='bold')
    
    # Escala logarítmica si hay mucha variación
    if df['mean_time_ms'].max() / df['mean_time_ms'].min() > 10:
        ax.set_yscale('log')
        ax.set_ylabel('Tiempo por Slice (ms) [escala log]', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'benchmark_violin_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico violin guardado: {output_file}")
    plt.close()


def plot_speedup_scatter(df: pd.DataFrame, output_dir: Path):
    """Gráfico scatter de speedup por paciente"""
    pivot = df.pivot_table(
        values='mean_time_ms',
        index='patient_id',
        columns='algorithm',
        aggfunc='first'
    )
    
    pivot['Kamata_speedup'] = pivot['Taha (Baseline)'] / pivot['Kamata']
    pivot['K2Tree_speedup'] = pivot['Taha (Baseline)'] / pivot['K2-Tree MaxHeap V2']
    pivot['slices'] = df[df['algorithm'] == 'Taha (Baseline)'].set_index('patient_id')['slices_with_lesion']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scatter plots
    ax.scatter(pivot.index, pivot['Kamata_speedup'], label='Kamata', s=100, alpha=0.6, color='#4ECDC4')
    ax.scatter(pivot.index, pivot['K2Tree_speedup'], label='K2-Tree MaxHeap V2', s=100, alpha=0.6, color='#45B7D1')
    
    # Línea baseline
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
    
    ax.set_xlabel('Patient ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup vs Taha', fontsize=12, fontweight='bold')
    ax.set_title('Speedup por Paciente', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Rotar labels si hay muchos pacientes
    if len(pivot) > 20:
        ax.set_xticks(range(0, len(pivot), max(1, len(pivot)//10)))
    
    plt.tight_layout()
    output_file = output_dir / 'benchmark_speedup_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico scatter guardado: {output_file}")
    plt.close()


def plot_complexity_vs_performance(df: pd.DataFrame, output_dir: Path):
    """Analiza cómo el número de slices afecta el rendimiento"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Tiempo vs Complejidad por algoritmo
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo].sort_values('slices_with_lesion')
        axes[0].scatter(algo_data['slices_with_lesion'], algo_data['mean_time_ms'], 
                       label=algo, s=80, alpha=0.6)
    
    axes[0].set_xlabel('Slices con Lesión', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Tiempo Promedio (ms)', fontsize=11, fontweight='bold')
    axes[0].set_title('Tiempo vs Complejidad', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Speedup vs Complejidad
    pivot = df.pivot_table(
        values='mean_time_ms',
        index='patient_id',
        columns='algorithm',
        aggfunc='first'
    )
    pivot['Kamata_speedup'] = pivot['Taha (Baseline)'] / pivot['Kamata']
    
    slices_data = df[df['algorithm'] == 'Taha (Baseline)'][['patient_id', 'slices_with_lesion']].set_index('patient_id')
    pivot = pivot.join(slices_data)
    
    axes[1].scatter(pivot['slices_with_lesion'], pivot['Kamata_speedup'], 
                   s=100, alpha=0.6, color='#4ECDC4')
    axes[1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline')
    
    axes[1].set_xlabel('Slices con Lesión', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Speedup (Kamata)', fontsize=11, fontweight='bold')
    axes[1].set_title('Speedup vs Complejidad', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'benchmark_complexity_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de análisis de complejidad guardado: {output_file}")
    plt.close()


def generate_summary_report(df: pd.DataFrame, stats: dict, output_dir: Path):
    """Genera reporte de texto con resumen de análisis"""
    report = []
    report.append("="*80)
    report.append("BENCHMARK DE HAUSDORFF: ANÁLISIS DETALLADO")
    report.append("="*80)
    report.append("")
    
    report.append("RESUMEN EJECUTIVO")
    report.append("-" * 80)
    report.append(f"Total de pacientes procesados: {df['patient_id'].nunique()}")
    report.append(f"Total de muestras: {len(df)}")
    report.append("")
    
    report.append("ALGORITMOS EVALUADOS:")
    for algo, algo_stats in stats.items():
        report.append(f"  • {algo}")
        report.append(f"    - Tiempo promedio: {algo_stats['mean_time_ms']:.4f} ± {algo_stats['std_time_ms']:.4f} ms")
        report.append(f"    - Speedup: {algo_stats['speedup']:.2f}x")
        report.append(f"    - Muestras: {algo_stats['samples']}")
    report.append("")
    
    # Análisis por paciente
    report.append("VARIABILIDAD POR PACIENTE:")
    pivot = df.pivot_table(
        values='mean_time_ms',
        index='patient_id',
        columns='algorithm',
        aggfunc='first'
    )
    pivot['Kamata_speedup'] = pivot['Taha (Baseline)'] / pivot['Kamata']
    
    report.append(f"  Speedup Kamata vs Taha:")
    report.append(f"    - Mínimo: {pivot['Kamata_speedup'].min():.2f}x")
    report.append(f"    - Máximo: {pivot['Kamata_speedup'].max():.2f}x")
    report.append(f"    - Promedio: {pivot['Kamata_speedup'].mean():.2f}x")
    report.append(f"    - Mediana: {pivot['Kamata_speedup'].median():.2f}x")
    report.append("")
    
    # Casos extremos
    report.append("CASOS ESPECIALES:")
    
    # Pacientes más complejos
    complex_patients = df[df['algorithm'] == 'Taha (Baseline)'].nlargest(3, 'slices_with_lesion')
    report.append(f"  Pacientes más complejos (más slices con lesión):")
    for _, row in complex_patients.iterrows():
        report.append(f"    - {row['patient_id']}: {row['slices_with_lesion']} slices, {row['mean_time_ms']:.2f} ms (Taha)")
    
    report.append("")
    
    # Pacientes con mayor speedup
    fastest_patients = pivot['Kamata_speedup'].nlargest(3)
    report.append(f"  Pacientes con mayor speedup (Kamata):")
    for patient, speedup in fastest_patients.items():
        report.append(f"    - {patient}: {speedup:.2f}x")
    
    report.append("")
    report.append("="*80)
    report.append("FIN DEL REPORTE")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Guardar a archivo
    report_file = output_dir / 'BENCHMARK_ANALYSIS_REPORT.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Reporte guardado: {report_file}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description='Análisis detallado de resultados de benchmark'
    )
    parser.add_argument('--csv', type=str, required=True,
                       help='Archivo CSV con resultados crudos')
    parser.add_argument('--json', type=str, default=None,
                       help='Archivo JSON con estadísticas (opcional)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Validar archivos
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: No se encuentra {csv_path}")
        return False
    
    output_dir = Path(args.output) if args.output else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    print(f"Cargando datos desde: {csv_path}")
    df = load_results(str(csv_path))
    
    stats = {}
    if args.json and Path(args.json).exists():
        stats = load_statistics(args.json)
    
    # Análisis
    pivot = analyze_per_patient(df)
    df_with_complexity = analyze_per_slice(df)
    
    # Visualizaciones
    print("\nGenerando visualizaciones...")
    plot_violin_by_algorithm(df, output_dir)
    plot_speedup_scatter(df, output_dir)
    plot_complexity_vs_performance(df, output_dir)
    
    # Reporte
    if stats:
        generate_summary_report(df, stats, output_dir)
    
    print(f"\n✓ Análisis completado. Resultados en: {output_dir}")
    return True


if __name__ == '__main__':
    main()
