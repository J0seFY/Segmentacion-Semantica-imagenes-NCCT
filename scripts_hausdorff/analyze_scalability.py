#!/usr/bin/env python3
"""
Análisis de Escalabilidad: Tiempo de Ejecución vs Número de Puntos

Script para analizar cómo escalan los algoritmos de Hausdorff en función
de la cantidad de puntos en las máscaras. Esto permite determinar:
- Complejidad computacional teórica vs práctica
- Comportamiento O(n log n), O(n²), etc.
- Puntos de quiebre en rendimiento

Uso:
    python analyze_scalability.py --csv hausdorff_benchmark_raw.csv --output analysis/
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import linregress
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def load_benchmark_data(csv_path: str) -> pd.DataFrame:
    """Carga datos del benchmark"""
    df = pd.read_csv(csv_path)
    return df


def extract_points_from_csv(csv_path: str, real_points_csv: str = None) -> dict:
    """
    Extrae información de puntos desde el CSV.
    Si real_points_csv está disponible, usa datos reales.
    Caso contrario, estima basados en slices_with_lesion.
    """
    df = pd.read_csv(csv_path)
    
    # Intentar cargar datos reales de puntos
    if real_points_csv and Path(real_points_csv).exists():
        print(f"✓ Usando datos REALES de puntos desde: {real_points_csv}")
        real_points_df = pd.read_csv(real_points_csv)
        
        # Merge con datos de benchmark
        df = df.merge(
            real_points_df[['patient_id', 'total_points_in_lesion_slices', 'avg_points_per_slice']],
            on='patient_id',
            how='left'
        )
        
        # Usar puntos reales
        df['estimated_points'] = df['total_points_in_lesion_slices']
        
        print(f"  Promedio de puntos reales: {df['estimated_points'].mean():.1f}")
        print(f"  Rango: {df['estimated_points'].min():.0f} - {df['estimated_points'].max():.0f}")
    else:
        print(f"⚠ Usando estimaciones de puntos (no hay datos reales)")
        # Estimación conservadora
        df['estimated_points'] = df['slices_with_lesion'] * 800
    
    return df


def analyze_complexity_scaling(df: pd.DataFrame):
    """Analiza cómo escala el tiempo con número de puntos"""
    print("\n" + "="*80)
    print("ANÁLISIS DE ESCALABILIDAD: TIEMPO vs NÚMERO DE PUNTOS")
    print("="*80 + "\n")
    
    # Agrupar por algoritmo
    algorithms = df['algorithm'].unique()
    
    results = {}
    
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo].copy()
        
        # Remover outliers extremos
        Q1 = algo_data['mean_time_ms'].quantile(0.25)
        Q3 = algo_data['mean_time_ms'].quantile(0.75)
        IQR = Q3 - Q1
        
        algo_data_clean = algo_data[
            (algo_data['mean_time_ms'] >= Q1 - 1.5*IQR) & 
            (algo_data['mean_time_ms'] <= Q3 + 1.5*IQR)
        ]
        
        if len(algo_data_clean) < 3:
            continue
        
        # Calcular regresión lineal en escala log-log
        # log(t) = log(a) + b*log(n)  =>  t = a*n^b
        x = np.log10(algo_data_clean['estimated_points'].values)
        y = np.log10(algo_data_clean['mean_time_ms'].values)
        
        # Omitir inf/nan
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) < 3:
            continue
        
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            a = 10 ** intercept
            b = slope
            
            results[algo] = {
                'a': a,
                'b': b,
                'r_squared': r_value**2,
                'complexity': _get_complexity_label(b),
                'points': algo_data_clean['estimated_points'].values,
                'times': algo_data_clean['mean_time_ms'].values,
                'num_samples': len(algo_data_clean)
            }
            
        except Exception as e:
            print(f"Error en {algo}: {e}")
            continue
    
    # Mostrar resultados
    print(f"{'Algoritmo':<30} {'Complejidad':<15} {'a':<12} {'b':<8} {'R²':<10} {'Muestras':<10}")
    print("-" * 90)
    
    for algo, res in sorted(results.items()):
        print(f"{algo:<30} {res['complexity']:<15} {res['a']:.2e}     {res['b']:.2f}    {res['r_squared']:.3f}    {res['num_samples']:<10}")
    
    print("\n" + "="*80)
    print("EXPLICACIÓN:")
    print("="*80)
    print("""
t(n) = a * n^b

- b ≈ 1.0:  Complejidad lineal O(n)
- b ≈ 1.5:  Complejidad O(n^1.5) [como sqrt(n²)]
- b ≈ 2.0:  Complejidad cuadrática O(n²)
- b ≈ 0.5:  Complejidad sublineal O(√n)

R²: Bondad del ajuste (1.0 = ajuste perfecto, 0.5 = aceptable)
""")
    
    return results


def _get_complexity_label(b: float) -> str:
    """Etiqueta la complejidad basada en exponente b"""
    if b < 0.7:
        return "O(√n) o mejor"
    elif b < 1.2:
        return "O(n) lineal"
    elif b < 1.7:
        return "O(n^1.5)"
    elif b < 2.2:
        return "O(n²) cuadrática"
    else:
        return "O(n²+) peor"


def plot_scalability_comparison(df: pd.DataFrame, results: dict, output_dir: Path, using_real_data: bool = False):
    """Gráfico comparativo de escalabilidad"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Etiqueta del eje X según el tipo de datos
    xlabel = 'Número de Puntos' if using_real_data else 'Número de Puntos (estimado)'
    
    colors = {'Taha (Baseline)': '#FF6B6B', 'Kamata': '#4ECDC4', 'K2-Tree MaxHeap V2': '#45B7D1'}
    
    # Gráfico 1: Escala lineal
    ax = axes[0]
    for algo, res in results.items():
        ax.scatter(res['points'], res['times'], s=100, alpha=0.6, 
                  label=f"{algo} (b={res['b']:.2f})", color=colors.get(algo, '#95A5A6'))
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiempo de Ejecución (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Escalabilidad: Escala Lineal', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: Escala logarítmica (log-log)
    ax = axes[1]
    for algo, res in results.items():
        ax.scatter(res['points'], res['times'], s=100, alpha=0.6, 
                  label=f"{algo} (b={res['b']:.2f}, R²={res['r_squared']:.3f})", 
                  color=colors.get(algo, '#95A5A6'))
        
        # Ajuste teórico
        points_sorted = np.sort(res['points'])
        times_fitted = res['a'] * (points_sorted ** res['b'])
        ax.plot(points_sorted, times_fitted, linestyle='--', linewidth=2, 
               color=colors.get(algo, '#95A5A6'), alpha=0.7)
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiempo de Ejecución (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Escalabilidad: Escala Logarítmica Log-Log (con ajuste)', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_file = output_dir / 'scalability_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de escalabilidad guardado: {output_file}")
    plt.close()


def plot_speedup_vs_points(df: pd.DataFrame, output_dir: Path):
    """Gráfico de speedup en función de número de puntos"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'Kamata': '#4ECDC4', 'K2-Tree MaxHeap V2': '#45B7D1'}
    
    # Gráfico 1: Speedup Kamata vs Taha
    ax = axes[0, 0]
    baseline = df[df['algorithm'] == 'Taha (Baseline)'].copy()
    kamata = df[df['algorithm'] == 'Kamata'].copy()
    
    # Merge para emparejar pacientes
    speedup_data_kamata = []
    for idx, base_row in baseline.iterrows():
        pat_id = base_row['patient_id']
        kam_row = kamata[kamata['patient_id'] == pat_id]
        if not kam_row.empty:
            speedup = base_row['mean_time_ms'] / kam_row.iloc[0]['mean_time_ms']
            points = base_row['estimated_points']
            speedup_data_kamata.append({'points': points, 'speedup': speedup})
    
    if speedup_data_kamata:
        speedup_df = pd.DataFrame(speedup_data_kamata)
        ax.scatter(speedup_df['points'], speedup_df['speedup'], s=100, alpha=0.6, 
                  color=colors['Kamata'])
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Tendencia
        z = np.polyfit(np.log10(speedup_df['points']), speedup_df['speedup'], 2)
        p = np.poly1d(z)
        points_sorted = np.sort(speedup_df['points'].values)
        ax.plot(points_sorted, p(np.log10(points_sorted)), linestyle='--', 
               linewidth=2, color=colors['Kamata'], alpha=0.7, label='Tendencia')
    
    ax.set_xlabel('Número de Puntos', fontsize=11, fontweight='bold')
    ax.set_ylabel('Speedup vs Taha', fontsize=11, fontweight='bold')
    ax.set_title('Speedup Kamata vs Complejidad', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Gráfico 2: Speedup K2-Tree vs Taha
    ax = axes[0, 1]
    k2tree = df[df['algorithm'] == 'K2-Tree MaxHeap V2'].copy()
    
    speedup_data_k2tree = []
    for idx, base_row in baseline.iterrows():
        pat_id = base_row['patient_id']
        k2_row = k2tree[k2tree['patient_id'] == pat_id]
        if not k2_row.empty:
            speedup = base_row['mean_time_ms'] / k2_row.iloc[0]['mean_time_ms']
            points = base_row['estimated_points']
            speedup_data_k2tree.append({'points': points, 'speedup': speedup})
    
    if speedup_data_k2tree:
        speedup_df_k2 = pd.DataFrame(speedup_data_k2tree)
        ax.scatter(speedup_df_k2['points'], speedup_df_k2['speedup'], s=100, alpha=0.6, 
                  color=colors['K2-Tree MaxHeap V2'])
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Número de Puntos', fontsize=11, fontweight='bold')
    ax.set_ylabel('Speedup vs Taha', fontsize=11, fontweight='bold')
    ax.set_title('Speedup K2-Tree vs Complejidad', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Tiempo por punto (Taha)
    ax = axes[1, 0]
    baseline['time_per_point'] = baseline['mean_time_ms'] / baseline['estimated_points']
    ax.scatter(baseline['estimated_points'], baseline['time_per_point'], s=100, alpha=0.6, 
              color='#FF6B6B')
    ax.set_xlabel('Número de Puntos', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tiempo por Punto (ms/punto)', fontsize=11, fontweight='bold')
    ax.set_title('Eficiencia Taha: Tiempo/Punto', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    # Gráfico 4: Tiempo por punto (todos los algoritmos)
    ax = axes[1, 1]
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo].copy()
        algo_data['time_per_point'] = algo_data['mean_time_ms'] / algo_data['estimated_points']
        
        color = colors.get(algo, '#95A5A6')
        if algo == 'Taha (Baseline)':
            color = '#FF6B6B'
        
        ax.scatter(algo_data['estimated_points'], algo_data['time_per_point'], 
                  s=100, alpha=0.6, label=algo, color=color)
    
    ax.set_xlabel('Número de Puntos', fontsize=11, fontweight='bold')
    ax.set_ylabel('Tiempo por Punto (ms/punto)', fontsize=11, fontweight='bold')
    ax.set_title('Comparación: Eficiencia (Tiempo/Punto)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'speedup_vs_points.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de speedup vs puntos guardado: {output_file}")
    plt.close()


def generate_scalability_report(df: pd.DataFrame, results: dict, output_dir: Path):
    """Genera reporte de texto sobre escalabilidad"""
    report = []
    report.append("="*80)
    report.append("ANÁLISIS DE ESCALABILIDAD: HAUSDORFF ALGORITHMS")
    report.append("="*80)
    report.append("")
    
    report.append("MODELO MATEMÁTICO DE ESCALABILIDAD")
    report.append("-" * 80)
    report.append("""
Cada algoritmo sigue un modelo de escalabilidad: t(n) = a * n^b

Donde:
  t(n)  = Tiempo de ejecución en función de número de puntos
  n     = Número total de puntos en las máscaras
  a     = Constante que refleja overhead y optimizaciones
  b     = Exponente que indica la complejidad (crecimiento con n)
  R²    = Bondad del ajuste (1.0 = perfecto, 0.5 = bueno, 0.3 = aceptable)
""")
    report.append("")
    
    report.append("RESULTADOS POR ALGORITMO:")
    report.append("-" * 80)
    
    for algo in ['Taha (Baseline)', 'Kamata', 'K2-Tree MaxHeap V2']:
        if algo in results:
            res = results[algo]
            report.append(f"\n{algo}")
            report.append(f"  Complejidad teórica: {res['complexity']}")
            report.append(f"  Ecuación: t(n) = {res['a']:.2e} * n^{res['b']:.2f}")
            report.append(f"  Bondad del ajuste (R²): {res['r_squared']:.3f}")
            report.append(f"  Muestras analizadas: {res['num_samples']}")
            
            # Interpretación del exponente
            if res['b'] < 1.2:
                report.append(f"  ➜ El tiempo crece LINEALMENTE con puntos")
                report.append(f"  ➜ Escalable: 2x puntos ≈ 2x tiempo")
            elif res['b'] < 1.7:
                report.append(f"  ➜ El tiempo crece entre LINEAR y CUADRATICO")
                report.append(f"  ➜ Moderadamente escalable: 2x puntos ≈ {2**res['b']:.1f}x tiempo")
            else:
                report.append(f"  ➜ El tiempo crece CUADRATICAMENTE o peor")
                report.append(f"  ➜ Mal escalable: 2x puntos ≈ {2**res['b']:.1f}x tiempo")
    
    report.append("\n" + "="*80)
    report.append("ANÁLISIS COMPARATIVO")
    report.append("="*80)
    
    if 'Taha (Baseline)' in results and 'Kamata' in results:
        res_taha = results['Taha (Baseline)']
        res_kamata = results['Kamata']
        
        report.append(f"\nTaha vs Kamata:")
        report.append(f"  Exponente Taha: {res_taha['b']:.2f}")
        report.append(f"  Exponente Kamata: {res_kamata['b']:.2f}")
        report.append(f"  Diferencia en exponente: {res_taha['b'] - res_kamata['b']:.2f}")
        
        if res_kamata['b'] < res_taha['b']:
            report.append(f"  ➜ Kamata escala MEJOR que Taha")
            report.append(f"  ➜ Ventaja aumenta con más puntos")
        else:
            report.append(f"  ➜ Taha escala mejor (sorpresa)")
    
    if 'Taha (Baseline)' in results and 'K2-Tree MaxHeap V2' in results:
        res_taha = results['Taha (Baseline)']
        res_k2tree = results['K2-Tree MaxHeap V2']
        
        report.append(f"\nTaha vs K2-Tree MaxHeap V2:")
        report.append(f"  Exponente Taha: {res_taha['b']:.2f}")
        report.append(f"  Exponente K2-Tree: {res_k2tree['b']:.2f}")
        report.append(f"  Diferencia en exponente: {res_taha['b'] - res_k2tree['b']:.2f}")
        
        if res_k2tree['b'] < res_taha['b']:
            report.append(f"  ➜ K2-Tree escala MEJOR que Taha")
        else:
            report.append(f"  ➜ Taha escala mejor")
    
    report.append("\n" + "="*80)
    report.append("RECOMENDACIONES")
    report.append("="*80)
    report.append("""
1. Para PEQUEÑOS volúmenes (< 1000 puntos por slice):
   - Todos los algoritmos tienen rendimiento similar
   - Elegir por precisión o características específicas

2. Para VOLÚMENES MEDIANOS (1000-10000 puntos):
   - Kamata probablemente es más eficiente
   - K2-Tree también viable si estructuras datos necesarias

3. Para VOLÚMENES GRANDES (> 10000 puntos):
   - Diferencias de escalabilidad se hacen críticas
   - El algoritmo con menor exponente (mejor escalabilidad) gana
   - Considerar paralelización o GPU si es necesario

4. Consideraciones prácticas:
   - Benchmark en tu hardware específico (CPU, memoria)
   - Validar que los speedups teóricos se cumplen en práctica
   - Monitor de memoria también importante para n² algorithms
""")
    
    report.append("\n" + "="*80)
    report.append(f"Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Guardar reporte
    report_file = output_dir / 'SCALABILITY_ANALYSIS_REPORT.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Reporte de escalabilidad guardado: {report_file}")
    
    # También mostrar en consola
    print("\n" + '\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description='Análisis de Escalabilidad de Algoritmos de Hausdorff',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python analyze_scalability.py \\
    --csv hausdorff_benchmark_raw.csv \\
    --output analysis/
        """
    )
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Archivo CSV del benchmark')
    parser.add_argument('--real-points', type=str, default=None,
                       help='Archivo CSV con datos reales de puntos (opcional)')
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
    
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE ESCALABILIDAD - ALGORITMOS HAUSDORFF")
    print(f"{'='*80}\n")
    
    # Cargar datos
    print(f"Cargando datos del benchmark...")
    df = load_benchmark_data(str(csv_path))
    
    # Intentar usar datos reales si están disponibles
    real_points_path = args.real_points if args.real_points else csv_path.parent / 'real_points_data.csv'
    df = extract_points_from_csv(str(csv_path), str(real_points_path) if real_points_path else None)
    
    print(f"✓ Datos cargados: {len(df)} registros\n")
    
    # Analizar escalabilidad
    results = analyze_complexity_scaling(df)
    
    if not results:
        print("Error: No se pudieron analizar los datos")
        return False
    
    # Generar gráficos
    print(f"\nGenerando visualizaciones...")
    using_real_data = args.real_points or (csv_path.parent / 'real_points_data.csv').exists()
    plot_scalability_comparison(df, results, output_dir, using_real_data)
    plot_speedup_vs_points(df, output_dir)
    
    # Generar reporte
    generate_scalability_report(df, results, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✓ Análisis de escalabilidad completado")
    print(f"✓ Resultados guardados en: {output_dir}")
    print(f"{'='*80}\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
