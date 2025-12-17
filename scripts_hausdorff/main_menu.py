#!/usr/bin/env python3
"""
Menú Principal - Suite de Benchmark de Hausdorff

Interfaz unificada para ejecutar todas las herramientas de benchmark.
"""

import sys
import subprocess
from pathlib import Path


def print_banner():
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║        BENCHMARK DE ALGORITMOS DE HAUSDORFF                  ║
║   Comparación de Eficiencia Computacional (3D → 2D)          ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)


def print_menu():
    print("""
┌──────────────────────────────────────────────────────────────┐
│ MENÚ PRINCIPAL                                               │
└──────────────────────────────────────────────────────────────┘

[1] Ejecutar Benchmark Completo
    - Procesa predicciones y Ground Truth
    - Mide tiempos de ejecución
    - Genera reportes y gráficos

[2] Analizar Resultados Existentes
    - Carga CSV/JSON previos
    - Análisis estadístico detallado
    - Genera visualizaciones adicionales

[3] Comparar Dos Benchmarks
    - Compara resultados de dos ejecuciones
    - Tabla comparativa y gráficos

[4] Generar Datos de Prueba Sintéticos
    - Crea volúmenes 3D sintéticos
    - Útil para validación rápida

[5] Ejecutar Demostración Rápida
    - Genera datos sintéticos
    - Ejecuta benchmark
    - Muestra resultados

[6] Mostrar Documentación
    - Explicación detallada de cada script
    - Ejemplos de uso
    - Troubleshooting

[0] Salir

    """)


def menu_benchmark():
    """Opción 1: Ejecutar benchmark"""
    print("\n" + "="*60)
    print("OPCIÓN 1: Ejecutar Benchmark Completo")
    print("="*60 + "\n")
    
    pred_dir = input("Directorio con predicciones (.nii.gz): ").strip()
    gt_dir = input("Directorio con Ground Truth (.nii.gz): ").strip()
    output_dir = input("Directorio de salida (Enter = predicciones): ").strip() or None
    
    if not Path(pred_dir).exists() or not Path(gt_dir).exists():
        print("❌ Error: Directorios no existen")
        return False
    
    cmd = [
        'python3', 'benchmark_hausdorff_algorithms.py',
        '--pred', pred_dir,
        '--gt', gt_dir
    ]
    if output_dir:
        cmd.extend(['--output', output_dir])
    
    print(f"\n▶ Ejecutando: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def menu_analyze():
    """Opción 2: Analizar resultados"""
    print("\n" + "="*60)
    print("OPCIÓN 2: Analizar Resultados Existentes")
    print("="*60 + "\n")
    
    csv_file = input("Archivo CSV con resultados crudos: ").strip()
    json_file = input("Archivo JSON con estadísticas (Enter = auto): ").strip() or None
    output_dir = input("Directorio de salida (Enter = CSV directory): ").strip() or None
    
    if not Path(csv_file).exists():
        print("❌ Error: CSV no existe")
        return False
    
    cmd = [
        'python3', 'analyze_benchmark_results.py',
        '--csv', csv_file
    ]
    if json_file:
        cmd.extend(['--json', json_file])
    if output_dir:
        cmd.extend(['--output', output_dir])
    
    print(f"\n▶ Ejecutando: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def menu_compare():
    """Opción 3: Comparar benchmarks"""
    print("\n" + "="*60)
    print("OPCIÓN 3: Comparar Dos Benchmarks")
    print("="*60 + "\n")
    
    bench1 = input("Primer benchmark (JSON): ").strip()
    bench2 = input("Segundo benchmark (JSON): ").strip()
    name1 = input("Nombre del primero (Enter = Benchmark 1): ").strip() or "Benchmark 1"
    name2 = input("Nombre del segundo (Enter = Benchmark 2): ").strip() or "Benchmark 2"
    output_dir = input("Directorio de salida: ").strip() or None
    
    if not Path(bench1).exists() or not Path(bench2).exists():
        print("❌ Error: Archivos JSON no existen")
        return False
    
    cmd = [
        'python3', 'compare_benchmarks.py',
        '--benchmark1', bench1,
        '--benchmark2', bench2,
        '--name1', name1,
        '--name2', name2
    ]
    if output_dir:
        cmd.extend(['--output', output_dir])
    
    print(f"\n▶ Ejecutando: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def menu_generate():
    """Opción 4: Generar datos sintéticos"""
    print("\n" + "="*60)
    print("OPCIÓN 4: Generar Datos de Prueba Sintéticos")
    print("="*60 + "\n")
    
    output_dir = input("Directorio de salida: ").strip()
    num_patients = input("Número de pacientes (Enter = 5): ").strip() or "5"
    slices = input("Slices por paciente (Enter = 30): ").strip() or "30"
    
    if not output_dir:
        print("❌ Error: Directorio requerido")
        return False
    
    cmd = [
        'python3', 'generate_test_data.py',
        '--output', output_dir,
        '--num_patients', num_patients,
        '--slices_per_patient', slices
    ]
    
    print(f"\n▶ Ejecutando: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def menu_demo():
    """Opción 5: Demostración rápida"""
    print("\n" + "="*60)
    print("OPCIÓN 5: Ejecutar Demostración Rápida")
    print("="*60 + "\n")
    
    print("▶ Ejecutando demostración...\n")
    result = subprocess.run(['python3', 'run_benchmark_demo.py'], cwd=Path(__file__).parent)
    return result.returncode == 0


def menu_docs():
    """Opción 6: Mostrar documentación"""
    print("\n" + "="*60)
    print("DOCUMENTACIÓN - SCRIPTS DISPONIBLES")
    print("="*60 + "\n")
    
    docs = """
benchmark_hausdorff_algorithms.py
──────────────────────────────────
Script principal que ejecuta el benchmark completo.

Entrada:
  - Carpeta con predicciones .nii.gz
  - Carpeta con Ground Truth .nii.gz

Salida:
  - CSV con datos crudos
  - JSON con estadísticas agregadas
  - Gráfico de comparación (tiempo y speedup)
  - Box-plot con distribución


analyze_benchmark_results.py
────────────────────────────
Análisis estadístico detallado de resultados.

Entrada:
  - CSV generado por benchmark_hausdorff_algorithms.py
  - JSON con estadísticas (opcional)

Salida:
  - Violin plot (distribución)
  - Scatter plot (speedup por paciente)
  - Análisis de complejidad
  - Reporte de texto


compare_benchmarks.py
─────────────────────
Compara resultados de dos benchmarks diferentes.

Entrada:
  - Dos archivos JSON con estadísticas

Salida:
  - Tabla comparativa
  - Gráficos lado a lado
  - CSV con diferencias


generate_test_data.py
────────────────────
Genera volúmenes 3D sintéticos para pruebas.

Opciones:
  --num_patients: Número de pacientes a generar
  --slices_per_patient: Slices por volumen


run_benchmark_demo.py
────────────────────
Demostración automática de extremo a extremo:
  1. Genera datos sintéticos
  2. Ejecuta benchmark
  3. Muestra resultados


FLUJO RECOMENDADO
─────────────────

1. Primera ejecución (datos reales):
   [1] → Benchmark Completo
   [2] → Analizar Resultados

2. Comparar múltiples datasets:
   [1] → Benchmark dataset 1
   [1] → Benchmark dataset 2
   [3] → Comparar resultados

3. Validación rápida:
   [5] → Demostración


PARÁMETROS PRINCIPALES
──────────────────────

--pred:     Directorio con predicciones (REQUERIDO)
--gt:       Directorio con Ground Truth (REQUERIDO)
--output:   Directorio de salida (default: predicciones)
--csv:      Archivo CSV de entrada (para análisis)
--json:     Archivo JSON con estadísticas (para análisis)


FORMATO DE DATOS
────────────────

Predicciones: XXXXX.nii.gz
Ground Truth: XXXXX.nii.gz

Ambos deben estar en 3D (altura, ancho, profundidad)
y tener dimensiones compatibles.


INTERPRETACIÓN DE RESULTADOS
────────────────────────────

Speedup = tiempo_baseline / tiempo_algoritmo

Ejemplo:
  - Speedup 1.0x = equivalente a baseline
  - Speedup 50.0x = 50 veces más rápido
  - Speedup 0.5x = 50% más lento que baseline

Mayor speedup = Mejor rendimiento ✓
    """
    
    print(docs)
    input("\nPresiona Enter para volver al menú...")


def main():
    while True:
        print_banner()
        print_menu()
        
        choice = input("Selecciona opción (0-6): ").strip()
        
        if choice == '1':
            if menu_benchmark():
                print("\n✓ Benchmark completado")
            else:
                print("\n✗ Error en benchmark")
        
        elif choice == '2':
            if menu_analyze():
                print("\n✓ Análisis completado")
            else:
                print("\n✗ Error en análisis")
        
        elif choice == '3':
            if menu_compare():
                print("\n✓ Comparación completada")
            else:
                print("\n✗ Error en comparación")
        
        elif choice == '4':
            if menu_generate():
                print("\n✓ Datos generados")
            else:
                print("\n✗ Error generando datos")
        
        elif choice == '5':
            if menu_demo():
                print("\n✓ Demostración completada")
            else:
                print("\n✗ Error en demostración")
        
        elif choice == '6':
            menu_docs()
        
        elif choice == '0':
            print("\n¡Hasta luego!\n")
            sys.exit(0)
        
        else:
            print("\n❌ Opción inválida")
        
        input("\nPresiona Enter para continuar...")


if __name__ == '__main__':
    main()
