# Segmentación Semántica en Imágenes Biomédicas de ACV Isquémico con Arquitecturas U-Net y Evaluación mediante Distancia de Hausdorff

## Resumen del Proyecto

Este repositorio contiene el código y resultados del proyecto de título para obtener el grado de **Ingeniero Civil en Informática** en la Universidad del Bío-Bío.

El proyecto implementa y evalúa arquitecturas de redes neuronales profundas (U-Net, Attention U-Net, HybridUNet) para la segmentación semántica de lesiones por accidente cerebrovascular (ACV) isquémico en imágenes CT de tomografía computada. La evaluación se realiza utilizando métricas estándar (Dice, F1, Precision, Recall) y la distancia de Hausdorff para medir similitud entre geometrías de segmentación.

---

## Autores y Tutores

**Autores:**
- Sebastián Jouannet Contreras
- José Fuentes Yáñez

**Profesor Guía:**
- Dra. Carola Andrea Figueroa Flores

**Profesor Co-Guía:**
- Dr. Miguel Esteban Romero Vásquez

---

## Estructura del Proyecto

```
Segmentacion-Semantica-imagenes-NCCT/
├── Hausdorff_K2Tree/                    # Módulo Hausdorff (Proyecto Fondecyt 11251944)
│   ├── CMakeLists.txt
│   ├── src/                             # Código C++ compilado
│   ├── include/                         # Headers
│   ├── libcds/                          # Biblioteca CDS externa
│   ├── Snapshot/                        # Utilidades de snapshots
│   ├── Logs/                            # Manejadores de logs
│   ├── Util/                            # Utilidades genéricas
│   ├── IO/                              # Adaptadores de entrada/salida
│   └── python_module/                   # BINDING DE PYTHON (Autoría del Proyecto)
│       ├── setup.py
│       ├── CMakeLists.txt
│       ├── hausdorff_k2tree/            # Módulo Python
│       ├── src/                         # Código fuente C++ del binding
│       ├── tests/                       # Tests del módulo
│       └── examples/                    # Ejemplos de uso
│
├── scripts/                             # Scripts de procesamiento y entrenamiento
│   ├── generacion_volumenes_nifti.py    # Conversión DICOM -> NIfTI y alineación de máscaras
│   ├── preparar_nnunet.py               # Orquestación de pipeline nnU-Net v2
│   ├── train.py                         # Entrenamiento de modelos personalizados
│   ├── evaluate.py                      # Evaluación de modelos
│   ├── evaluate_nnunet.py               # Evaluación específica para nnU-Net
│   ├── run_experiments.sh               # Script bash para ejecutar experimentos
│   ├── run_ablations.sh                 # Script bash para estudios de ablación
│   ├── run_hybrid_exp3.sh               # Script bash para experimentos HybridUNet
│   └── scripts_hausdorff/               # Scripts de distancia Hausdorff
│       ├── benchmark_hausdorff_algorithms.py
│       ├── analyze_benchmark_results.py
│       ├── validate_hausdorff_accuracy.py
│       ├── scalability_predictions.py
│       └── main_menu.py
│
├── src/                                 # Código fuente Python
│   ├── __init__.py
│   ├── models.py                        # Definiciones de arquitecturas (UNet, AttentionUNet, HybridUNet)
│   ├── dataset.py                       # DataLoaders y pre-procesamiento
│   └── __pycache__/
│
├── Hausdorff_K2Tree/                    # (Referencia: Proyecto Fondecyt de Iniciación 2025 N° 11251944)
│
└── README.md                            # Este archivo
```

---

## Descripción de Módulos y Carpetas

### 1. **Hausdorff_K2Tree**
#### Autoría y Proveniencia del Código

El directorio **`Hausdorff_K2Tree/`** corresponde al repositorio original del proyecto Fondecyt de Iniciación en Investigación 2025 N° 11251944, disponible públicamente en:

https://github.com/fsantolaya/Hausdorff_k2tree

El código base en C++ (algoritmos, estructuras de datos y utilidades) **NO es de autoría de los autores de este proyecto de título** y se incluye únicamente con fines de integración y evaluación experimental.

#### Aporte realizado en este proyecto (de autoría propia)
- **`python_module/`**: Módulo de integración C++/Python desarrollado en el contexto de este proyecto.
  - Binding de C++ a Python usando **PyBind11**.
  - Exposición de funciones para el cálculo de distancia de Hausdorff mediante distintos algoritmos.
  - Compatibilidad con tensores de **PyTorch** para su uso en pipelines de deep learning.

  - El archivo **`python_module/src/python_binding.cpp`** agrupa las funciones principales del proyecto relacionadas con el cálculo de distancias de Hausdorff utilizando diferentes algoritmos.
    - Las funciones desarrolladas específicamente en este proyecto se encuentran **explícitamente indicadas mediante comentarios en el código**.
    - El resto de las funciones corresponden a implementaciones de **autoría de los integrantes del proyecto Fondecyt**.

  - La carpeta **`hausdorff_k2tree_python/`** (dentro de `python_module/`) corresponde íntegramente a código de **autoría de los autores de este proyecto**, e incluye los wrappers y utilidades Python para facilitar la integración y el uso del módulo desde entornos de aprendizaje profundo.

 
#### Nota sobre la integración del repositorio

El repositorio original **Hausdorff_K2Tree** no se incluyó como submódulo de Git debido a que el módulo `python_module/` requiere una integración directa con la estructura de directorios del proyecto para:

- Mantener rutas relativas consistentes durante la compilación con CMake y PyBind11.
- Evitar modificaciones extensivas en los scripts de entrenamiento y evaluación existentes.

Por esta razón, el código fue incorporado directamente manteniendo su estructura original, respetando su autoría y referencia explícita al repositorio fuente.


### 2. **scripts/**
Scripts principales para procesamiento, entrenamiento y evaluación:

- **`generacion_volumenes_nifti.py`**: Preprocesamiento de datos
  - Convierte series DICOM a volúmenes NIfTI
  - Alinea máscaras PNG 2D con imágenes 3D
  - Genera estructura `Dataset<ID>_<NAME>` compatible con nnU-Net
  - Crea archivo `dataset.json` con metadatos

- **`preparar_nnunet.py`**: Orquestación completa del pipeline nnU-Net v2
  - Etapa 0: Preprocesamiento (delegado a `generacion_volumenes_nifti.py`)
  - Etapa 1: Plan and Preprocess (nnU-Net)
  - Etapa 2: Training
  - Etapa 3: Prediction/Inference
  - Etapa 4: Evaluation con métricas custom y MONAI

- **`train.py`**: Entrenamiento de modelos personalizados (UNet, AttentionUNet, HybridUNet)
  - Soporte para múltiples configuraciones
  - Logging de métricas
  - Guardado de checkpoints

- **`evaluate.py`**: Evaluación de predicciones
  - Cálculo de Dice, F1, Precision, Recall
  - Métricas de distancia Hausdorff
  - Visualización de resultados

- **`evaluate_nnunet.py`**: Evaluación específica para modelos nnU-Net

- **`run_experiments.sh`**: Automatización de experimentos
  - Ejecución de múltiples configuraciones
  - Análisis de resultados

- **`run_ablations.sh`**: Estudios de ablación
  - Desactivación selectiva de componentes (ASPP, etc.)
  - Análisis de impacto de cada módulo

- **`scripts_hausdorff/`**: Herramientas para validación y benchmarking
  - `benchmark_hausdorff_algorithms.py`: Comparación de algoritmos
  - `validate_hausdorff_accuracy.py`: Validación de precisión
  - `scalability_predictions.py`: Análisis de escalabilidad
  - `main_menu.py`: Interfaz interactiva

### 3. **src/**
Código fuente Python central:

- **`models.py`**: Definiciones de arquitecturas
  - **UNet**: Baseline clásico
  - **AttentionUNet**: U-Net con módulos de atención
  - **HybridUNet**: SOTA (UNet3+ + ASPP)
    - UNet3+ para fusión multi-escala
    - ASPP (Atrous Spatial Pyramid Pooling) para contexto multi-resolución
  - Factory function para instanciación

- **`dataset.py`**: DataLoaders y utilidades
  - Cargadores de datos para NIfTI
  - Augmentación de datos
  - Normalización
  - Splits train/validation/test

---

## Especificaciones del Servidor

Los experimentos fueron ejecutados en un servidor de alto rendimiento con las siguientes especificaciones:

| Componente | Especificación |
|-----------|----------------|
| **CPU** | Intel Core i9-14900KF (24 núcleos, 32 hilos) |
| **RAM** | 125 GB |
| **GPU** | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| **CUDA** | 12.2 |
| **cuDNN** | 8.6+ |
| **OS** | Ubuntu 22.04 LTS |
| **Python** | 3.10+ |
| **PyTorch** | 2.0+ |

---

## Requisitos e Instalación

### Dependencias Principales

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
pip install nibabel numpy opencv-python pydicom scipy scikit-learn tensorboard
pip install monai  # Opcional, para métricas avanzadas
pip install nnunet  # Opcional, para pipeline nnU-Net v2
```

### Compilación del Módulo Hausdorff (Python Binding)

```bash
cd Hausdorff_K2Tree/python_module
python setup.py build_ext --inplace
# O usando CMake:
mkdir build && cd build
cmake ..
make
cd .. && pip install -e .
```

Verificar instalación:
```python
import hausdorff_k2tree
print("Módulo Hausdorff cargado correctamente")
```

---

## Uso del Código

### 1. Preprocesamiento de Datos (DICOM -> NIfTI)

```bash
cd scripts
python generacion_volumenes_nifti.py \
  --dataset-id 001 \
  --dataset-name AISD \
  --dicom-root /ruta/a/Dataset_dicom
```

**Entrada esperada:**
```
Dataset_dicom/
├── image/
│   ├── <patient_id>/
│   │   └── CT/
│   │       └── *.dcm
└── mask/
    └── <patient_id>/
        └── *.png
```

**Salida generada:**
```
nnUNet_raw/Dataset001_AISD/
├── imagesTr/      # Imágenes de entrenamiento (NIfTI)
├── labelsTr/      # Máscaras de entrenamiento (NIfTI alineadas)
├── imagesTs/      # Imágenes de prueba
├── labelsTs/      # Máscaras de prueba
└── dataset.json   # Metadatos del dataset
```

### 2. Entrenamiento con nnU-Net v2

```bash
cd scripts
python preparar_nnunet.py \
  --start-stage 0 \
  --end-stage 4 \
  --configurations 3d_fullres ensemble
```

**Etapas disponibles:**
- `0`: Preprocesamiento
- `1`: Plan and Preprocess (nnU-Net)
- `2`: Training
- `3`: Prediction
- `4`: Evaluation

### 3. Entrenamiento de Modelos Personalizados

```bash
cd scripts
python train.py \
  --model hybrid_unet \
  --batch-size 16 \
  --epochs 100 \
  --learning-rate 1e-3 \
  --dropout 0.3 \
  --use-aspp \
  --data-dir /ruta/a/Dataset001_AISD
```

### 4. Evaluación

```bash
python evaluate.py \
  --model hybrid_unet \
  --checkpoint /ruta/a/best_model.pth \
  --predictions /ruta/a/predictions \
  --labels /ruta/a/labelsTs
```

### 5. Cálculo de Distancia Hausdorff

```bash
cd scripts/scripts_hausdorff
python main_menu.py
# O directamente:
python benchmark_hausdorff_algorithms.py \
  --pred-dir /ruta/a/predictions \
  --label-dir /ruta/a/labelsTs
```

### 6. Estudios de Ablación

```bash
cd scripts
bash run_ablations.sh
```

---

## Dataset

El dataset utilizado fue obtenido de:

**Fuente:** [AISD - Ischemic Stroke Dataset](https://github.com/GriffinLiang/AISD)

**Características:**
- Imágenes CT de tomografía computada
- Máscaras de lesión por ACV isquémico
- Múltiples pacientes con anotaciones experto

**Estructura esperada:**
```
Dataset_dicom/
├── image/<patient_id>/CT/           # Series DICOM
└── mask/<patient_id>/               # Máscaras PNG alineadas
```

---

## Modelos Disponibles

### 1. UNet (Baseline)
- Arquitectura clásica de encoder-decoder
- 5 niveles de profundidad
- Skip connections directas

### 2. Attention U-Net
- UNet mejorado con módulos de atención
- Atiende características relevantes en cada escala
- Mejor rendimiento en datasets pequeños

### 3. HybridUNet (SOTA)
- Combina **UNet3+** y **ASPP**
- **UNet3+**: Decodificador con conexiones densas entre todos los niveles
- **ASPP**: Módulo multi-escala para capturar contexto a diferentes resoluciones
- Mejor generalización y precisión

---

## Resultados Principales

El proyecto evalúa arquitecturas usando:

- **Métricas pixelwise**: Dice, F1-Score, Precision, Recall
- **Métricas geométricas**: Distancia Hausdorff
- **Visualización**: Mapas de segmentación, heatmaps de confianza

---

## Estructura de Experimentos

### Entrenamiento

```bash
bash run_experiments.sh
```

Ejecuta automáticamente:
1. Entrenamiento 
2. Validación en set de prueba
3. Cálculo de métricas estándar y Hausdorff
4. Generación de reportes

### Ablación

```bash
bash run_ablations.sh
```

Estudia impacto de:
- ASPP vs. sin ASPP en UNet 3+

---

## Referencias Clave

1. **UNet**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
2. **Attention U-Net**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018
3. **UNet3+**: Huang et al., "UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation", ICASSP 2020
4. **ASPP**: Chen et al., "DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation", ECCV 2018
5. **nnU-Net**: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods 2021


## Agradecimientos

- Proyecto Fondecyt de Iniciación en Investigación 2025 N° 11251944: "Algoritmos y Estructuras de Datos Compactas para Calcular Eficientemente la Similitud de Conjuntos de Puntos usando la Distancia de Hausdorff"
- Dataset AISD por GriffinLiang et al.
- Comunidades de PyTorch, nnU-Net, MONAI por herramientas fundamentales
