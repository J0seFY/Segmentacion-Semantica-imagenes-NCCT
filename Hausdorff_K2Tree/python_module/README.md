# Hausdorff K2Tree - Métricas de Hausdorff para Deep Learning

Este módulo proporciona implementaciones eficientes de algoritmos de distancia de Hausdorff usando estructuras de datos K2-Tree, optimizadas para su uso como métricas de evaluación en modelos de segmentación (como U-Net).

## Características

- **Algoritmos implementados:**
  - K2T-MAXHEAP: Algoritmo optimizado con heap para K2-Trees
  - K2T-MAXHEAPv2: Versión mejorada del algoritmo anterior
  - Kamata: Algoritmo de Kamata con parámetro λ configurable
  - TAHA: Algoritmo de Taha

- **Integración con PyTorch:**
  - Métricas listas para usar con tensores
  - Funciones de pérdida (loss functions) para entrenamiento
  - Soporte para batches
  - Compatible con GPU/CPU

## Instalación

### Requisitos previos

```bash
pip install numpy torch torchvision pybind11 cmake
```

### Compilar el módulo

1. **Compilar libcds primero:**
```bash
cd libcds
./configure
make
```

2. **Compilar el módulo Python:**
```bash
cd python_module
pip install -e .
```

O usando CMake directamente:
```bash
cd python_module
mkdir build
cd build
cmake ..
make -j4
```

### Instalación alternativa (solo Python)
Si ya tienes las librerías compiladas:
```bash
cd python_module
python setup.py build_ext --inplace
python setup.py install
```

## Uso Básico

### 1. Importar el módulo

```python
import hausdorff_k2tree as hk2t
import numpy as np
import torch
```

### 2. Calcular distancia directamente

```python
# Crear máscaras binarias de ejemplo (512x512)
mask1 = np.zeros((512, 512), dtype=np.uint8)
mask2 = np.zeros((512, 512), dtype=np.uint8)

mask1[100:200, 100:200] = 1  # Cuadrado
mask2[150:250, 150:250] = 1  # Cuadrado desplazado

# Calcular distancias con diferentes algoritmos
dist_k2t = hk2t.hausdorff_k2t_maxheap(mask1, mask2)
dist_kamata = hk2t.hausdorff_kamata(mask1, mask2, lambda_param=3)
dist_taha = hk2t.hausdorff_taha(mask1, mask2)

print(f"K2T-MAXHEAP: {dist_k2t:.4f}")
print(f"Kamata: {dist_kamata:.4f}")
print(f"TAHA: {dist_taha:.4f}")
```

### 3. Usar con PyTorch

```python
# Convertir a tensores de PyTorch
tensor1 = torch.from_numpy(mask1).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
tensor2 = torch.from_numpy(mask2).float().unsqueeze(0).unsqueeze(0)

# Crear métrica
metric = hk2t.HausdorffMetric('k2t_maxheap')
distance = metric(tensor1, tensor2)
```

## Uso en Modelos de Segmentación

### 1. Como métrica de evaluación

```python
import torch.nn as nn
import hausdorff_k2tree as hk2t

class UNet(nn.Module):
    # ... tu implementación de U-Net ...
    pass

# Crear modelo y datos
model = UNet()
images = torch.randn(4, 1, 512, 512)  # Batch de imágenes
ground_truth = torch.randint(0, 2, (4, 1, 512, 512)).float()  # Máscaras GT

# Predicción
with torch.no_grad():
    predictions = torch.sigmoid(model(images))
    binary_preds = (predictions > 0.5).float()

# Evaluar con diferentes algoritmos de Hausdorff
metrics = {
    'hausdorff_k2t': hk2t.HausdorffMetric('k2t_maxheap'),
    'hausdorff_kamata': hk2t.HausdorffMetric('kamata'),
    'hausdorff_taha': hk2t.HausdorffMetric('taha')
}

for name, metric in metrics.items():
    distance = metric(binary_preds, ground_truth)
    print(f"{name}: {distance:.4f}")
```

### 2. Como función de pérdida

```python
# Función de pérdida solo Hausdorff
hausdorff_loss = hk2t.HausdorffLoss(algorithm='k2t_maxheap')

# Función de pérdida combinada (recomendado)
combined_loss = hk2t.CombinedLoss(
    hausdorff_algorithm='k2t_maxheap',
    hausdorff_weight=0.1,    # Peso para Hausdorff
    bce_weight=1.0,          # Peso para BCE
    dice_weight=0.5          # Peso para Dice
)

# En el loop de entrenamiento
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for images, masks in dataloader:
    optimizer.zero_grad()
    
    outputs = model(images)
    loss = combined_loss(outputs, masks)
    
    loss.backward()
    optimizer.step()
```

### 3. Ejemplo completo de entrenamiento

```python
def train_model_with_hausdorff():
    model = UNet().cuda()
    
    # Configurar pérdida combinada
    criterion = hk2t.CombinedLoss(
        hausdorff_algorithm='k2t_maxheap',
        hausdorff_weight=0.1,
        bce_weight=1.0
    )
    
    # Métricas de evaluación
    eval_metrics = {
        'hausdorff_fast': hk2t.HausdorffMetric('k2t_maxheap'),
        'hausdorff_kamata': hk2t.HausdorffMetric('kamata'),
    }
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.cuda(), masks.cuda()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Evaluación
        model.eval()
        hausdorff_scores = {name: [] for name in eval_metrics.keys()}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.cuda(), masks.cuda()
                outputs = torch.sigmoid(model(images))
                preds = (outputs > 0.5).float()
                
                for name, metric in eval_metrics.items():
                    score = metric(preds, masks)
                    hausdorff_scores[name].append(score)
        
        # Reportar resultados
        for name, scores in hausdorff_scores.items():
            avg_score = np.mean(scores)
            print(f'Epoch {epoch} - {name}: {avg_score:.4f}')
```

## Parámetros de Configuración

### HausdorffMetric / HausdorffCalculator

- **algorithm**: `'k2t_maxheap'`, `'k2t_maxheap_v2'`, `'kamata'`, `'taha'`
- **max_level**: Nivel máximo del K2-Tree (default: 10)
- **lambda_param**: Parámetro λ para algoritmo Kamata (default: 3)

### HausdorffLoss / CombinedLoss

- **hausdorff_weight**: Peso para la pérdida de Hausdorff (default: 0.1)
- **bce_weight**: Peso para Binary Cross Entropy (default: 1.0)
- **dice_weight**: Peso para Dice Loss (default: 0.0)

## Consideraciones de Rendimiento

1. **K2T-MAXHEAP** es generalmente el más rápido
2. **Kamata** tiene mejor precisión pero es más lento
3. **TAHA** es una implementación de referencia
4. Para máscaras grandes (>512x512), considera usar `max_level=12` o mayor
5. Las funciones de pérdida tienen overhead computacional; úsalas con peso pequeño

## Formato de Entrada

- **Máscaras NumPy**: `np.array` de tipo `uint8` con valores 0 y 1
- **Tensores PyTorch**: `torch.Tensor` de tipo `float` con valores 0.0 y 1.0
- **Dimensiones soportadas**: 
  - 2D: `(H, W)`
  - 3D: `(1, H, W)` - un canal
  - 4D: `(B, 1, H, W)` - batch con un canal

## Troubleshooting

### Error de compilación
```bash
# Asegúrate de que libcds esté compilado
cd libcds && make clean && ./configure && make

# Reinstalar el módulo
cd python_module && pip uninstall hausdorff-k2tree && pip install -e .
```

### Errores de memoria
```python
# Reduce max_level para máscaras muy grandes
metric = hk2t.HausdorffMetric('k2t_maxheap', max_level=8)
```

### Rendimiento lento
```python
# Usa el algoritmo más rápido
metric = hk2t.HausdorffMetric('k2t_maxheap')

# O reduce el peso en la función de pérdida
loss = hk2t.CombinedLoss(hausdorff_weight=0.05)
```

## Ejemplos

Ver `examples/unet_example.py` para un ejemplo completo de entrenamiento con U-Net.

Ver `tests/test_hausdorff.py` para ejemplos de uso y tests de validación.

## Licencia

MIT License - ver archivo LICENSE para detalles.