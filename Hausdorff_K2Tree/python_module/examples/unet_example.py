"""
Ejemplo de uso de las métricas de Hausdorff con un modelo U-Net
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Importar nuestro módulo
import hausdorff_k2tree_python as hk2t

class SimpleUNet(nn.Module):
    """U-Net simplificado para el ejemplo"""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Decoder
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Simplificado - implementar skip connections completas en modelo real
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d3 = self.dec3(e3)
        d2 = self.dec2(self.upsample(d3))
        d1 = self.dec1(self.upsample(d2))
        
        return d1

def train_with_hausdorff_metrics():
    """
    Ejemplo de entrenamiento usando métricas de Hausdorff
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Crear modelo
    model = SimpleUNet().to(device)
    
    # 2. Configurar función de pérdida combinada
    # Usaremos BCE + Hausdorff para entrenamiento más estable
    criterion = hk2t.CombinedLoss(
        hausdorff_algorithm='k2t_maxheap',  # Algoritmo más rápido
        hausdorff_weight=0.1,              # Peso menor para Hausdorff
        bce_weight=1.0,                    # Peso principal para BCE
        dice_weight=0.5                    # También incluir Dice
    )
    
    # 3. Configurar métricas de evaluación
    hausdorff_metrics = {
        'hausdorff_k2t_maxheap': hk2t.HausdorffMetric('k2t_maxheap'),
        'hausdorff_k2t_maxheap_v2': hk2t.HausdorffMetric('k2t_maxheap_v2'),
        'hausdorff_kamata': hk2t.HausdorffMetric('kamata', lambda_param=3),
        'hausdorff_taha': hk2t.HausdorffMetric('taha')
    }
    
    # 4. Optimizador
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 5. Datos de ejemplo (reemplaza con tu DataLoader real)
    def create_sample_data():
        """Crea datos de ejemplo para demostración"""
        batch_size = 4
        # Imágenes de entrada
        images = torch.randn(batch_size, 1, 512, 512).to(device)
        # Máscaras ground truth binarias
        masks = torch.randint(0, 2, (batch_size, 1, 512, 512)).float().to(device)
        return images, masks
    
    # 6. Loop de entrenamiento de ejemplo
    model.train()
    
    for epoch in range(5):  # Solo 5 epochs para el ejemplo
        print(f"\nEpoch {epoch + 1}/5")
        
        # Crear batch de ejemplo (reemplaza con tu DataLoader)
        images, masks = create_sample_data()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calcular pérdida
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
        
        # 7. Evaluación con métricas de Hausdorff
        if epoch % 2 == 0:  # Evaluar cada 2 epochs
            model.eval()
            with torch.no_grad():
                # Obtener predicciones
                pred_masks = torch.sigmoid(outputs) > 0.5
                
                # Calcular todas las métricas de Hausdorff
                print("Métricas de Hausdorff:")
                for metric_name, metric_fn in hausdorff_metrics.items():
                    try:
                        hausdorff_dist = metric_fn(pred_masks.float(), masks)
                        print(f"  {metric_name}: {hausdorff_dist:.4f}")
                    except Exception as e:
                        print(f"  {metric_name}: Error - {e}")
            
            model.train()

def evaluate_model_hausdorff(model, test_loader, device):
    """
    Evalúa un modelo entrenado usando métricas de Hausdorff
    """
    
    # Configurar métricas
    metrics = {
        'k2t_maxheap': hk2t.HausdorffMetric('k2t_maxheap'),
        'k2t_maxheap_v2': hk2t.HausdorffMetric('k2t_maxheap_v2'), 
        'kamata': hk2t.HausdorffMetric('kamata'),
        'taha': hk2t.HausdorffMetric('taha')
    }
    
    model.eval()
    results = {name: [] for name in metrics.keys()}
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Predicción
            outputs = model(images)
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            
            # Calcular métricas para cada algoritmo
            for metric_name, metric_fn in metrics.items():
                try:
                    hausdorff_dist = metric_fn(pred_masks, masks)
                    results[metric_name].append(hausdorff_dist)
                except Exception as e:
                    print(f"Error en {metric_name}: {e}")
            
            if batch_idx % 10 == 0:
                print(f"Evaluado batch {batch_idx}")
    
    # Calcular estadísticas
    print("\nResultados de evaluación:")
    for metric_name, distances in results.items():
        if distances:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            print(f"{metric_name}: {mean_dist:.4f} ± {std_dist:.4f}")
        else:
            print(f"{metric_name}: No se pudo calcular")
    
    return results

def simple_usage_example():
    """
    Ejemplo simple de uso directo de las funciones
    """
    
    print("Ejemplo de uso directo:")
    
    # Crear máscaras de ejemplo (512x512)
    mask1 = np.zeros((512, 512), dtype=np.uint8)
    mask2 = np.zeros((512, 512), dtype=np.uint8)
    
    # Crear algunos píxeles activos
    mask1[100:200, 100:200] = 1  # Cuadrado
    mask2[150:250, 150:250] = 1  # Cuadrado desplazado
    
    # Calcular distancias con diferentes algoritmos
    print("Calculando distancias de Hausdorff...")
    
    # Método 1: Funciones directas
    dist1 = hk2t.hausdorff_k2t_maxheap(mask1, mask2)
    dist2 = hk2t.hausdorff_k2t_maxheap_v2(mask1, mask2) 
    dist3 = hk2t.hausdorff_kamata(mask1, mask2, lambda_param=3)
    dist4 = hk2t.hausdorff_taha(mask1, mask2)
    
    print(f"K2T-MAXHEAP: {dist1:.4f}")
    print(f"K2T-MAXHEAPv2: {dist2:.4f}")
    print(f"Kamata: {dist3:.4f}")
    print(f"TAHA: {dist4:.4f}")
    
    # Método 2: Usando la clase HausdorffCalculator
    calculator = hk2t.HausdorffCalculator(max_level=10)
    
    dist_calc1 = calculator.calculate_k2t_maxheap(mask1, mask2)
    dist_calc2 = calculator.calculate_kamata(mask1, mask2, lambda_param=3)
    
    print(f"\nUsando HausdorffCalculator:")
    print(f"K2T-MAXHEAP: {dist_calc1:.4f}")
    print(f"Kamata: {dist_calc2:.4f}")

if __name__ == "__main__":
    print("Ejemplo de uso de Hausdorff K2Tree con U-Net")
    
    # Ejemplo simple
    simple_usage_example()
    
    # Ejemplo de entrenamiento
    print("\n" + "="*50)
    print("Iniciando ejemplo de entrenamiento...")
    train_with_hausdorff_metrics()