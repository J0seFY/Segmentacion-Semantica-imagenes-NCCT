"""
Utilidades de PyTorch para usar los algoritmos de Hausdorff como métricas y funciones de pérdida
"""

import torch
import torch.nn as nn
import numpy as np
from . import hausdorff_k2tree_core

HausdorffCalculator = hausdorff_k2tree_core.HausdorffCalculator
hausdorff_k2t_maxheap = hausdorff_k2tree_core.hausdorff_k2t_maxheap
hausdorff_k2t_maxheap_v2 = hausdorff_k2tree_core.hausdorff_k2t_maxheap_v2
hausdorff_kamata = hausdorff_k2tree_core.hausdorff_kamata
hausdorff_taha = hausdorff_k2tree_core.hausdorff_taha

class HausdorffMetric:
    """
    Métrica de Hausdorff para evaluación de modelos de segmentación
    """
    
    def __init__(self, algorithm='k2t_maxheap', max_level=10, lambda_param=3):
        """
        Args:
            algorithm: Algoritmo a usar ('k2t_maxheap', 'k2t_maxheap_v2', 'kamata', 'taha')
            max_level: Nivel máximo para K2-Tree (default: 10)
            lambda_param: Parámetro lambda para algoritmo Kamata (default: 3)
        """
        self.algorithm = algorithm
        self.max_level = max_level
        self.lambda_param = lambda_param
        
        # Mapeo de algoritmos a funciones
        self.algorithm_map = {
            'k2t_maxheap': hausdorff_k2t_maxheap,
            'k2t_maxheap_v2': hausdorff_k2t_maxheap_v2,
            'kamata': hausdorff_kamata,
            'taha': hausdorff_taha
        }
        
        if algorithm not in self.algorithm_map:
            raise ValueError(f"Algoritmo {algorithm} no soportado. Opciones: {list(self.algorithm_map.keys())}")
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calcula la distancia de Hausdorff entre predicción y ground truth
        
        Args:
            pred: Tensor de predicción (B, 1, H, W) o (1, H, W) o (H, W)
            target: Tensor ground truth con las mismas dimensiones que pred
            
        Returns:
            float: Distancia de Hausdorff promedio si hay múltiples batches, 
                   o distancia única si es un solo ejemplo
        """
        return self.calculate(pred, target)
    
    def calculate(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calcula la distancia de Hausdorff
        """
        # Convertir a numpy y asegurar que sean máscaras binarias
        pred_np = self._prepare_mask(pred)
        target_np = self._prepare_mask(target)
        
        # Si es un batch, calcular para cada ejemplo
        if pred_np.ndim == 3:  # (B, H, W)
            distances = []
            for i in range(pred_np.shape[0]):
                dist = self._calculate_single(pred_np[i], target_np[i])
                distances.append(dist)
            return np.mean(distances)
        else:  # (H, W)
            return self._calculate_single(pred_np, target_np)
    
    def _prepare_mask(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Prepara la máscara: convierte a numpy, binariza y asegura tipo uint8
        """
        # Mover a CPU si está en GPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Convertir a numpy
        mask = tensor.detach().numpy()
        
        # Eliminar dimensión de canal si existe
        if mask.ndim == 4:  # (B, C, H, W)
            mask = mask.squeeze(1)  # (B, H, W)
        elif mask.ndim == 3 and mask.shape[0] == 1:  # (1, H, W)
            mask = mask.squeeze(0)  # (H, W)
        
        # Binarizar (threshold en 0.5 para probabilidades, 0 para logits)
        mask = (mask > 0.5).astype(np.uint8)
        
        return mask
    
    def _calculate_single(self, pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
        """
        Calcula la distancia de Hausdorff para un solo par de máscaras
        """
        # Verificar que las máscaras no estén vacías
        if np.sum(pred_mask) == 0 and np.sum(target_mask) == 0:
            return 0.0  # Ambas vacías = distancia 0
        
        if np.sum(pred_mask) == 0 or np.sum(target_mask) == 0:
            return float('inf')  # Una vacía, otra no = distancia infinita
        
        # Seleccionar algoritmo y calcular
        if self.algorithm == 'kamata':
            return self.algorithm_map[self.algorithm](
                pred_mask, target_mask, self.lambda_param, self.max_level
            )
        else:
            return self.algorithm_map[self.algorithm](
                pred_mask, target_mask, self.max_level
            )


class HausdorffLoss(nn.Module):
    """
    Función de pérdida basada en distancia de Hausdorff para entrenamiento
    """
    
    def __init__(self, algorithm='k2t_maxheap', max_level=10, lambda_param=3, 
                 weight=1.0, reduction='mean'):
        """
        Args:
            algorithm: Algoritmo de Hausdorff a usar
            max_level: Nivel máximo para K2-Tree
            lambda_param: Parámetro lambda para Kamata
            weight: Peso de la pérdida de Hausdorff
            reduction: 'mean', 'sum' o 'none'
        """
        super(HausdorffLoss, self).__init__()
        
        self.hausdorff_metric = HausdorffMetric(algorithm, max_level, lambda_param)
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicción del modelo (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
            
        Returns:
            torch.Tensor: Pérdida de Hausdorff
        """
        # Aplicar sigmoid si las predicciones no están normalizadas
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        batch_size = pred.shape[0]
        losses = []
        
        for i in range(batch_size):
            # Extraer máscaras individuales
            pred_i = pred[i].squeeze().cpu().detach().numpy()
            target_i = target[i].squeeze().cpu().detach().numpy()
            
            # Binarizar
            pred_i = (pred_i > 0.5).astype(np.uint8)
            target_i = (target_i > 0.5).astype(np.uint8)
            
            # Calcular distancia de Hausdorff
            hausdorff_dist = self.hausdorff_metric._calculate_single(pred_i, target_i)
            
            # Manejar infinitos (cuando una máscara está vacía)
            if np.isinf(hausdorff_dist):
                hausdorff_dist = 1000.0  # Penalty alto pero finito
            
            losses.append(hausdorff_dist)
        
        # Convertir a tensor
        losses = torch.tensor(losses, dtype=torch.float32, device=pred.device)
        
        # Aplicar peso
        losses = losses * self.weight
        
        # Aplicar reducción
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses


class CombinedLoss(nn.Module):
    """
    Combina BCE/Dice loss con Hausdorff loss para entrenamientos más estables
    """
    
    def __init__(self, hausdorff_algorithm='k2t_maxheap', hausdorff_weight=0.1, 
                 bce_weight=1.0, dice_weight=0.0):
        """
        Args:
            hausdorff_algorithm: Algoritmo de Hausdorff
            hausdorff_weight: Peso para la pérdida de Hausdorff
            bce_weight: Peso para Binary Cross Entropy
            dice_weight: Peso para Dice Loss
        """
        super(CombinedLoss, self).__init__()
        
        self.hausdorff_loss = HausdorffLoss(algorithm=hausdorff_algorithm, 
                                          weight=hausdorff_weight)
        self.bce_loss = nn.BCEWithLogitsLoss() if bce_weight > 0 else None
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcula Dice Loss
        """
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        return 1 - dice.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida combinada
        """
        total_loss = 0
        
        # BCE Loss
        if self.bce_loss is not None:
            bce = self.bce_loss(pred, target)
            total_loss += self.bce_weight * bce
        
        # Dice Loss
        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            total_loss += self.dice_weight * dice
        
        # Hausdorff Loss
        hausdorff = self.hausdorff_loss(pred, target)
        total_loss += hausdorff
        
        return total_loss


# Funciones de conveniencia
def calculate_hausdorff_batch(pred: torch.Tensor, target: torch.Tensor, 
                             algorithm='k2t_maxheap', **kwargs) -> list:
    """
    Calcula distancia de Hausdorff para un batch completo
    
    Returns:
        list: Lista de distancias para cada ejemplo del batch
    """
    metric = HausdorffMetric(algorithm=algorithm, **kwargs)
    
    pred_np = metric._prepare_mask(pred)
    target_np = metric._prepare_mask(target)
    
    if pred_np.ndim == 3:  # Batch
        return [metric._calculate_single(pred_np[i], target_np[i]) 
                for i in range(pred_np.shape[0])]
    else:  # Single example
        return [metric._calculate_single(pred_np, target_np)]