"""
Tests para el módulo hausdorff_k2tree
"""

import unittest
import numpy as np
import torch
import sys
import os

# Agregar el path del módulo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import hausdorff_k2tree_python as hk2t
    MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Módulo no disponible: {e}")
    MODULE_AVAILABLE = False

class TestHausdorffK2Tree(unittest.TestCase):
    
    def setUp(self):
        """Preparar datos de prueba"""
        if not MODULE_AVAILABLE:
            self.skipTest("Módulo hausdorff_k2tree no disponible")
        
        # Crear máscaras de prueba simples
        self.mask1 = np.zeros((100, 100), dtype=np.uint8)
        self.mask2 = np.zeros((100, 100), dtype=np.uint8)
        
        # Máscara 1: cuadrado 20x20 en posición (10,10)
        self.mask1[10:30, 10:30] = 1
        
        # Máscara 2: cuadrado 20x20 en posición (15,15) - solapado
        self.mask2[15:35, 15:35] = 1
        
        # Máscaras idénticas
        self.mask_identical = np.copy(self.mask1)
        
        # Máscara vacía
        self.mask_empty = np.zeros((100, 100), dtype=np.uint8)
    
    def test_direct_functions(self):
        """Probar funciones directas"""
        
        # Test con máscaras solapadas
        dist1 = hk2t.hausdorff_k2t_maxheap(self.mask1, self.mask2)
        dist2 = hk2t.hausdorff_kamata(self.mask1, self.mask2)
        
        # Las distancias deben ser números positivos
        self.assertIsInstance(dist1, float)
        self.assertIsInstance(dist2, float)
        self.assertGreaterEqual(dist1, 0)
        self.assertGreaterEqual(dist2, 0)
        
        print(f"Distancia K2T-MAXHEAP: {dist1}")
        print(f"Distancia Kamata: {dist2}")
    
    def test_identical_masks(self):
        """Probar con máscaras idénticas"""
        
        dist = hk2t.hausdorff_k2t_maxheap(self.mask1, self.mask_identical)
        
        # La distancia entre máscaras idénticas debe ser 0
        self.assertAlmostEqual(dist, 0.0, places=5)
    
    def test_empty_masks(self):
        """Probar con máscaras vacías"""
        
        # Ambas vacías
        dist1 = hk2t.hausdorff_k2t_maxheap(self.mask_empty, self.mask_empty)
        self.assertAlmostEqual(dist1, 0.0, places=5)
        
        # Una vacía, otra no - esto podría dar infinito o error
        try:
            dist2 = hk2t.hausdorff_k2t_maxheap(self.mask1, self.mask_empty)
            # Si no da error, debería ser un número muy grande
            self.assertTrue(dist2 >= 0)
        except Exception:
            # Es aceptable que de error con máscaras vacías
            pass
    
    def test_hausdorff_calculator(self):
        """Probar la clase HausdorffCalculator"""
        
        calculator = hk2t.HausdorffCalculator(max_level=8)
        
        dist1 = calculator.calculate_k2t_maxheap(self.mask1, self.mask2)
        dist2 = calculator.calculate_kamata(self.mask1, self.mask2)
        
        self.assertIsInstance(dist1, float)
        self.assertIsInstance(dist2, float)
        self.assertGreaterEqual(dist1, 0)
        self.assertGreaterEqual(dist2, 0)
    
    def test_pytorch_metric(self):
        """Probar HausdorffMetric con tensores de PyTorch"""
        
        # Convertir máscaras a tensores de PyTorch
        tensor1 = torch.from_numpy(self.mask1).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        tensor2 = torch.from_numpy(self.mask2).float().unsqueeze(0).unsqueeze(0)
        
        metric = hk2t.HausdorffMetric('k2t_maxheap')
        
        dist = metric(tensor1, tensor2)
        
        self.assertIsInstance(dist, float)
        self.assertGreaterEqual(dist, 0)
        
        print(f"Distancia PyTorch Metric: {dist}")
    
    def test_pytorch_loss(self):
        """Probar HausdorffLoss"""
        
        # Crear tensores de ejemplo
        pred = torch.randn(2, 1, 50, 50)  # Logits
        target = torch.randint(0, 2, (2, 1, 50, 50)).float()
        
        loss_fn = hk2t.HausdorffLoss(algorithm='k2t_maxheap')
        
        loss = loss_fn(pred, target)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreaterEqual(loss.item(), 0)
        
        print(f"Hausdorff Loss: {loss.item()}")
    
    def test_combined_loss(self):
        """Probar CombinedLoss"""
        
        pred = torch.randn(2, 1, 50, 50, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 50, 50)).float()
        
        loss_fn = hk2t.CombinedLoss(
            hausdorff_weight=0.1,
            bce_weight=1.0,
            dice_weight=0.5
        )
        
        loss = loss_fn(pred, target)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.requires_grad)
        
        # Test backward pass
        loss.backward()
        self.assertIsNotNone(pred.grad)
        
        print(f"Combined Loss: {loss.item()}")
    
    def test_batch_calculation(self):
        """Probar cálculo en lotes"""
        
        # Crear batch de máscaras
        batch_size = 3
        masks1 = np.stack([self.mask1, self.mask2, self.mask_identical])
        masks2 = np.stack([self.mask2, self.mask1, self.mask1])
        
        distances = hk2t.calculate_hausdorff_batch(
            torch.from_numpy(masks1).float(),
            torch.from_numpy(masks2).float(),
            algorithm='k2t_maxheap'
        )
        
        self.assertEqual(len(distances), batch_size)
        
        for i, dist in enumerate(distances):
            self.assertIsInstance(dist, float)
            self.assertGreaterEqual(dist, 0)
            print(f"Distancia batch {i}: {dist}")
    
    def test_different_algorithms(self):
        """Probar diferentes algoritmos"""
        
        algorithms = ['k2t_maxheap', 'k2t_maxheap_v2', 'kamata', 'taha']
        
        results = {}
        
        for algo in algorithms:
            try:
                if algo == 'kamata':
                    dist = hk2t.hausdorff_kamata(self.mask1, self.mask2, lambda_param=3)
                else:
                    metric = hk2t.HausdorffMetric(algo)
                    dist = metric.calculate(
                        torch.from_numpy(self.mask1).float(),
                        torch.from_numpy(self.mask2).float()
                    )
                
                results[algo] = dist
                self.assertIsInstance(dist, float)
                self.assertGreaterEqual(dist, 0)
                
            except Exception as e:
                print(f"Error en algoritmo {algo}: {e}")
        
        print("Resultados por algoritmo:")
        for algo, dist in results.items():
            print(f"  {algo}: {dist:.4f}")
        
        # Al menos un algoritmo debe funcionar
        self.assertGreater(len(results), 0)

def run_performance_test():
    """Test de rendimiento básico"""
    
    if not MODULE_AVAILABLE:
        print("Módulo no disponible para test de rendimiento")
        return
    
    import time
    
    print("\nTest de rendimiento:")
    
    # Crear máscaras más grandes
    mask1 = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    mask2 = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    
    algorithms = ['k2t_maxheap', 'kamata']
    
    for algo in algorithms:
        try:
            start_time = time.time()
            
            if algo == 'kamata':
                dist = hk2t.hausdorff_kamata(mask1, mask2)
            else:
                dist = hk2t.hausdorff_k2t_maxheap(mask1, mask2)
            
            elapsed = time.time() - start_time
            
            print(f"{algo}: {dist:.4f} (tiempo: {elapsed:.4f}s)")
            
        except Exception as e:
            print(f"Error en {algo}: {e}")

if __name__ == '__main__':
    # Ejecutar tests
    unittest.main(verbosity=2, exit=False)
    
    # Ejecutar test de rendimiento
    run_performance_test()