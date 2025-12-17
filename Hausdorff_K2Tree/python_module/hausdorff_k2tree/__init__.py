"""
Hausdorff K2Tree: Efficient Hausdorff distance calculations using K2-Tree data structures
"""

from . import hausdorff_k2tree_core

HausdorffCalculator = hausdorff_k2tree_core.HausdorffCalculator
hausdorff_k2t_maxheap = hausdorff_k2tree_core.hausdorff_k2t_maxheap
hausdorff_k2t_maxheap_v2 = hausdorff_k2tree_core.hausdorff_k2t_maxheap_v2
hausdorff_kamata = hausdorff_k2tree_core.hausdorff_kamata
hausdorff_taha = hausdorff_k2tree_core.hausdorff_taha

from .pytorch_utils import HausdorffLoss, HausdorffMetric

__version__ = "1.0.0"
__all__ = [
    "HausdorffCalculator",
    "hausdorff_k2t_maxheap", 
    "hausdorff_k2t_maxheap_v2",
    "hausdorff_kamata",
    "hausdorff_taha",
    "HausdorffLoss",
    "HausdorffMetric"
]