"""
DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping

This package implements a hierarchical vision-language-action framework for dexterous grasping
that achieves high success rates in cluttered scenes with unseen objects.
"""

__version__ = "0.1.0"
__author__ = "siky"

# Import main modules for easy access
from . import controller
from . import planner
from . import inference_utils

__all__ = [
    "controller",
    "planner",
    "inference_utils",
]
