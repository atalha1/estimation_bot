"""
Training module initialization.
Location: training/__init__.py
"""

from .trainer import SelfPlayTrainer, GameResult, ModelStats, ModelInterface
from .models import HeuristicModelWrapper, LearnableModelWrapper, EnsembleModelWrapper

__all__ = [
    'SelfPlayTrainer',
    'GameResult', 
    'ModelStats',
    'ModelInterface',
    'HeuristicModelWrapper',
    'LearnableModelWrapper',
    'EnsembleModelWrapper'
]