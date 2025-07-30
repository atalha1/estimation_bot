"""
Model wrapper to integrate existing bots with training system.
Location: training/models.py
"""

from pathlib import Path
import pickle
import uuid
from typing import Dict, Any, Optional
import copy

from estimation_bot.player import BotInterface
from bot.heuristic_bot import HeuristicBot, AdvancedHeuristicBot
from bot.random_bot import RandomBot, WeightedRandomBot
from training.trainer import ModelInterface


class HeuristicModelWrapper(ModelInterface):
    """Wraps heuristic bots for training system."""
    
    def __init__(self, bot_class=AdvancedHeuristicBot, model_id: str = None):
        self.bot_class = bot_class
        self.model_id = model_id or f"heuristic_{uuid.uuid4().hex[:8]}"
        self.config = {}  # Future: store hyperparameters
        self.generation = 0
        
    def get_id(self) -> str:
        return self.model_id
    
    def save(self, path: Path):
        """Save model state."""
        state = {
            'model_id': self.model_id,
            'bot_class_name': self.bot_class.__name__,
            'config': self.config,
            'generation': self.generation
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: Path):
        """Load model state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model_id = state['model_id']
        self.config = state['config']
        self.generation = state['generation']
        # Note: bot_class should be set based on class name
    
    def clone(self) -> 'HeuristicModelWrapper':
        """Create a copy with new ID."""
        clone = HeuristicModelWrapper(self.bot_class)
        clone.config = copy.deepcopy(self.config)
        clone.generation = self.generation + 1
        return clone
    
    def create_bot(self, name: str) -> BotInterface:
        """Create bot instance."""
        return self.bot_class(name)


class LearnableModelWrapper(ModelInterface):
    """
    Wrapper for future neural network models.
    Placeholder for RL/supervised learning integration.
    """
    
    def __init__(self, model_id: str = None):
        self.model_id = model_id or f"neural_{uuid.uuid4().hex[:8]}"
        self.weights = {}  # Future: neural network weights
        self.config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'hidden_size': 256,
            'num_layers': 3
        }
        self.generation = 0
        self.training_steps = 0
        
    def get_id(self) -> str:
        return self.model_id
    
    def save(self, path: Path):
        """Save model weights and config."""
        state = {
            'model_id': self.model_id,
            'weights': self.weights,
            'config': self.config,
            'generation': self.generation,
            'training_steps': self.training_steps
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: Path):
        """Load model state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.__dict__.update(state)
    
    def clone(self) -> 'LearnableModelWrapper':
        """Create copy with optional weight perturbation."""
        clone = LearnableModelWrapper()
        clone.weights = copy.deepcopy(self.weights)
        clone.config = copy.deepcopy(self.config)
        clone.generation = self.generation + 1
        
        # Future: Add small random perturbations for diversity
        # self._add_noise(clone.weights, noise_scale=0.01)
        
        return clone
    
    def create_bot(self, name: str) -> BotInterface:
        """Create bot that uses neural network for decisions."""
        # For now, fallback to heuristic bot
        # Future: Return NeuralBot(self.weights, name)
        return AdvancedHeuristicBot(name)
    
    def update_weights(self, gradients: Dict[str, Any]):
        """Update model weights from training."""
        # Future: Implement weight updates
        self.training_steps += 1


class EnsembleModelWrapper(ModelInterface):
    """
    Ensemble of multiple models for robust decision making.
    Can combine heuristic and learned models.
    """
    
    def __init__(self, models: list[ModelInterface], model_id: str = None):
        self.models = models
        self.model_id = model_id or f"ensemble_{uuid.uuid4().hex[:8]}"
        self.weights = [1.0 / len(models)] * len(models)  # Equal weights initially
        
    def get_id(self) -> str:
        return self.model_id
    
    def save(self, path: Path):
        """Save ensemble state."""
        ensemble_dir = Path(path).parent / f"{self.model_id}_ensemble"
        ensemble_dir.mkdir(exist_ok=True)
        
        # Save each model
        for i, model in enumerate(self.models):
            model.save(ensemble_dir / f"model_{i}.pkl")
        
        # Save ensemble metadata
        state = {
            'model_id': self.model_id,
            'weights': self.weights,
            'num_models': len(self.models)
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: Path):
        """Load ensemble state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.model_id = state['model_id']
        self.weights = state['weights']
        
        # Load individual models
        ensemble_dir = Path(path).parent / f"{self.model_id}_ensemble"
        self.models = []
        for i in range(state['num_models']):
            # Need model class info to properly instantiate
            # For now, assume HeuristicModelWrapper
            model = HeuristicModelWrapper()
            model.load(ensemble_dir / f"model_{i}.pkl")
            self.models.append(model)
    
    def clone(self) -> 'EnsembleModelWrapper':
        """Clone ensemble with all sub-models."""
        cloned_models = [m.clone() for m in self.models]
        clone = EnsembleModelWrapper(cloned_models)
        clone.weights = copy.deepcopy(self.weights)
        return clone
    
    def create_bot(self, name: str) -> BotInterface:
        """Create ensemble bot that votes on decisions."""
        # For now, use first model
        # Future: Return EnsembleBot(self.models, self.weights, name)
        return self.models[0].create_bot(name)