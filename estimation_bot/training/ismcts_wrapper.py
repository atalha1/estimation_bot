#!/usr/bin/env python3
"""
ISMCTS Training Wrapper - Integration with existing training system
Location: estimation_bot/training/ismcts_wrapper.py

Wraps ISMCTS agent for training compatibility and self-play tuning.
"""

import pickle
import uuid
import copy
import random
from pathlib import Path
from typing import Dict, Any, Optional

from .trainer import ModelInterface
from .models import HeuristicModelWrapper
from ..player import BotInterface
from ..bot.ismcts_agent import ISMCTSAgent


class ISMCTSModelWrapper(ModelInterface):
    """Wrapper for ISMCTS agent in training system."""
    
    def __init__(self, model_id: str = None, **config):
        self.model_id = model_id or f"ismcts_{uuid.uuid4().hex[:8]}"
        self.generation = 0
        
        # Tunable hyperparameters
        self.config = {
            'simulations_per_move': config.get('simulations_per_move', 150),
            'nil_threshold': config.get('nil_threshold', 0.4),
            'endgame_threshold': config.get('endgame_threshold', 5),
            'uct_c_param': config.get('uct_c_param', 1.4),
            
            # Heuristic weights (for future tuning)
            'ace_weight': config.get('ace_weight', 1.0),
            'king_weight': config.get('king_weight', 0.8),
            'queen_weight': config.get('queen_weight', 0.5),
            'jack_weight': config.get('jack_weight', 0.3),
            'long_suit_bonus': config.get('long_suit_bonus', 1.0),
            'void_penalty': config.get('void_penalty', 1.0),
            'trump_bonus': config.get('trump_bonus', 0.3),
        }
        
        # Performance tracking
        self.training_stats = {
            'games_played': 0,
            'wins': 0,
            'avg_move_time': 0.0,
            'total_simulations': 0
        }
    
    def get_id(self) -> str:
        return self.model_id
    
    def save(self, path: Path):
        """Save model configuration and stats."""
        state = {
            'model_id': self.model_id,
            'generation': self.generation,
            'config': self.config,
            'training_stats': self.training_stats
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: Path):
        """Load model state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.__dict__.update(state)
    
    def clone(self) -> 'ISMCTSModelWrapper':
        """Create evolved copy with parameter mutations."""
        clone = ISMCTSModelWrapper()
        clone.config = copy.deepcopy(self.config)
        clone.generation = self.generation + 1
        
        # Add small mutations for evolution
        mutation_rate = 0.1
        mutation_strength = 0.05
        
        for key, value in clone.config.items():
            if isinstance(value, (int, float)) and key != 'endgame_threshold':
                if random.random() < mutation_rate:
                    if isinstance(value, int):
                        delta = int(value * mutation_strength * random.uniform(-1, 1))
                        clone.config[key] = max(1, value + delta)
                    else:
                        delta = value * mutation_strength * random.uniform(-1, 1)
                        clone.config[key] = max(0.1, value + delta)
        
        return clone
    
    def create_bot(self, name: str) -> BotInterface:
        """Create ISMCTS bot instance with current config."""
        agent = ISMCTSAgent(name, self.config['simulations_per_move'])
        
        # Apply configuration
        agent.nil_threshold = self.config['nil_threshold']
        agent.endgame_threshold = self.config['endgame_threshold']
        
        # TODO: Apply heuristic weights when implemented
        
        return agent
    
    def update_from_performance(self, stats: Dict[str, Any]):
        """Update model based on performance feedback."""
        self.training_stats.update(stats)
        
        # Adaptive parameter tuning based on performance
        win_rate = stats.get('win_rate', 0.0)
        avg_move_time = stats.get('avg_move_time', 0.0)
        
        # If too slow, reduce simulations
        if avg_move_time > 0.5:
            self.config['simulations_per_move'] = max(50, 
                int(self.config['simulations_per_move'] * 0.9))
        
        # If winning too little and moving fast, increase simulations
        elif win_rate < 0.4 and avg_move_time < 0.1:
            self.config['simulations_per_move'] = min(300,
                int(self.config['simulations_per_move'] * 1.1))
        
        # Adaptive nil threshold based on success
        nil_success_rate = stats.get('nil_success_rate', 0.5)
        if nil_success_rate < 0.3:  # Too many failed nils
            self.config['nil_threshold'] += 0.05
        elif nil_success_rate > 0.8:  # Too conservative
            self.config['nil_threshold'] -= 0.05
        
        self.config['nil_threshold'] = max(0.2, min(0.8, self.config['nil_threshold']))


class ISMCTSTrainer:
    """Specialized trainer for ISMCTS agent evolution."""
    
    def __init__(self, base_config: Dict[str, Any] = None):
        self.base_config = base_config or {}
        self.population_size = 4
        self.generation = 0
        
    def create_initial_population(self) -> list[ISMCTSModelWrapper]:
        """Create diverse initial population."""
        population = []
        
        # Base configurations to try
        configs = [
            {'simulations_per_move': 100, 'nil_threshold': 0.3},
            {'simulations_per_move': 150, 'nil_threshold': 0.4},
            {'simulations_per_move': 200, 'nil_threshold': 0.5},
            {'simulations_per_move': 250, 'nil_threshold': 0.4},
        ]
        
        for i, config in enumerate(configs):
            merged_config = {**self.base_config, **config}
            model = ISMCTSModelWrapper(f"ismcts_gen0_{i}", **merged_config)
            population.append(model)
        
        return population
    
    def evolve_population(self, population: list[ISMCTSModelWrapper], 
                         performance_data: Dict[str, Dict]) -> list[ISMCTSModelWrapper]:
        """Evolve population based on performance."""
        
        # Sort by performance (win rate + efficiency)
        def fitness(model: ISMCTSModelWrapper) -> float:
            model_id = model.get_id()
            if model_id not in performance_data:
                return 0.0
            
            stats = performance_data[model_id]
            win_rate = stats.get('win_rate', 0.0)
            avg_time = stats.get('avg_move_time', 1.0)
            
            # Fitness = win_rate - time_penalty
            time_penalty = max(0, (avg_time - 0.2) * 0.5)  # Penalize > 0.2s moves
            return win_rate - time_penalty
        
        population.sort(key=fitness, reverse=True)
        
        # Next generation: keep top 2, evolve 2 new ones
        next_gen = []
        
        # Elite selection
        next_gen.extend(population[:2])
        
        # Mutation of best performers
        for i in range(2):
            parent = population[i % 2]
            child = parent.clone()
            next_gen.append(child)
        
        self.generation += 1
        return next_gen
    
    def run_training_session(self, num_generations: int = 10, 
                            games_per_generation: int = 50):
        """Run complete ISMCTS training session."""
        
        from .trainer import SelfPlayTrainer
        from ..game import EstimationGame
        from ..player import Player
        
        print(f"🤖 Starting ISMCTS training: {num_generations} generations")
        
        # Initialize population
        population = self.create_initial_population()
        
        for gen in range(num_generations):
            print(f"\n--- Generation {gen + 1}/{num_generations} ---")
            
            # Run games for current population
            performance_data = {}
            
            for model in population:
                model_stats = {
                    'games_played': 0,
                    'wins': 0,
                    'total_move_time': 0.0,
                    'total_simulations': 0,
                    'nil_attempts': 0,
                    'nil_successes': 0
                }
                
                # Run games with this model
                for game_num in range(games_per_generation // len(population)):
                    try:
                        # Create mixed population for game
                        players = []
                        for i in range(4):
                            if i == 0:
                                # Our model being tested
                                player = Player(i, f"{model.get_id()}_{i}")
                                player.strategy = model.create_bot(player.name)
                            else:
                                # Random opponent from population or baseline
                                opponent_model = random.choice(population)
                                player = Player(i, f"Opponent_{i}")
                                player.strategy = opponent_model.create_bot(player.name)
                            players.append(player)
                        
                        # Run game
                        game = EstimationGame(players, "MICRO")  # Fast games for training
                        final_scores = game.play_game()
                        winner_id, _ = game.get_winner()
                        
                        # Update stats
                        model_stats['games_played'] += 1
                        if winner_id == 0:  # Our model won
                            model_stats['wins'] += 1
                        
                        # Collect performance data from agent
                        agent = players[0].strategy
                        if hasattr(agent, 'get_performance_stats'):
                            perf_stats = agent.get_performance_stats()
                            model_stats['total_move_time'] += perf_stats.get('avg_move_time', 0)
                            model_stats['total_simulations'] += perf_stats.get('total_simulations', 0)
                        
                    except Exception as e:
                        print(f"Game failed for {model.get_id()}: {e}")
                        continue
                
                # Calculate final stats
                if model_stats['games_played'] > 0:
                    model_stats['win_rate'] = model_stats['wins'] / model_stats['games_played']
                    model_stats['avg_move_time'] = model_stats['total_move_time'] / model_stats['games_played']
                    model_stats['nil_success_rate'] = (model_stats['nil_successes'] / 
                                                     max(1, model_stats['nil_attempts']))
                else:
                    model_stats['win_rate'] = 0.0
                    model_stats['avg_move_time'] = 1.0
                    model_stats['nil_success_rate'] = 0.5
                
                performance_data[model.get_id()] = model_stats
                
                # Update model with performance feedback
                model.update_from_performance(model_stats)
                
                print(f"{model.get_id()}: WR={model_stats['win_rate']:.1%}, "
                      f"Time={model_stats['avg_move_time']:.3f}s, "
                      f"Sims={model_stats.get('total_simulations', 0)}")
            
            # Evolve population
            population = self.evolve_population(population, performance_data)
            
            # Save best model
            best_model = population[0]
            save_path = Path(f"ismcts_models/gen_{gen+1}_best.pkl")
            save_path.parent.mkdir(exist_ok=True)
            best_model.save(save_path)
        
        print(f"\n✅ Training complete! Best model: {population[0].get_id()}")
        return population[0]


def create_competitive_ismcts() -> ISMCTSModelWrapper:
    """Create competitive ISMCTS model with optimized parameters."""
    config = {
        'simulations_per_move': 180,  # Good balance of speed/accuracy
        'nil_threshold': 0.45,        # Slightly conservative
        'endgame_threshold': 6,       # Switch to MCTS bit earlier  
        'uct_c_param': 1.4,          # Standard UCB exploration
        
        # Tuned heuristic weights
        'ace_weight': 1.1,
        'king_weight': 0.85, 
        'queen_weight': 0.55,
        'jack_weight': 0.35,
        'long_suit_bonus': 1.2,
        'void_penalty': 0.9,
        'trump_bonus': 0.4,
    }
    
    return ISMCTSModelWrapper("ISMCTS_Competitive_v1", **config)


# Integration with existing training system
class ISMCTSIntegration:
    """Integration helper for existing training infrastructure."""
    
    @staticmethod
    def wrap_for_champion_system() -> 'HeuristicModelWrapper':
        """Create wrapper compatible with existing champion system."""
        
        class ISMCTSBotWrapper(BotInterface):
            def __init__(self, name: str):
                super().__init__(name)
                self.ismcts_model = create_competitive_ismcts()
                self.agent = self.ismcts_model.create_bot(name)
            
            def make_bid(self, hand, other_bids, is_last_bidder, can_bid_dash):
                return self.agent.make_bid(hand, other_bids, is_last_bidder, can_bid_dash)
            
            def make_estimation(self, hand, trump_suit, declarer_bid, current_estimations, 
                             is_last_estimator, can_dash):
                return self.agent.make_estimation(hand, trump_suit, declarer_bid, 
                                                current_estimations, is_last_estimator, can_dash)
            
            def choose_card(self, hand, valid_plays, trump_suit, led_suit, cards_played):
                return self.agent.choose_card(hand, valid_plays, trump_suit, led_suit, cards_played)
        
        # Create wrapper that creates our ISMCTS bot
        def bot_factory(name: str) -> BotInterface:
            return ISMCTSBotWrapper(name)
        
        wrapper = HeuristicModelWrapper()
        wrapper.model_id = "ISMCTS_Champion_v1"
        wrapper.bot_class = bot_factory
        
        return wrapper


if __name__ == "__main__":
    # Quick training test
    trainer = ISMCTSTrainer()
    best_model = trainer.run_training_session(num_generations=3, games_per_generation=20)
    
    print(f"Best model config: {best_model.config}")
    print("ISMCTS training system ready!")