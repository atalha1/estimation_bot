"""
Self-play training system for Estimation Bot.
Manages parallel game simulations, model evolution, and data collection.
Location: training/trainer.py
"""

import multiprocessing as mp
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import logging
from abc import ABC, abstractmethod

from estimation_bot.game import EstimationGame
from estimation_bot.player import Player, BotInterface


@dataclass
class GameResult:
    """Stores comprehensive game data for training."""
    game_id: str
    timestamp: datetime
    players: List[str]  # Bot IDs
    winner_id: int
    final_scores: Dict[int, int]
    game_mode: str
    rounds: List[Dict]  # Detailed round data
    
    # Performance metrics
    score_deltas: Dict[int, int]  # Score change per player
    win_margins: Dict[int, int]   # How much each player won/lost by
    estimation_accuracy: Dict[int, float]  # Avg accuracy per player
    successful_calls: Dict[int, int]  # Successful CALL/WITH/DASH per player
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'game_id': self.game_id,
            'timestamp': self.timestamp.isoformat(),
            'players': self.players,
            'winner_id': self.winner_id,
            'final_scores': self.final_scores,
            'game_mode': self.game_mode,
            'rounds': self.rounds,
            'score_deltas': self.score_deltas,
            'win_margins': self.win_margins,
            'estimation_accuracy': self.estimation_accuracy,
            'successful_calls': self.successful_calls
        }


@dataclass
class ModelStats:
    """Tracks performance statistics for a model."""
    model_id: str
    games_played: int = 0
    wins: int = 0
    total_score: int = 0
    estimation_accuracy: float = 0.0
    successful_calls: int = 0
    failed_calls: int = 0
    
    # Advanced metrics
    avg_score_per_game: float = 0.0
    win_rate: float = 0.0
    avg_margin_of_victory: float = 0.0
    consistency_score: float = 0.0  # Std dev of scores
    
    def update(self, game_result: GameResult, player_id: int):
        """Update stats based on game result."""
        self.games_played += 1
        if game_result.winner_id == player_id:
            self.wins += 1
        self.total_score += game_result.final_scores[player_id]
        
        # Update running averages
        self.win_rate = self.wins / self.games_played
        self.avg_score_per_game = self.total_score / self.games_played
        
        # Update accuracy if available
        if player_id in game_result.estimation_accuracy:
            alpha = 0.1  # Learning rate for moving average
            self.estimation_accuracy = (1 - alpha) * self.estimation_accuracy + \
                                     alpha * game_result.estimation_accuracy[player_id]


class ModelInterface(ABC):
    """Abstract interface for trainable models."""
    
    @abstractmethod
    def get_id(self) -> str:
        """Return unique model identifier."""
        pass
    
    @abstractmethod
    def save(self, path: Path):
        """Save model state."""
        pass
    
    @abstractmethod
    def load(self, path: Path):
        """Load model state."""
        pass
    
    @abstractmethod
    def clone(self) -> 'ModelInterface':
        """Create a copy of the model."""
        pass
    
    @abstractmethod
    def create_bot(self, name: str) -> BotInterface:
        """Create a bot instance using this model."""
        pass


class SelfPlayTrainer:
    """Manages self-play training loop and model evolution."""
    
    def __init__(self, 
                 base_model: ModelInterface,
                 data_dir: str = "training_data",
                 num_workers: int = None,
                 games_per_generation: int = 100,
                 game_mode: str = "MICRO"):
        """
        Initialize trainer.
        
        Args:
            base_model: Initial model to start training from
            data_dir: Directory to store training data
            num_workers: Number of parallel workers (defaults to CPU count)
            games_per_generation: Games to play before model selection
            game_mode: Game mode for training (MICRO/MINI/FULL)
        """
        self.base_model = base_model
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.num_workers = num_workers or mp.cpu_count()
        self.games_per_generation = games_per_generation
        self.game_mode = game_mode
        
        self.generation = 0
        self.all_game_results: List[GameResult] = []
        self.model_stats: Dict[str, ModelStats] = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup training logger."""
        logger = logging.getLogger('SelfPlayTrainer')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.data_dir / 'training.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def train(self, num_generations: int):
        """
        Run training for specified number of generations.
        
        Args:
            num_generations: Number of evolution cycles to run
        """
        self.logger.info(f"Starting training for {num_generations} generations")
        self.logger.info(f"Games per generation: {self.games_per_generation}")
        self.logger.info(f"Workers: {self.num_workers}")
        
        current_champion = self.base_model
        
        for gen in range(num_generations):
            self.generation = gen
            self.logger.info(f"\n=== Generation {gen + 1}/{num_generations} ===")
            
            # Create population (champion + 3 clones)
            population = self._create_population(current_champion)
            
            # Run parallel games
            game_results = self._run_generation(population)
            
            # Analyze results
            stats = self._analyze_generation(game_results, population)
            
            # Select new champion
            new_champion = self._select_champion(population, stats)
            
            # Save checkpoint
            self._save_checkpoint(new_champion, stats, game_results)
            
            # Log progress
            self._log_generation_summary(stats)
            
            current_champion = new_champion
        
        self.logger.info("\nTraining complete!")
        self._save_final_results()
    
    def _create_population(self, champion: ModelInterface) -> List[ModelInterface]:
        """Create population of 4 models (champion + 3 clones)."""
        population = [champion]
        for i in range(3):
            clone = champion.clone()
            # Could add mutation/noise here for diversity
            population.append(clone)
        return population
    
    def _run_generation(self, population: List[ModelInterface]) -> List[GameResult]:
        """Run games in parallel for current generation."""
        game_results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit game tasks
            futures = []
            for i in range(self.games_per_generation):
                future = executor.submit(self._simulate_game, population, i)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    game_results.append(result)
                    self.all_game_results.append(result)
                except Exception as e:
                    self.logger.error(f"Game simulation failed: {e}")
        
        return game_results
    
    def _simulate_game(self, population: List[ModelInterface], game_num: int) -> GameResult:
        """Simulate a single game with 4 models."""
        # Create players with bots from models
        players = []
        for i, model in enumerate(population):
            player = Player(i, f"{model.get_id()}_{i}")
            player.strategy = model.create_bot(player.name)
            players.append(player)
        
        # Run game
        game = EstimationGame(players, self.game_mode)
        game_id = f"gen{self.generation}_game{game_num}"
        
        try:
            final_scores = game.play_game()
            winner_id, _ = game.get_winner()
            
            # Calculate metrics
            result = self._create_game_result(game, game_id, population)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in game {game_id}: {e}")
            raise
    
    def _create_game_result(self, game: EstimationGame, game_id: str, 
                           population: List[ModelInterface]) -> GameResult:
        """Extract comprehensive game data for training."""
        # Basic info
        winner_id, winning_score = game.get_winner()
        final_scores = {i: p.score for i, p in enumerate(game.players)}
        
        # Calculate score deltas (final - expected)
        expected_score = sum(final_scores.values()) / len(final_scores)
        score_deltas = {i: score - expected_score for i, score in final_scores.items()}
        
        # Win margins
        win_margins = {i: final_scores[winner_id] - score 
                      for i, score in final_scores.items()}
        
        # Estimation accuracy per player
        estimation_accuracy = {}
        successful_calls = defaultdict(int)
        
        for round_data in game.game_data['rounds']:
            for player_id in range(4):
                estimated = round_data['estimations'].get(player_id, 0)
                actual = round_data['actual_tricks'].get(player_id, 0)
                
                if player_id not in estimation_accuracy:
                    estimation_accuracy[player_id] = []
                
                accuracy = 1.0 - abs(estimated - actual) / max(estimated, 1)
                estimation_accuracy[player_id].append(accuracy)
                
                # Track successful special calls
                if actual == estimated:
                    if player_id == round_data.get('declarer_id'):
                        successful_calls[player_id] += 2  # CALL bonus
                    elif player_id in round_data.get('with_players', []):
                        successful_calls[player_id] += 2  # WITH bonus
                    elif player_id in round_data.get('dash_players', []):
                        successful_calls[player_id] += 3  # DASH bonus
        
        # Average accuracy
        for player_id in estimation_accuracy:
            estimation_accuracy[player_id] = np.mean(estimation_accuracy[player_id])
        
        return GameResult(
            game_id=game_id,
            timestamp=datetime.now(),
            players=[model.get_id() for model in population],
            winner_id=winner_id,
            final_scores=final_scores,
            game_mode=game.game_mode,
            rounds=game.game_data['rounds'],
            score_deltas=score_deltas,
            win_margins=win_margins,
            estimation_accuracy=dict(estimation_accuracy),
            successful_calls=dict(successful_calls)
        )
    
    def _analyze_generation(self, game_results: List[GameResult], 
                           population: List[ModelInterface]) -> Dict[str, ModelStats]:
        """Analyze game results and update model statistics."""
        stats = {}
        
        # Initialize stats for each model
        for i, model in enumerate(population):
            model_id = f"{model.get_id()}_{i}"
            if model_id not in self.model_stats:
                self.model_stats[model_id] = ModelStats(model_id)
            stats[model_id] = self.model_stats[model_id]
        
        # Update stats from games
        for result in game_results:
            for i, model_id in enumerate(result.players):
                if model_id in stats:
                    stats[model_id].update(result, i)
        
        return stats
    
    def _select_champion(self, population: List[ModelInterface], 
                        stats: Dict[str, ModelStats]) -> ModelInterface:
        """Select best performing model as new champion."""
        # Calculate composite score for each model
        scores = {}
        
        for i, model in enumerate(population):
            model_id = f"{model.get_id()}_{i}"
            if model_id not in stats:
                continue
                
            stat = stats[model_id]
            
            # Composite score weighing different factors
            composite = (
                stat.win_rate * 0.4 +
                (stat.avg_score_per_game / 100) * 0.3 +
                stat.estimation_accuracy * 0.2 +
                (stat.successful_calls / max(stat.games_played, 1)) * 0.1
            )
            
            scores[i] = composite
        
        # Select model with highest score
        if scores:
            champion_idx = max(scores.keys(), key=lambda k: scores[k])
            self.logger.info(f"New champion: Model {champion_idx} with score {scores[champion_idx]:.3f}")
            return population[champion_idx]
        else:
            return population[0]  # Default to first
    
    def _save_checkpoint(self, champion: ModelInterface, stats: Dict[str, ModelStats], 
                        game_results: List[GameResult]):
        """Save training checkpoint."""
        checkpoint_dir = self.data_dir / f"generation_{self.generation}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save champion model
        champion.save(checkpoint_dir / "champion_model.pkl")
        
        # Save statistics
        with open(checkpoint_dir / "stats.json", 'w') as f:
            stats_dict = {k: v.__dict__ for k, v in stats.items()}
            json.dump(stats_dict, f, indent=2)
        
        # Save game results
        with open(checkpoint_dir / "games.json", 'w') as f:
            results_dict = [r.to_dict() for r in game_results]
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _log_generation_summary(self, stats: Dict[str, ModelStats]):
        """Log summary of generation performance."""
        self.logger.info("\nGeneration Summary:")
        for model_id, stat in stats.items():
            self.logger.info(f"  {model_id}:")
            self.logger.info(f"    Win Rate: {stat.win_rate:.2%}")
            self.logger.info(f"    Avg Score: {stat.avg_score_per_game:.1f}")
            self.logger.info(f"    Est. Accuracy: {stat.estimation_accuracy:.2%}")
    
    def _save_final_results(self):
        """Save complete training results."""
        # Save all game data
        with open(self.data_dir / "all_games.json", 'w') as f:
            all_games = [r.to_dict() for r in self.all_game_results]
            json.dump(all_games, f, indent=2)
        
        # Save final statistics
        with open(self.data_dir / "final_stats.json", 'w') as f:
            final_stats = {k: v.__dict__ for k, v in self.model_stats.items()}
            json.dump(final_stats, f, indent=2)
        
        self.logger.info(f"Final results saved to {self.data_dir}")