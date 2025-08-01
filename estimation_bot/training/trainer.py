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
import signal

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
    """Fixed model statistics tracking."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.games_played = 0
        self.wins = 0
        self.total_score = 0
        self.estimation_accuracy_sum = 0.0
        self.estimation_count = 0  # Track number of estimations for proper averaging
        self.successful_calls = 0
        
        # Derived properties
        self.win_rate = 0.0
        self.avg_score_per_game = 0.0
        self.estimation_accuracy = 0.0
    
    def update(self, game_result: GameResult, player_id: int):
        """Update statistics with proper averaging."""
        self.games_played += 1
        
        if game_result.winner_id == player_id:
            self.wins += 1
        
        self.total_score += game_result.final_scores[player_id]
        
        # Update derived stats
        self.win_rate = self.wins / self.games_played
        self.avg_score_per_game = self.total_score / self.games_played
        
        # Handle estimation accuracy properly
        if player_id in game_result.estimation_accuracy:
            accuracy = game_result.estimation_accuracy[player_id]
            if not np.isnan(accuracy):  # Guard against NaN values
                self.estimation_accuracy_sum += accuracy
                self.estimation_count += 1
                self.estimation_accuracy = self.estimation_accuracy_sum / self.estimation_count
        
        # Update successful calls
        if player_id in game_result.successful_calls:
            self.successful_calls += game_result.successful_calls[player_id]


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
        """Run training against progressively stronger opponents."""
        from .champion import ChampionManager
        
        champion_mgr = ChampionManager()
        champion_mgr.increment_training_session()
        
        current_champion = champion_mgr.get_champion()
        
        print(f"🎯 Training NADL v{champion_mgr.generation_count}.0")
        print(f"Training sessions: {champion_mgr.champion_stats['training_sessions']}")
        
        for gen in range(num_generations):
            self.generation = gen
            
            # Get strong opponents for this generation
            opponents = champion_mgr.get_training_opponents()
            population = [current_champion] + opponents
            
            # Run games with reduced logging
            game_results = self._run_generation_quiet(population)
            
            # Analyze results
            stats = self._analyze_generation(game_results, population)
            
            # Check if champion improved
            champion_stats = stats.get(current_champion.get_id())
            if champion_stats and champion_mgr.update_champion(current_champion, champion_stats):
                current_champion = champion_mgr.get_champion()
            
            # Brief progress update every 5 generations
            if (gen + 1) % 5 == 0:
                self._log_brief_progress(gen + 1, num_generations, champion_stats)
        
        self._save_final_results()
        print(f"✅ Training complete! NADL v{champion_mgr.generation_count}.0 ready")

    def _run_generation_quiet(self, population: List[ModelInterface]) -> List[GameResult]:
        """Run games with minimal logging."""
        game_results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.games_per_generation):
                future = executor.submit(self._simulate_game, population, i)
                futures.append(future)
            
            # Show progress bar instead of individual game logs
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result() 
                    game_results.append(result)
                    self.all_game_results.append(result)
                    completed += 1
                    
                    # Simple progress indicator
                    if completed % (self.games_per_generation // 4) == 0:
                        progress = (completed / self.games_per_generation) * 100
                        print(f"Progress: {progress:.0f}% ({completed}/{self.games_per_generation})")
                        
                except Exception as e:
                    self.logger.error(f"Game simulation failed: {e}")
        
        return game_results

    def _log_brief_progress(self, current_gen: int, total_gens: int, champion_stats: ModelStats):
        """Brief progress logging."""
        if champion_stats:
            print(f"Gen {current_gen}/{total_gens} | Win Rate: {champion_stats.win_rate:.1%} | "
                f"Avg Score: {champion_stats.avg_score_per_game:.1f}")
    
    def _create_population(self, champion: ModelInterface) -> List[ModelInterface]:
        """Create population of 4 models (champion + 3 clones)."""
        population = [champion]
        for i in range(3):
            clone = champion.clone()
            # Could add mutation/noise here for diversity
            population.append(clone)
        return population
    
    def _run_generation(self, population: List[ModelInterface]) -> List[GameResult]:
        """Run games with minimal logging."""
        game_results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(self.games_per_generation):
                future = executor.submit(self._simulate_game_robust, population, i)  # Use robust version
                futures.append(future)
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result() 
                    if result is not None:  # Only add valid results
                        game_results.append(result)
                        self.all_game_results.append(result)
                    completed += 1
                    
                    if completed % (self.games_per_generation // 4) == 0:
                        progress = (completed / self.games_per_generation) * 100
                        valid_count = len([r for r in game_results if r is not None])
                        print(f"Progress: {progress:.0f}% ({completed}/{self.games_per_generation}, {valid_count} valid)")
                        
                except Exception as e:
                    self.logger.error(f"Game simulation failed: {e}")
                    completed += 1
        
        return game_results
    
    def _simulate_game(self, population: List[ModelInterface], game_num: int) -> Optional[GameResult]:
        """Simulate game with robust error handling (Windows compatible)."""
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Create players with bots from models
                players = []
                for i, model in enumerate(population):
                    player = Player(i, f"{model.get_id()}_{i}")
                    player.strategy = model.create_bot(player.name)
                    players.append(player)
                
                # Run game 
                game = EstimationGame(players, self.game_mode)
                game_id = f"gen{self.generation}_game{game_num}_attempt{attempt}"
                
                # Simple timeout using threading (cross-platform)
                import threading
                import time
                
                result_container = {'result': None, 'error': None}
                
                def run_game():
                    try:
                        final_scores = game.play_game()
                        winner_id, _ = game.get_winner()
                        result_container['result'] = (final_scores, winner_id)
                    except Exception as e:
                        result_container['error'] = e
                
                # Start game in separate thread
                game_thread = threading.Thread(target=run_game, daemon=True)
                game_thread.start()
                game_thread.join(timeout=300)  # 5 minute timeout
                
                if game_thread.is_alive():
                    self.logger.warning(f"Game {game_id} timed out")
                    last_error = "Timeout"
                    continue
                
                if result_container['error']:
                    raise result_container['error']
                
                if result_container['result']:
                    final_scores, winner_id = result_container['result']
                    # Create detailed result
                    result = self._create_game_result(game, game_id, population, final_scores, winner_id)
                    return result
                else:
                    last_error = "No result returned"
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error in game {game_id}: {e}")
                last_error = str(e)
                continue
        
        # If all retries failed, return a dummy result
        self.logger.error(f"All attempts failed for game {game_num}, last error: {last_error}")
        return None
    
    def _create_game_result(self, game: EstimationGame, game_id: str, 
                               population: List[ModelInterface], final_scores: Dict[int, int],
                               winner_id: int) -> GameResult:
        """Create comprehensive game result with enhanced metrics."""
        
        # Calculate performance metrics
        total_score = sum(final_scores.values())
        avg_score = total_score / len(final_scores)
        
        score_deltas = {i: score - avg_score for i, score in final_scores.items()}
        win_margins = {i: final_scores[winner_id] - score for i, score in final_scores.items()}
        
        # Enhanced estimation accuracy calculation
        estimation_accuracy = {}
        successful_calls = defaultdict(int)
        
        for round_data in game.game_data.get('rounds', []):
            for player_id in range(4):
                estimated = round_data.get('estimations', {}).get(player_id, 0)
                actual = round_data.get('actual_tricks', {}).get(player_id, 0)
                
                if player_id not in estimation_accuracy:
                    estimation_accuracy[player_id] = []
                
                # More sophisticated accuracy calculation
                if estimated == 0:  # Dash estimation
                    accuracy = 1.0 if actual == 0 else 0.0
                else:
                    max_possible = max(estimated, actual, 1)
                    accuracy = 1.0 - (abs(estimated - actual) / max_possible)
                
                estimation_accuracy[player_id].append(accuracy)
                
                # Track successful special calls
                if actual == estimated:
                    if player_id == round_data.get('declarer_id'):
                        successful_calls[player_id] += 3  # CALL success
                    elif player_id in round_data.get('with_players', []):
                        successful_calls[player_id] += 2  # WITH success
                    elif player_id in round_data.get('dash_players', []):
                        successful_calls[player_id] += 4  # DASH success
                    else:
                        successful_calls[player_id] += 1  # Normal success
        
        # Average accuracy per player
        for player_id in estimation_accuracy:
            if estimation_accuracy[player_id]:
                estimation_accuracy[player_id] = np.mean(estimation_accuracy[player_id])
            else:
                estimation_accuracy[player_id] = 0.0
        
        return GameResult(
            game_id=game_id,
            timestamp=datetime.now(),
            players=[model.get_id() for model in population],
            winner_id=winner_id,
            final_scores=final_scores,
            game_mode=game.game_mode,
            rounds=game.game_data.get('rounds', []),
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
            base_id = model.get_id()
            if base_id not in self.model_stats:
                self.model_stats[base_id] = ModelStats(base_id)
            stats[base_id] = self.model_stats[base_id]
        
        # Update stats from games (filter out None results)
        valid_results = [result for result in game_results if result is not None]
        
        for result in valid_results:
            for i, model_id in enumerate(result.players):
                if model_id in stats:
                    stats[model_id].update(result, i)
        
        self.logger.info(f"Processed {len(valid_results)}/{len(game_results)} valid game results")
        
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
        # Filter out None results before saving
        valid_results = [r for r in self.all_game_results if r is not None]
        
        # Save all game data
        with open(self.data_dir / "all_games.json", 'w') as f:
            all_games = [r.to_dict() for r in valid_results]
            json.dump(all_games, f, indent=2)
        
        # Save final statistics
        with open(self.data_dir / "final_stats.json", 'w') as f:
            final_stats = {k: v.__dict__ for k, v in self.model_stats.items()}
            json.dump(final_stats, f, indent=2)
        
        self.logger.info(f"Final results saved to {self.data_dir} ({len(valid_results)} valid games)")