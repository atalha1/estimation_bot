"""
Training system for Estimation bots using self-play.
Supports bot evaluation, tournament play, and RL training.
"""

import json
import time
from typing import List, Dict, Any
from datetime import datetime
from ..game.game import EstimationGame
from ..game.player import Player
from ..game.utils import GameLogger, save_game_log
from .random_bot import RandomBot, WeightedRandomBot
from .heuristic_bot import HeuristicBot, AdvancedHeuristicBot


class SelfPlayTrainer:
    """Manages self-play training and bot evaluation."""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = GameLogger()
        self.results = []
        self.start_time = None
    
    def create_bot(self, bot_type: str, player_id: int) -> Player:
        """Create a bot player of specified type."""
        bot_classes = {
            'random': RandomBot,
            'weighted': WeightedRandomBot,
            'heuristic': HeuristicBot,
            'advanced': AdvancedHeuristicBot
        }
        
        if bot_type not in bot_classes:
            raise ValueError(f"Unknown bot type: {bot_type}")
        
        player = Player(player_id, f"{bot_type.title()}_{player_id}")
        player.strategy = bot_classes[bot_type](f"{bot_type.title()}_{player_id}")
        return player
    
    def run_single_game(self, bot_types: List[str], game_mode: str = "FULL", 
                       verbose: bool = False) -> Dict[str, Any]:
        """Run a single game between bots."""
        players = [self.create_bot(bot_type, i) for i, bot_type in enumerate(bot_types)]
        game = EstimationGame(players, game_mode)
        
        start_time = time.time()
        
        try:
            final_scores = game.play_game()
            winner_id, winning_score = game.get_winner()
            
            game_duration = time.time() - start_time
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'bot_types': bot_types,
                'game_mode': game_mode,
                'winner_id': winner_id,
                'winner_type': bot_types[winner_id],
                'final_scores': {f"{bot_types[i]}_{i}": score 
                               for i, score in final_scores.items()},
                'rounds_played': game.current_round,
                'duration_seconds': game_duration,
                'success': True
            }
            
            if verbose:
                self.logger.logger.info(f"Game complete: {bot_types[winner_id]} wins with {winning_score}")
            
            return result
            
        except Exception as e:
            self.logger.logger.error(f"Game failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'bot_types': bot_types,
                'game_mode': game_mode,
                'error': str(e),
                'success': False
            }
    
    def run_tournament(self, bot_types: List[str], games_per_matchup: int = 100,
                      game_mode: str = "FULL") -> Dict[str, Any]:
        """Run tournament between different bot types."""
        self.logger.logger.info(f"Starting tournament: {len(bot_types)} bot types, "
                               f"{games_per_matchup} games per matchup")
        
        tournament_results = {
            'start_time': datetime.now().isoformat(),
            'bot_types': bot_types,
            'games_per_matchup': games_per_matchup,
            'game_mode': game_mode,
            'matchups': [],
            'summary': {}
        }
        
        # Generate all possible 4-player matchups
        from itertools import combinations_with_replacement
        matchups = list(combinations_with_replacement(bot_types, 4))
        
        total_games = len(matchups) * games_per_matchup
        games_played = 0
        
        for matchup in matchups:
            matchup_results = []
            matchup_name = "_vs_".join(matchup)
            
            self.logger.logger.info(f"Running matchup: {matchup_name}")
            
            for game_num in range(games_per_matchup):
                result = self.run_single_game(list(matchup), game_mode)
                matchup_results.append(result)
                games_played += 1
                
                if games_played % 50 == 0:
                    self.logger.logger.info(f"Progress: {games_played}/{total_games} games")
            
            # Analyze matchup results
            wins_by_type = {}
            for bot_type in bot_types:
                wins_by_type[bot_type] = sum(1 for r in matchup_results 
                                           if r.get('winner_type') == bot_type)
            
            tournament_results['matchups'].append({
                'matchup': matchup,
                'results': matchup_results,
                'wins_by_type': wins_by_type,
                'total_games': len(matchup_results)
            })
        
        # Calculate overall tournament summary
        overall_wins = {bot_type: 0 for bot_type in bot_types}
        total_tournament_games = 0
        
        for matchup_data in tournament_results['matchups']:
            for bot_type, wins in matchup_data['wins_by_type'].items():
                overall_wins[bot_type] += wins
            total_tournament_games += matchup_data['total_games']
        
        tournament_results['summary'] = {
            'total_games': total_tournament_games,
            'overall_wins': overall_wins,
            'win_rates': {bot_type: wins / total_tournament_games 
                         for bot_type, wins in overall_wins.items()},
            'duration_seconds': time.time() - time.mktime(
                datetime.fromisoformat(tournament_results['start_time']).timetuple())
        }
        
        tournament_results['end_time'] = datetime.now().isoformat()
        
        # Save results
        filename = f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_game_log(tournament_results, f"data/logs/{filename}")
        
        self.logger.logger.info(f"Tournament complete! Results saved to {filename}")
        self.logger.logger.info(f"Win rates: {tournament_results['summary']['win_rates']}")
        
        return tournament_results
    
    def evaluate_bot(self, bot_type: str, opponent_types: List[str], 
                    num_games: int = 1000, game_mode: str = "FULL") -> Dict[str, Any]:
        """Evaluate a specific bot against various opponents."""
        self.logger.logger.info(f"Evaluating {bot_type} against {opponent_types} "
                               f"over {num_games} games")
        
        results = []
        wins = 0
        
        for game_num in range(num_games):
            # Create matchup with target bot in different positions
            position = game_num % 4
            matchup = opponent_types.copy()
            matchup[position] = bot_type
            
            result = self.run_single_game(matchup, game_mode)
            results.append(result)
            
            if result.get('winner_id') == position:
                wins += 1
            
            if (game_num + 1) % 100 == 0:
                current_win_rate = wins / (game_num + 1)
                self.logger.logger.info(f"Progress: {game_num + 1}/{num_games}, "
                                       f"Win rate: {current_win_rate:.3f}")
        
        evaluation = {
            'bot_type': bot_type,
            'opponent_types': opponent_types,
            'num_games': num_games,
            'wins': wins,
            'win_rate': wins / num_games,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save evaluation
        filename = f"evaluation_{bot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_game_log(evaluation, f"data/logs/{filename}")
        
        self.logger.logger.info(f"Evaluation complete: {bot_type} win rate = {wins/num_games:.3f}")
        
        return evaluation
    
    def run_training_games(self, num_episodes: int = 1000, game_mode: str = "MINI"):
        """Run training games for RL bot development."""
        self.logger.logger.info(f"Starting training run: {num_episodes} episodes")
        
        training_data = {
            'start_time': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            'game_mode': game_mode,
            'episodes': []
        }
        
        # For now, just collect game data - actual RL training would go here
        bot_types = ['random', 'weighted', 'heuristic', 'advanced']
        
        for episode in range(num_episodes):
            # Randomize bot matchup for diversity
            import random
            matchup = random.choices(bot_types, k=4)
            
            result = self.run_single_game(matchup, game_mode, verbose=False)
            training_data['episodes'].append(result)
            
            if (episode + 1) % 100 == 0:
                self.logger.logger.info(f"Training progress: {episode + 1}/{num_episodes}")
        
        training_data['end_time'] = datetime.now().isoformat()
        
        # Save training data
        filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_game_log(training_data, f"data/logs/{filename}")
        
        self.logger.logger.info(f"Training data collection complete: {filename}")
        
        return training_data


def run_quick_evaluation():
    """Quick evaluation script for testing."""
    trainer = SelfPlayTrainer()
    
    print("🤖 Running quick bot evaluation...")
    
    # Test all bot types against each other
    bot_types = ['random', 'weighted', 'heuristic', 'advanced']
    
    results = trainer.run_tournament(bot_types, games_per_matchup=25, game_mode="MICRO")
    
    print("\n📊 Results:")
    for bot_type, win_rate in results['summary']['win_rates'].items():
        print(f"  {bot_type}: {win_rate:.3f} win rate")
    
    return results


if __name__ == "__main__":
    run_quick_evaluation()