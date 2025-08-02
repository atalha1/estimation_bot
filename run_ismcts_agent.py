#!/usr/bin/env python3
"""
ISMCTS Final Integration - Complete competitive AI agent
Location: estimation_bot/run_ismcts_agent.py

Final integration script for the ISMCTS competitive agent.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from estimation_bot.training.champion import ChampionManager
from estimation_bot.training.ismcts_wrapper import ISMCTSModelWrapper, create_competitive_ismcts, ISMCTSIntegration
from estimation_bot.game import EstimationGame
from estimation_bot.player import Player
from bot.heuristic_bot import AdvancedHeuristicBot
from bot.random_bot import WeightedRandomBot


class ISMCTSChampion:
    """ISMCTS Champion Agent - Final competitive implementation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_optimal_config()
        self.model = ISMCTSModelWrapper("ISMCTS_Champion", **self.config)
        self.performance_history = []
        
    def _get_optimal_config(self) -> Dict[str, Any]:
        """Get optimized configuration based on target performance."""
        return {
            'simulations_per_move': 180,  # Balance of speed/accuracy for 0.2s target
            'nil_threshold': 0.45,        # Slightly conservative nil bidding
            'endgame_threshold': 6,       # Earlier MCTS switch for critical decisions
            'uct_c_param': 1.4,          # Standard exploration parameter
            
            # Heuristic weights (for enhanced version)
            'ace_weight': 1.1,
            'king_weight': 0.85,
            'queen_weight': 0.55,
            'jack_weight': 0.35,
            'long_suit_bonus': 1.2,
            'void_penalty': 0.9,
            'trump_bonus': 0.4,
        }
    
    def create_agent(self, name: str):
        """Create ISMCTS agent instance."""
        return self.model.create_bot(name)
    
    def run_competitive_benchmark(self, num_games: int = 100) -> Dict[str, Any]:
        """Run comprehensive benchmark against strong opponents."""
        
        print(f"🏆 Running ISMCTS Champion Benchmark ({num_games} games)")
        print("=" * 50)
        
        opponents = [
            ("Advanced Heuristic", AdvancedHeuristicBot),
            ("Weighted Random", WeightedRandomBot),
            ("Mixed Population", None)
        ]
        
        results = {}
        overall_stats = {
            'total_games': 0,
            'total_wins': 0,
            'total_time': 0.0,
            'total_simulations': 0
        }
        
        for opp_name, opp_class in opponents:
            print(f"\n🎯 Testing vs {opp_name}...")
            
            wins = 0
            scores = []
            game_times = []
            agent_stats = {'nil_attempts': 0, 'nil_successes': 0, 'total_simulations': 0}
            
            for game_num in range(num_games // len(opponents)):
                try:
                    # Create players
                    players = []
                    
                    # ISMCTS Champion (position 0)
                    ismcts_player = Player(0, "ISMCTS_Champion")
                    ismcts_agent = self.create_agent("ISMCTS_Champion")
                    ismcts_player.strategy = ismcts_agent
                    players.append(ismcts_player)
                    
                    # Create opponents
                    for i in range(1, 4):
                        player = Player(i, f"Opponent_{i}")
                        
                        if opp_name == "Mixed Population":
                            # Mix different opponent types
                            if i == 1:
                                player.strategy = AdvancedHeuristicBot(f"Advanced_{i}")
                            elif i == 2:
                                player.strategy = WeightedRandomBot(f"Weighted_{i}")
                            else:
                                # Use another ISMCTS as strong opponent
                                other_ismcts = self.create_agent(f"ISMCTS_Opp_{i}")
                                player.strategy = other_ismcts
                        else:
                            player.strategy = opp_class(f"{opp_name}_{i}")
                        
                        players.append(player)
                    
                    # Run game with timing
                    start_time = time.time()
                    game = EstimationGame(players, "MICRO")  # Fast games for benchmarking
                    final_scores = game.play_game()
                    game_time = time.time() - start_time
                    
                    winner_id, _ = game.get_winner()
                    
                    # Collect results
                    if winner_id == 0:  # ISMCTS won
                        wins += 1
                    
                    scores.append(final_scores[0])
                    game_times.append(game_time)
                    
                    # Get agent performance stats
                    if hasattr(ismcts_agent, 'get_performance_stats'):
                        perf_stats = ismcts_agent.get_performance_stats()
                        agent_stats['total_simulations'] += perf_stats.get('total_simulations', 0)
                        agent_stats['nil_attempts'] += perf_stats.get('nil_attempts', 0)
                        agent_stats['nil_successes'] += perf_stats.get('nil_successes', 0)
                    
                    # Progress indicator
                    if (game_num + 1) % 10 == 0:
                        current_wr = wins / (game_num + 1)
                        print(f"  Progress: {game_num + 1}/{num_games//len(opponents)} (WR: {current_wr:.1%})")
                
                except Exception as e:
                    print(f"Game {game_num} failed: {e}")
                    continue
            
            # Calculate stats for this opponent
            games_played = len(scores)
            win_rate = wins / games_played if games_played > 0 else 0
            avg_score = sum(scores) / games_played if scores else 0
            avg_game_time = sum(game_times) / games_played if game_times else 0
            nil_success_rate = (agent_stats['nil_successes'] / 
                              max(1, agent_stats['nil_attempts']))
            
            results[opp_name] = {
                'win_rate': win_rate,
                'avg_score': avg_score,
                'avg_game_time': avg_game_time,
                'nil_success_rate': nil_success_rate,
                'games_played': games_played
            }
            
            # Update overall stats
            overall_stats['total_games'] += games_played
            overall_stats['total_wins'] += wins
            overall_stats['total_time'] += sum(game_times)
            overall_stats['total_simulations'] += agent_stats['total_simulations']
            
            print(f"  Results: {win_rate:.1%} WR | {avg_score:.1f} avg score | "
                  f"{avg_game_time:.2f}s/game | {nil_success_rate:.1%} nil success")
        
        # Calculate overall performance
        overall_wr = overall_stats['total_wins'] / max(1, overall_stats['total_games'])
        overall_avg_time = overall_stats['total_time'] / max(1, overall_stats['total_games'])
        avg_sims_per_game = overall_stats['total_simulations'] / max(1, overall_stats['total_games'])
        
        print(f"\n{'='*50}")
        print(f"🏆 OVERALL PERFORMANCE")
        print(f"{'='*50}")
        print(f"Win Rate:        {overall_wr:.1%}")
        print(f"Games Played:    {overall_stats['total_games']}")
        print(f"Avg Game Time:   {overall_avg_time:.2f}s")
        print(f"Avg Sims/Game:   {avg_sims_per_game:.0f}")
        
        # Performance rating
        if overall_wr >= 0.70:
            rating = "🥇 SUPERHUMAN"
        elif overall_wr >= 0.60:
            rating = "🏆 EXPERT"
        elif overall_wr >= 0.50:
            rating = "🥈 STRONG"
        elif overall_wr >= 0.40:
            rating = "🥉 COMPETENT"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
        
        print(f"Performance:     {rating}")
        
        # Speed rating
        if overall_avg_time <= 2.0:
            speed_rating = "🟢 FAST"
        elif overall_avg_time <= 5.0:
            speed_rating = "🟡 ACCEPTABLE"
        else:
            speed_rating = "🔴 SLOW"
        
        print(f"Speed:           {speed_rating}")
        
        return {
            'overall_win_rate': overall_wr,
            'overall_avg_time': overall_avg_time,
            'opponent_results': results,
            'rating': rating,
            'speed_rating': speed_rating,
            'total_games': overall_stats['total_games']
        }
    
    def integrate_with_training_system(self):
        """Integrate ISMCTS champion with existing training system."""
        
        print("🔗 Integrating ISMCTS with existing training system...")
        
        # Create wrapper for champion system
        champion_wrapper = ISMCTSIntegration.wrap_for_champion_system()
        
        # Load existing champion manager
        champion_mgr = ChampionManager()
        
        # Run validation games
        print("Running validation against current champion...")
        
        validation_games = 50
        ismcts_wins = 0
        
        for game_num in range(validation_games):
            try:
                players = []
                
                # Half ISMCTS, half current champion
                for i in range(4):
                    player = Player(i, f"Player_{i}")
                    if i < 2:  # ISMCTS players
                        player.strategy = self.create_agent(f"ISMCTS_{i}")
                    else:  # Current champion
                        current_champion = champion_mgr.get_champion()
                        player.strategy = current_champion.create_bot(f"Champion_{i}")
                    players.append(player)
                
                game = EstimationGame(players, "MICRO")
                final_scores = game.play_game()
                winner_id, _ = game.get_winner()
                
                if winner_id < 2:  # ISMCTS won
                    ismcts_wins += 1
                
                if (game_num + 1) % 10 == 0:
                    current_wr = ismcts_wins / (game_num + 1)
                    print(f"  Validation progress: {game_num + 1}/{validation_games} (WR: {current_wr:.1%})")
            
            except Exception as e:
                print(f"Validation game {game_num} failed: {e}")
                continue
        
        validation_wr = ismcts_wins / validation_games
        print(f"Validation complete: {validation_wr:.1%} win rate vs current champion")
        
        if validation_wr > 0.55:  # Significant improvement
            print("🚀 ISMCTS Champion is stronger! Consider updating training system.")
        else:
            print("📊 ISMCTS performance is competitive but not clearly superior.")
        
        return validation_wr


def main():
    parser = argparse.ArgumentParser(description="ISMCTS Champion Agent")
    parser.add_argument('--mode', choices=['benchmark', 'integrate', 'quick'], 
                       default='benchmark', help='Run mode')
    parser.add_argument('--games', type=int, default=150, 
                       help='Number of games for benchmark')
    parser.add_argument('--simulations', type=int, default=180,
                       help='Simulations per move')
    
    args = parser.parse_args()
    
    print("🤖 ISMCTS Champion Agent")
    print("Advanced Estimation AI using hybrid ISMCTS + Heuristics")
    print("=" * 60)
    
    # Create champion with custom config
    config = {
        'simulations_per_move': args.simulations,
        'nil_threshold': 0.45,
        'endgame_threshold': 6,
        'uct_c_param': 1.4
    }
    
    champion = ISMCTSChampion(config)
    
    if args.mode == 'benchmark':
        print(f"Running comprehensive benchmark ({args.games} games)...")
        results = champion.run_competitive_benchmark(args.games)
        
        print(f"\n📊 BENCHMARK SUMMARY:")
        print(f"Target: >65% win rate vs strong opponents")
        print(f"Actual: {results['overall_win_rate']:.1%} win rate")
        print(f"Speed: {results['overall_avg_time']:.2f}s per game")
        print(f"Rating: {results['rating']}")
        
        if results['overall_win_rate'] >= 0.65:
            print("✅ TARGET ACHIEVED: Superhuman-level performance!")
        else:
            print("⚠️  Performance below target. Consider tuning.")
    
    elif args.mode == 'integrate':
        print("Integrating with existing training system...")
        validation_wr = champion.integrate_with_training_system()
        
        if validation_wr > 0.55:
            print("\n🎯 RECOMMENDATION: Replace current champion with ISMCTS")
            print("  - Superior performance validated")
            print("  - Can serve as new baseline for training")
        else:
            print("\n📈 RECOMMENDATION: Use ISMCTS as additional training opponent")
            print("  - Provides different strategic approach")
            print("  - Strengthens overall training population")
    
    else:  # quick mode
        print("Quick performance test...")
        
        # Run mini benchmark
        mini_results = champion.run_competitive_benchmark(30)
        
        print(f"\n⚡ QUICK RESULTS:")
        print(f"Win Rate: {mini_results['overall_win_rate']:.1%}")
        print(f"Speed: {mini_results['overall_avg_time']:.2f}s/game")
        print(f"Rating: {mini_results['rating']}")
        
        # Test specific scenarios
        print(f"\n🧪 Testing decision quality...")
        agent = champion.create_agent("TestAgent")
        
        # Test nil bid decision
        from estimation_bot.card import Card, Suit, Rank
        
        nil_test_hand = [
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.HEARTS, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR),
            Card(Suit.DIAMONDS, Rank.FIVE),
            Card(Suit.SPADES, Rank.SIX)
        ]
        
        start_time = time.time()
        bid_result = agent.make_bid(nil_test_hand, {}, False, True)
        bid_time = time.time() - start_time
        
        print(f"Nil test hand: {bid_result} (decided in {bid_time:.3f}s)")
        
        if bid_result == "DASH":
            print("✅ Correctly identified nil opportunity")
        else:
            print("⚠️  Missed nil opportunity or too conservative")


if __name__ == "__main__":
    main()