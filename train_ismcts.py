#!/usr/bin/env python3
"""
ISMCTS Training Script - Rapid self-play tuning
Location: estimation_bot/train_ismcts.py

Quick training and evaluation script for ISMCTS agent.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import statistics

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from estimation_bot.game import EstimationGame
from estimation_bot.player import Player
from estimation_bot.training.ismcts_wrapper import ISMCTSModelWrapper, ISMCTSTrainer, create_competitive_ismcts
from bot.heuristic_bot import AdvancedHeuristicBot
from bot.random_bot import WeightedRandomBot


class ISMCTSEvaluator:
    """Evaluates ISMCTS agent performance against baselines."""
    
    def __init__(self):
        self.results = []
    
    def run_benchmark(self, ismcts_model: ISMCTSModelWrapper, 
                     num_games: int = 100, game_mode: str = "MICRO") -> Dict[str, Any]:
        """Run ISMCTS vs baseline bots benchmark."""
        
        print(f"🎯 Benchmarking ISMCTS agent ({num_games} games)")
        
        # Opponent types to test against
        opponents = [
            ("Advanced", AdvancedHeuristicBot),
            ("Weighted", WeightedRandomBot),
            ("Mixed", None)  # Mixed opponents
        ]
        
        results = {}
        
        for opp_name, opp_class in opponents:
            print(f"\nTesting vs {opp_name} opponents...")
            
            wins = 0
            total_time = 0
            total_simulations = 0
            game_scores = []
            
            for game_num in range(num_games):
                try:
                    # Create players
                    players = []
                    
                    # ISMCTS player (position 0)
                    ismcts_player = Player(0, "ISMCTS")
                    ismcts_player.strategy = ismcts_model.create_bot("ISMCTS")
                    players.append(ismcts_player)
                    
                    # Opponent players
                    for i in range(1, 4):
                        player = Player(i, f"{opp_name}_{i}")
                        
                        if opp_name == "Mixed":
                            # Alternate opponent types
                            if i % 2 == 1:
                                player.strategy = AdvancedHeuristicBot(f"{opp_name}_{i}")
                            else:
                                player.strategy = WeightedRandomBot(f"{opp_name}_{i}")
                        else:
                            player.strategy = opp_class(f"{opp_name}_{i}")
                        
                        players.append(player)
                    
                    # Run game
                    start_time = time.time()
                    game = EstimationGame(players, game_mode)
                    final_scores = game.play_game()
                    game_time = time.time() - start_time
                    
                    winner_id, winning_score = game.get_winner()
                    
                    # Collect results
                    if winner_id == 0:  # ISMCTS won
                        wins += 1
                    
                    total_time += game_time
                    game_scores.append(final_scores[0])  # ISMCTS score
                    
                    # Get agent performance stats
                    if hasattr(ismcts_player.strategy, 'get_performance_stats'):
                        agent_stats = ismcts_player.strategy.get_performance_stats()
                        total_simulations += agent_stats.get('total_simulations', 0)
                    
                    # Progress indicator
                    if (game_num + 1) % 20 == 0:
                        current_wr = wins / (game_num + 1)
                        print(f"  Progress: {game_num + 1}/{num_games} (WR: {current_wr:.1%})")
                
                except Exception as e:
                    print(f"Game {game_num} failed: {e}")
                    continue
            
            # Calculate final stats
            win_rate = wins / num_games
            avg_game_time = total_time / num_games
            avg_score = statistics.mean(game_scores) if game_scores else 0
            avg_sims_per_game = total_simulations / num_games if total_simulations > 0 else 0
            
            results[opp_name] = {
                'win_rate': win_rate,
                'avg_game_time': avg_game_time,
                'avg_score': avg_score,
                'avg_simulations_per_game': avg_sims_per_game,
                'total_games': num_games
            }
            
            print(f"  Results: {win_rate:.1%} WR, {avg_game_time:.2f}s/game, "
                  f"{avg_score:.1f} avg score, {avg_sims_per_game:.0f} sims/game")
        
        return results
    
    def quick_performance_test(self, ismcts_model: ISMCTSModelWrapper) -> Dict[str, float]:
        """Quick performance test focusing on speed and decision quality."""
        
        print("⚡ Quick performance test...")
        
        # Create test player
        player = Player(0, "TestAgent")
        agent = ismcts_model.create_bot("TestAgent")
        player.strategy = agent
        
        # Test different hand scenarios
        from estimation_bot.card import Card, Suit, Rank
        
        test_scenarios = [
            # Strong hand
            [Card(Suit.SPADES, Rank.ACE), Card(Suit.SPADES, Rank.KING),
             Card(Suit.HEARTS, Rank.QUEEN), Card(Suit.CLUBS, Rank.JACK),
             Card(Suit.DIAMONDS, Rank.TEN)],
            
            # Weak hand
            [Card(Suit.SPADES, Rank.TWO), Card(Suit.HEARTS, Rank.THREE),
             Card(Suit.CLUBS, Rank.FOUR), Card(Suit.DIAMONDS, Rank.FIVE),
             Card(Suit.SPADES, Rank.SIX)],
            
            # Mixed hand
            [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.TWO),
             Card(Suit.CLUBS, Rank.KING), Card(Suit.DIAMONDS, Rank.THREE),
             Card(Suit.SPADES, Rank.SEVEN)]
        ]
        
        bid_times = []
        card_times = []
        
        for i, hand in enumerate(test_scenarios):
            player.receive_cards(hand)
            
            # Test bidding
            start_time = time.time()
            bid_result = agent.make_bid(hand, {}, False, True)
            bid_time = time.time() - start_time
            bid_times.append(bid_time)
            
            # Test card play
            valid_plays = hand[:3]  # Simulate some valid plays
            start_time = time.time()
            chosen_card = agent.choose_card(hand, valid_plays, Suit.SPADES, None, [])
            card_time = time.time() - start_time
            card_times.append(card_time)
            
            print(f"  Scenario {i+1}: Bid={bid_result}, "
                  f"Play={chosen_card}, Time={bid_time+card_time:.3f}s")
        
        return {
            'avg_bid_time': statistics.mean(bid_times),
            'avg_card_time': statistics.mean(card_times),
            'max_decision_time': max(bid_times + card_times),
            'total_test_time': sum(bid_times + card_times)
        }


def main():
    parser = argparse.ArgumentParser(description="ISMCTS Agent Training & Evaluation")
    parser.add_argument('--mode', choices=['train', 'benchmark', 'quick'], 
                       default='quick', help='Run mode')
    parser.add_argument('--games', type=int, default=50, 
                       help='Number of games for benchmark')
    parser.add_argument('--generations', type=int, default=5,
                       help='Training generations')
    parser.add_argument('--simulations', type=int, default=150,
                       help='Simulations per move')
    
    args = parser.parse_args()
    
    print("🤖 ISMCTS Agent Training System")
    print("=" * 40)
    
    if args.mode == 'train':
        print(f"Training ISMCTS agent ({args.generations} generations)...")
        
        trainer = ISMCTSTrainer()
        best_model = trainer.run_training_session(
            num_generations=args.generations,
            games_per_generation=args.games
        )
        
        print(f"\n🏆 Best model configuration:")
        for key, value in best_model.config.items():
            print(f"  {key}: {value}")
        
        # Save best model
        save_path = Path("ismcts_trained_model.pkl")
        best_model.save(save_path)
        print(f"Saved to: {save_path}")
    
    elif args.mode == 'benchmark':
        print(f"Benchmarking ISMCTS agent ({args.games} games)...")
        
        # Create competitive model
        ismcts_model = create_competitive_ismcts()
        ismcts_model.config['simulations_per_move'] = args.simulations
        
        evaluator = ISMCTSEvaluator()
        results = evaluator.run_benchmark(ismcts_model, args.games)
        
        print(f"\n🏆 Final Benchmark Results:")
        print("-" * 40)
        for opponent, stats in results.items():
            print(f"{opponent:>12}: {stats['win_rate']:6.1%} WR | "
                  f"{stats['avg_game_time']:5.2f}s/game | "
                  f"{stats['avg_score']:6.1f} avg score")
        
        # Overall performance
        overall_wr = statistics.mean([r['win_rate'] for r in results.values()])
        overall_time = statistics.mean([r['avg_game_time'] for r in results.values()])
        
        print("-" * 40)
        print(f"{'OVERALL':>12}: {overall_wr:6.1%} WR | {overall_time:5.2f}s/game")
        
        # Performance rating
        if overall_wr >= 0.65:
            rating = "🏆 EXCELLENT"
        elif overall_wr >= 0.55:
            rating = "🥈 GOOD"
        elif overall_wr >= 0.45:
            rating = "🥉 AVERAGE"
        else:
            rating = "❌ NEEDS WORK"
        
        print(f"\nPerformance Rating: {rating}")
    
    else:  # quick mode
        print("Quick performance test...")
        
        ismcts_model = create_competitive_ismcts()
        ismcts_model.config['simulations_per_move'] = args.simulations
        
        evaluator = ISMCTSEvaluator()
        perf_stats = evaluator.quick_performance_test(ismcts_model)
        
        print(f"\n⚡ Performance Stats:")
        print(f"Avg bid time: {perf_stats['avg_bid_time']:.3f}s")
        print(f"Avg card time: {perf_stats['avg_card_time']:.3f}s") 
        print(f"Max decision time: {perf_stats['max_decision_time']:.3f}s")
        
        # Speed rating
        max_time = perf_stats['max_decision_time']
        if max_time <= 0.2:
            speed_rating = "🟢 FAST"
        elif max_time <= 0.5:
            speed_rating = "🟡 ACCEPTABLE"
        else:
            speed_rating = "🔴 SLOW"
        
        print(f"Speed Rating: {speed_rating}")
        
        # Run mini benchmark
        print(f"\nMini benchmark (10 games)...")
        mini_results = evaluator.run_benchmark(ismcts_model, 10, "MICRO")
        avg_wr = statistics.mean([r['win_rate'] for r in mini_results.values()])
        print(f"Quick win rate: {avg_wr:.1%}")


if __name__ == "__main__":
    main()