"""
Example script demonstrating self-play training.
Location: train_selfplay.py
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from training.trainer import SelfPlayTrainer
from training.models import HeuristicModelWrapper, LearnableModelWrapper
from bot.heuristic_bot import AdvancedHeuristicBot


def main():
    parser = argparse.ArgumentParser(description="Train Estimation bot through self-play")
    parser.add_argument('--generations', type=int, default=10,
                       help='Number of generations to train')
    parser.add_argument('--games-per-gen', type=int, default=100,
                       help='Games per generation')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--game-mode', choices=['MICRO', 'MINI', 'FULL'],
                       default='MICRO', help='Game mode for training')
    parser.add_argument('--data-dir', type=str, default='training_data',
                       help='Directory to save training data')
    parser.add_argument('--base-bot', choices=['heuristic', 'advanced', 'learnable'],
                       default='advanced', help='Base bot type to start with')
    
    args = parser.parse_args()
    
    print("🎯 Estimation Bot Self-Play Training 🎯")
    print("=" * 50)
    print(f"Generations: {args.generations}")
    print(f"Games per generation: {args.games_per_gen}")
    print(f"Game mode: {args.game_mode}")
    print(f"Base bot: {args.base_bot}")
    print("=" * 50)
    
    # Create base model
    if args.base_bot == 'advanced':
        base_model = HeuristicModelWrapper(AdvancedHeuristicBot)
    elif args.base_bot == 'heuristic':
        base_model = HeuristicModelWrapper()
    else:  # learnable
        base_model = LearnableModelWrapper()
    
    # Initialize trainer
    trainer = SelfPlayTrainer(
        base_model=base_model,
        data_dir=args.data_dir,
        num_workers=args.workers,
        games_per_generation=args.games_per_gen,
        game_mode=args.game_mode
    )
    
    # Run training
    try:
        trainer.train(args.generations)
        print("\n✅ Training completed successfully!")
        
        # Show final statistics
        print("\nFinal Model Statistics:")
        for model_id, stats in trainer.model_stats.items():
            if stats.games_played > 0:
                print(f"\n{model_id}:")
                print(f"  Games: {stats.games_played}")
                print(f"  Win Rate: {stats.win_rate:.2%}")
                print(f"  Avg Score: {stats.avg_score_per_game:.1f}")
                print(f"  Est. Accuracy: {stats.estimation_accuracy:.2%}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()