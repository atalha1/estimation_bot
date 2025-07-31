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
    parser = argparse.ArgumentParser(description="Train NADL through self-play")
    parser.add_argument('--generations', type=int, default=20,
                       help='Number of generations to train')
    parser.add_argument('--games-per-gen', type=int, default=50,
                       help='Games per generation')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--game-mode', choices=['MICRO', 'MINI', 'FULL'],
                       default='MICRO', help='Game mode for training')
    
    args = parser.parse_args()
    
    print("🎯 NADL Training System")
    print("=" * 30)
    
    # Initialize trainer with champion management
    from training.champion import ChampionManager
    champion_mgr = ChampionManager()
    base_model = champion_mgr.get_champion()
    
    trainer = SelfPlayTrainer(
        base_model=base_model,
        num_workers=args.workers,
        games_per_generation=args.games_per_gen,
        game_mode=args.game_mode
    )
    
    # Run training
    try:
        trainer.train(args.generations)
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()