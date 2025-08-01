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
    parser = argparse.ArgumentParser(description="Enhanced Estimation Bot Training")
    parser.add_argument('--generations', type=int, default=20, help='Number of generations')
    parser.add_argument('--games-per-gen', type=int, default=50, help='Games per generation')
    parser.add_argument('--game-mode', choices=['MICRO', 'MINI', 'FULL'], default='MICRO')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--neural', action='store_true', help='Use neural training')
    
    args = parser.parse_args()
    
    print("🎯 Enhanced Estimation Bot Training")
    print("=" * 40)
    
    if args.neural:
        print("Neural training not yet implemented - using self-play")
        # TODO: Implement neural training later
    
    # Run original self-play training (your existing code)
    from training.champion import ChampionManager
    from training.trainer import SelfPlayTrainer
    
    champion_mgr = ChampionManager()
    base_model = champion_mgr.get_champion()
    
    trainer = SelfPlayTrainer(
        base_model=base_model,
        num_workers=args.workers,
        games_per_generation=args.games_per_gen,
        game_mode=args.game_mode
    )
    
    trainer.train(args.generations)


if __name__ == "__main__":
    main()