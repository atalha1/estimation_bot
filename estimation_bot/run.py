#!/usr/bin/env python3
"""
Main entry point for Estimation card game.
Run a complete game with bots or human players.
"""

import argparse
from typing import List
from estimation_bot.player import Player, HumanPlayer
from estimation_bot.game import EstimationGame
from estimation_bot.utils import setup_logging, GameLogger
from bot.random_bot import RandomBot, WeightedRandomBot
from bot.heuristic_bot import HeuristicBot, AdvancedHeuristicBot


def create_bot_player(bot_type: str, player_id: int) -> Player:
    """Create a bot player of specified type."""
    bot_map = {
        'random': RandomBot,
        'weighted': WeightedRandomBot,
        'heuristic': HeuristicBot,
        'advanced': AdvancedHeuristicBot
    }
    
    if bot_type not in bot_map:
        raise ValueError(f"Unknown bot type: {bot_type}. Available: {list(bot_map.keys())}")
    
    player = Player(player_id, f"{bot_type.title()}Bot_{player_id}")
    player.strategy = bot_map[bot_type](f"{bot_type.title()}Bot_{player_id}")
    return player


def create_human_player(player_id: int, name: str = None) -> HumanPlayer:
    """Create a human player."""
    return HumanPlayer(player_id, name or f"Human_{player_id}")


def run_bot_game(bot_types: List[str], game_mode: str = "FULL") -> dict:
    """Run a game with only bots."""
    players = []
    for i, bot_type in enumerate(bot_types):
        players.append(create_bot_player(bot_type, i))
    
    game = EstimationGame(players, game_mode)
    
    print(f"🎴 Starting {game_mode} Bola with bots: {', '.join(bot_types)} 🎴")
    
    try:
        final_scores = game.play_game()
        winner_id, winning_score = game.get_winner()
        
        return {
            'winner': players[winner_id].name,
            'scores': {p.name: p.score for p in players},
            'rounds_played': game.current_round
        }
    
    except Exception as e:
        print(f"Game error: {e}")
        raise


def run_mixed_game(human_count: int, bot_types: List[str], game_mode: str = "FULL") -> dict:
    """Run a game with humans and bots."""
    players = []
    
    # Add human players
    for i in range(human_count):
        name = input(f"Enter name for Player {i+1}: ").strip() or f"Player_{i+1}"
        players.append(create_human_player(i, name))
    
    # Add bot players
    for i, bot_type in enumerate(bot_types):
        players.append(create_bot_player(bot_type, human_count + i))
    
    if len(players) != 4:
        raise ValueError("Need exactly 4 players total")
    
    game = EstimationGame(players, game_mode)
    
    print(f"\n🎴 Starting {game_mode} Bola with {game.total_rounds} rounds 🎴")
    print(f"Players: {', '.join(p.name for p in players)}")
    
    try:
        final_scores = game.play_game()
        winner_id, winning_score = game.get_winner()
        
        return {
            'winner': players[winner_id].name,
            'scores': {p.name: p.score for p in players},
            'rounds_played': game.current_round
        }
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        return {'interrupted': True}
    except Exception as e:
        print(f"Game error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Play Estimation card game")
    parser.add_argument('--mode', choices=['FULL', 'MINI', 'MICRO'], 
                       default='FULL', help='Game mode (Bola type)')
    parser.add_argument('--bots', nargs='*', 
                       choices=['random', 'weighted', 'heuristic', 'advanced'],
                       help='Bot types for remaining players')
    parser.add_argument('--humans', type=int, default=1,
                       help='Number of human players (0-4)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    # Handle positional arguments for bot types (for backward compatibility)
    # This allows: python -m estimation_bot.run --mode MICRO --humans 1 advanced weighted
    parser.add_argument('extra_bots', nargs='*', 
                       choices=['random', 'weighted', 'heuristic', 'advanced'],
                       help='Additional bot types')
    
    args = parser.parse_args()
    
    # Combine --bots and positional bot arguments
    if args.extra_bots:
        if args.bots:
            args.bots.extend(args.extra_bots)
        else:
            args.bots = args.extra_bots
    
    # Setup logging
    if args.verbose:
        setup_logging(level=10)  # DEBUG
    else:
        setup_logging()
    
    print("🎴 Welcome to Estimation! 🎴")
    
    # Calculate total players needed
    total_needed = 4 - args.humans
    
    # If no bots specified, fill with random bots
    if not args.bots:
        args.bots = ['random'] * total_needed
    elif len(args.bots) < total_needed:
        # Fill remaining slots with random bots
        args.bots.extend(['random'] * (total_needed - len(args.bots)))
    elif len(args.bots) > total_needed:
        # Truncate excess bots
        args.bots = args.bots[:total_needed]
    
    if args.humans == 0:
        # Bot-only game
        result = run_bot_game(args.bots, args.mode)
        
    else:
        # Mixed game
        result = run_mixed_game(args.humans, args.bots, args.mode)
    
    if not result.get('interrupted'):
        print(f"\n✅ Game completed successfully!")


if __name__ == "__main__":
    main()