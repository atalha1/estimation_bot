#!/usr/bin/env python3
"""
Main entry point for Estimation card game.
Run a complete game with random bots or human players.
"""

import argparse
from typing import List
from estimation_bot.player import Player, HumanPlayer
from estimation_bot.game import EstimationGame
from estimation_bot.utils import setup_logging, GameLogger
from bot.random_bot import RandomBot


def create_bot_player(bot_type: str, player_id: int) -> Player:
    """Create a bot player of specified type."""
    if bot_type != 'random':
        raise ValueError(f"Only 'random' bot type is supported, got: {bot_type}")
    
    player = Player(player_id, f"RandomBot_{player_id}")
    player.strategy = RandomBot(f"RandomBot_{player_id}")
    return player


def create_human_player(player_id: int, name: str = None) -> HumanPlayer:
    """Create a human player."""
    return HumanPlayer(player_id, name or f"Human_{player_id}")


def run_bot_game(bot_types: List[str], game_mode: str = "FULL") -> dict:
    """Run a game with only bots."""
    # Validate all bot types are 'random'
    for bot_type in bot_types:
        if bot_type != 'random':
            raise ValueError(f"Only 'random' bot type is supported, got: {bot_type}")
    
    players = []
    for i, bot_type in enumerate(bot_types):
        players.append(create_bot_player(bot_type, i))
    
    game = EstimationGame(players, game_mode)
    logger = GameLogger()
    
    logger.log_game_start(game_mode, [p.name for p in players])
    
    try:
        final_scores = game.play_game()
        winner_id, winning_score = game.get_winner()
        
        logger.log_game_end(players[winner_id].name, 
                          {p.name: p.score for p in players})
        
        return {
            'winner': players[winner_id].name,
            'scores': {p.name: p.score for p in players},
            'rounds_played': game.current_round
        }
    
    except Exception as e:
        logger.logger.error(f"Game error: {e}")
        raise


def run_mixed_game(human_count: int, bot_types: List[str], game_mode: str = "FULL") -> dict:
    """Run a game with humans and bots."""
    # Validate all bot types are 'random'
    for bot_type in bot_types:
        if bot_type != 'random':
            raise ValueError(f"Only 'random' bot type is supported, got: {bot_type}")
    
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
    logger = GameLogger()
    
    print(f"\nStarting {game_mode} Bola with {game.total_rounds} rounds")
    print(f"Players: {', '.join(p.name for p in players)}")
    
    try:
        final_scores = game.play_game()
        winner_id, winning_score = game.get_winner()
        
        print(f"\n🎉 Game Complete! 🎉")
        print(f"Winner: {players[winner_id].name} with {winning_score} points")
        print("\nFinal Scores:")
        for player in sorted(players, key=lambda p: p.score, reverse=True):
            print(f"  {player.name}: {player.score}")
        
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
    parser.add_argument('--bots', nargs='+', 
                       choices=['random'],
                       default=['random', 'random', 'random', 'random'],
                       help='Bot types for players (only random supported)')
    parser.add_argument('--humans', type=int, default=0,
                       help='Number of human players (0-4)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        setup_logging(level=10)  # DEBUG
    else:
        setup_logging()
    
    print("🎴 Welcome to Estimation! 🎴")
    
    if args.humans == 0:
        # Bot-only game
        if len(args.bots) != 4:
            print("Error: Need exactly 4 bot types for bot-only game")
            return
        
        print(f"Running {args.mode} Bola with bots: {', '.join(args.bots)}")
        result = run_bot_game(args.bots, args.mode)
        
    elif args.humans + len(args.bots) == 4:
        # Mixed game
        result = run_mixed_game(args.humans, args.bots, args.mode)
        
    else:
        print(f"Error: Need exactly 4 players total. "
              f"Got {args.humans} humans + {len(args.bots)} bots")
        return
    
    if not result.get('interrupted'):
        print(f"\nGame completed in {result['rounds_played']} rounds")


if __name__ == "__main__":
    main()