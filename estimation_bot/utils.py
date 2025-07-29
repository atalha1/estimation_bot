"""
Utility module for Estimation card game.
Contains logging, formatting, and helper functions.
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from estimation_bot.card import Card, Suit
from estimation_bot.player import Player


def setup_logging(log_file: str = "estimation_game.log", level: int = logging.INFO):
    """Set up logging configuration for the game."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def format_hand(hand: List[Card], sort_by_suit: bool = True) -> str:
    """
    Format a hand of cards for display.
    
    Args:
        hand: List of cards
        sort_by_suit: Whether to sort by suit then rank
        
    Returns:
        Formatted string representation
    """
    if not hand:
        return "Empty hand"
    
    if sort_by_suit:
        # Group by suit
        by_suit = {}
        for card in hand:
            suit_name = card.suit.name
            if suit_name not in by_suit:
                by_suit[suit_name] = []
            by_suit[suit_name].append(card)
        
        # Sort within each suit
        for suit in by_suit:
            by_suit[suit].sort(key=lambda c: c.rank.value)
        
        # Format by suit
        suit_strings = []
        for suit in ['SPADES', 'HEARTS', 'DIAMONDS', 'CLUBS']:
            if suit in by_suit:
                cards_str = ' '.join(str(card) for card in by_suit[suit])
                suit_strings.append(f"{suit[0]}: {cards_str}")
        
        return ' | '.join(suit_strings)
    else:
        # Simple list format
        return ', '.join(str(card) for card in sorted(hand))


def format_scores(players: List[Player]) -> str:
    """Format current scores for display."""
    score_lines = []
    for player in players:
        score_lines.append(f"{player.name}: {player.score}")
    return '\n'.join(score_lines)


def format_bids_and_tricks(players: List[Player]) -> str:
    """Format bids vs actual tricks for display."""
    lines = []
    for player in players:
        bid = player.bid if player.bid is not None else "?"
        lines.append(f"{player.name}: Bid {bid}, Won {player.tricks_won}")
    return '\n'.join(lines)


def log_game_state(game_state: Dict[str, Any], logger: logging.Logger = None):
    """Log current game state."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Game State: {json.dumps(game_state, indent=2)}")


def log_round_result(round_number: int, trump_suit: Suit, 
                    players: List[Player], round_scores: Dict[int, int],
                    logger: logging.Logger = None):
    """Log the results of a completed round."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"\n--- Round {round_number} Complete ---")
    logger.info(f"Trump suit: {trump_suit}")
    
    for i, player in enumerate(players):
        bid = player.bid
        tricks = player.tricks_won
        score = round_scores.get(i, 0)
        logger.info(f"{player.name}: Bid {bid}, Won {tricks}, Score {score:+d}")


def save_game_log(game_data: Dict[str, Any], filename: str = None):
    """Save complete game data to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_log_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(game_data, f, indent=2, default=str)


def load_game_log(filename: str) -> Dict[str, Any]:
    """Load game data from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_hand_strength(hand: List[Card], trump_suit: Suit) -> float:
    """
    Calculate a simple hand strength metric (0-1).
    
    Args:
        hand: Cards in hand
        trump_suit: Current trump suit
        
    Returns:
        Strength score between 0 and 1
    """
    if not hand:
        return 0.0
    
    total_strength = 0.0
    
    for card in hand:
        # Base strength from rank (Ace = 1.0, 2 = 0.0)
        rank_strength = (card.rank.value - 2) / 12
        
        # Trump bonus
        if card.suit == trump_suit:
            rank_strength *= 1.5
        
        total_strength += rank_strength
    
    # Normalize by hand size
    return min(total_strength / len(hand), 1.0)


def get_suit_distribution(hand: List[Card]) -> Dict[str, int]:
    """Get count of cards by suit."""
    distribution = {}
    for card in hand:
        suit_name = card.suit.name
        distribution[suit_name] = distribution.get(suit_name, 0) + 1
    return distribution


def has_high_cards(hand: List[Card], trump_suit: Suit, threshold: int = 11) -> int:
    """Count high cards (J, Q, K, A) in hand."""
    high_cards = 0
    for card in hand:
        if card.rank.value >= threshold:
            high_cards += 1
            # Trump high cards count double
            if card.suit == trump_suit:
                high_cards += 1
    return high_cards


class GameLogger:
    """Enhanced logging class for game events."""
    
    def __init__(self, log_file: str = "estimation_detailed.log"):
        self.logger = logging.getLogger("EstimationGame")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_bid(self, player_name: str, bid: int, hand: List[Card], trump_suit: Suit):
        """Log a player's bid with hand info."""
        hand_str = format_hand(hand)
        self.logger.info(f"{player_name} bids {bid} (Trump: {trump_suit})")
        self.logger.debug(f"{player_name} hand: {hand_str}")
    
    def log_card_play(self, player_name: str, card: Card, trick_state: str):
        """Log a card play."""
        self.logger.info(f"{player_name} plays {card} ({trick_state})")
    
    def log_trick_winner(self, winner_name: str, trick_cards: List[Card]):
        """Log trick winner and cards played."""
        cards_str = ', '.join(str(card) for card in trick_cards)
        self.logger.info(f"{winner_name} wins trick with: {cards_str}")
    
    def log_round_start(self, round_num: int, trump_suit: Suit):
        """Log start of new round."""
        self.logger.info(f"\n=== Round {round_num} - Trump: {trump_suit} ===")
    
    def log_game_start(self, game_mode: str, player_names: List[str]):
        """Log the start of a new game session."""
        self.logger.info(f"\n=== NEW GAME STARTED ===")
        self.logger.info(f"Mode: {game_mode}")
        self.logger.info(f"Players: {', '.join(player_names)}")

    
    def log_game_end(self, winner_name: str, final_scores: Dict[str, int]):
        """Log game completion."""
        self.logger.info(f"\n=== GAME COMPLETE ===")
        self.logger.info(f"Winner: {winner_name}")
        for name, score in final_scores.items():
            self.logger.info(f"{name}: {score} points")