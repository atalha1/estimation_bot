"""
Deck module for Estimation card game.
Handles deck creation, shuffling, and dealing cards to players.
"""

import random
from typing import List, Dict
from .card import Card, create_deck


class Deck:
    """Manages a deck of cards with shuffling and dealing capabilities."""
    
    def __init__(self):
        self.cards = create_deck()
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck randomly."""
        random.shuffle(self.cards)
    
    def deal_hand(self, size: int) -> List[Card]:
        """
        Deal a hand of specified size.
        
        Args:
            size: Number of cards to deal
            
        Returns:
            List of cards dealt
            
        Raises:
            ValueError: If not enough cards remaining
        """
        if len(self.cards) < size:
            raise ValueError(f"Not enough cards in deck. Need {size}, have {len(self.cards)}")
        
        hand = []
        for _ in range(size):
            hand.append(self.cards.pop())
        return hand
    
    def deal_round(self, num_players: int = 4) -> Dict[int, List[Card]]:
        """
        Deal cards for a round of Estimation.
        
        Args:
            num_players: Number of players (default 4)
            
        Returns:
            Dictionary mapping player_id to their hand
        """
        cards_per_player = len(self.cards) // num_players
        hands = {}
        
        for player_id in range(num_players):
            hands[player_id] = self.deal_hand(cards_per_player)
            
        return hands
    
    def cards_remaining(self) -> int:
        """Return number of cards remaining in deck."""
        return len(self.cards)
    
    def is_empty(self) -> bool:
        """Check if deck is empty."""
        return len(self.cards) == 0
    
    def reset(self):
        """Reset deck to full 52 cards and shuffle."""
        self.cards = create_deck()
        self.shuffle()


def sort_hand(hand: List[Card]) -> List[Card]:
    """
    Sort a hand by suit and rank for display purposes.
    
    Args:
        hand: List of cards to sort
        
    Returns:
        Sorted list of cards
    """
    return sorted(hand, key=lambda card: (card.suit.name, card.rank.value))


def get_cards_by_suit(hand: List[Card]) -> Dict[str, List[Card]]:
    """
    Group cards in hand by suit.
    
    Args:
        hand: List of cards
        
    Returns:
        Dictionary mapping suit names to lists of cards
    """
    by_suit = {}
    for card in hand:
        suit_name = card.suit.name
        if suit_name not in by_suit:
            by_suit[suit_name] = []
        by_suit[suit_name].append(card)
    
    # Sort cards within each suit
    for suit in by_suit:
        by_suit[suit] = sorted(by_suit[suit], key=lambda c: c.rank.value)
    
    return by_suit