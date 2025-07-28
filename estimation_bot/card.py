"""
Card module for Estimation card game.
Defines Card, Suit, and Rank classes with proper ordering and comparison.
"""

from enum import Enum
from typing import List


class Suit(Enum):
    """Card suits with proper ordering for trick resolution."""
    CLUBS = "♣"
    DIAMONDS = "♦"
    HEARTS = "♥"
    SPADES = "♠"
    
    def __str__(self):
        return self.value


class Rank(Enum):
    """Card ranks with proper ordering (2 lowest, Ace highest)."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    
    def __str__(self):
        if self.value <= 10:
            return str(self.value)
        return {11: "J", 12: "Q", 13: "K", 14: "A"}[self.value]
    
    def __lt__(self, other):
        return self.value < other.value


class Card:
    """Represents a playing card with suit and rank."""
    
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        return f"Card({self.suit.name}, {self.rank.name})"
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank
    
    def __hash__(self):
        return hash((self.suit, self.rank))
    
    def __lt__(self, other):
        """Compare cards by rank only (suit comparison handled by game logic)."""
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank < other.rank
    
    def beats(self, other: 'Card', trump_suit: Suit, led_suit: Suit) -> bool:
        """
        Determine if this card beats another in a trick.
        
        Args:
            other: The card to compare against
            trump_suit: Current trump suit
            led_suit: Suit that was led this trick
            
        Returns:
            True if this card beats the other card
        """
        # Trump beats non-trump
        if self.suit == trump_suit and other.suit != trump_suit:
            return True
        if other.suit == trump_suit and self.suit != trump_suit:
            return False
            
        # Both trump or both non-trump: higher rank wins if same suit
        if self.suit == other.suit:
            return self.rank > other.rank
            
        # Different suits, neither trump: led suit wins
        if self.suit == led_suit:
            return True
        if other.suit == led_suit:
            return False
            
        # Neither follows led suit, neither trump: first played wins
        return False


def create_deck() -> List[Card]:
    """Create a standard 52-card deck."""
    deck = []
    for suit in Suit:
        for rank in Rank:
            deck.append(Card(suit, rank))
    return deck