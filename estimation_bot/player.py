"""
Player module for Estimation card game.
Defines player state and strategy interface.
"""

from typing import List, Optional
from abc import ABC, abstractmethod
from .card import Card, Suit


class Player:
    """Represents a player in the Estimation game."""
    
    def __init__(self, player_id: int, name: str = None):
        self.player_id = player_id
        self.name = name or f"Player {player_id}"
        self.hand: List[Card] = []
        self.bid: Optional[int] = None
        self.trump_suit: Optional[Suit] = None
        self.estimation: Optional[int] = None
        self.tricks_won = 0
        self.score = 0
        self.is_declarer = False
        self.is_with = False
        self.is_dash = False
        self.has_avoid = False  # No cards of one or more suits
        
    def declare_avoid(self) -> List[Suit]:
        """Check and declare missing suits."""
        suits_in_hand = {card.suit for card in self.hand}
        missing_suits = [suit for suit in Suit if suit not in suits_in_hand]
        self.has_avoid = len(missing_suits) > 0
        return missing_suits
        
    def receive_cards(self, cards: List[Card]):
        """Add cards to player's hand."""
        self.hand = cards
        
    def play_card(self, card: Card) -> Card:
        """
        Remove and return a card from hand.
        
        Args:
            card: Card to play
            
        Returns:
            The played card
            
        Raises:
            ValueError: If card not in hand
        """
        if card not in self.hand:
            raise ValueError(f"Card {card} not in hand")
        self.hand.remove(card)
        return card
    
    def has_suit(self, suit: Suit) -> bool:
        """Check if player has any cards of given suit."""
        return any(card.suit == suit for card in self.hand)
    
    def get_valid_plays(self, led_suit: Optional[Suit]) -> List[Card]:
        """
        Get cards that can legally be played.
        
        Args:
            led_suit: Suit that was led (None if leading)
            
        Returns:
            List of valid cards to play
        """
        if led_suit is None or not self.has_suit(led_suit):
            return self.hand.copy()
        
        # Must follow suit if possible
        return [card for card in self.hand if card.suit == led_suit]
    
    def reset_round(self):
        """Reset player state for new round."""
        self.hand = []
        self.bid = None
        self.tricks_won = 0
    
    def add_score(self, points: int):
        """Add points to player's total score."""
        self.score += points
    
    def __str__(self):
        return f"{self.name} (Score: {self.score})"


class BotInterface(ABC):
    """Abstract interface that all bots must implement."""
    
    @abstractmethod
    def make_bid(self, hand: List[Card], other_bids: List[Optional[tuple]], 
                 is_speed_round: bool = False) -> tuple:
        """
        Make a bid for the round.
        
        Args:
            hand: Current hand of cards
            other_bids: List of (amount, trump_suit) tuples from other players
            is_speed_round: Whether this is a speed round
            
        Returns:
            Tuple of (bid_amount, trump_suit) or None for pass
        """
        pass
    
    @abstractmethod
    def make_estimation(self, hand: List[Card], trump_suit: Optional[Suit],
                       declarer_bid: int, other_estimations: List[Optional[int]],
                       is_last_estimator: bool = False) -> int:
        """
        Make estimation after trump is determined.
        
        Args:
            hand: Current hand
            trump_suit: Determined trump suit (None for No Trump)
            declarer_bid: Declarer's winning bid amount
            other_estimations: Estimations from other players
            is_last_estimator: Whether this player is last to estimate (Risk)
            
        Returns:
            Estimation (0-13)
        """
        pass
    
    @abstractmethod
    def choose_card(self, hand: List[Card], valid_plays: List[Card],
                   trump_suit: Suit, led_suit: Optional[Suit],
                   trick_cards: List[Card]) -> Card:
        """
        Choose which card to play.
        
        Args:
            hand: Current hand
            valid_plays: Cards that can legally be played
            trump_suit: Trump suit for this round
            led_suit: Suit led in current trick (None if leading)
            trick_cards: Cards already played in current trick
            
        Returns:
            Card to play
        """
        pass


class HumanPlayer(Player):
    """Human player that gets input from console."""
    
    def make_bid_interactive(self, trump_suit: Suit, other_bids: List[Optional[int]]) -> int:
        """Get bid from human player via console input."""
        print(f"\n{self.name}'s turn to bid")
        print(f"Trump suit: {trump_suit}")
        print(f"Your hand: {', '.join(str(card) for card in sorted(self.hand))}")
        print(f"Other bids: {other_bids}")
        
        while True:
            try:
                bid = int(input(f"Enter your bid (0-{len(self.hand)}): "))
                if 0 <= bid <= len(self.hand):
                    return bid
                print(f"Bid must be between 0 and {len(self.hand)}")
            except ValueError:
                print("Please enter a valid number")
    
    def choose_card_interactive(self, valid_plays: List[Card], 
                               trump_suit: Suit, led_suit: Optional[Suit]) -> Card:
        """Get card choice from human player via console input."""
        print(f"\n{self.name}'s turn to play")
        print(f"Trump: {trump_suit}, Led: {led_suit or 'Leading'}")
        print(f"Valid plays: {', '.join(str(card) for card in valid_plays)}")
        
        while True:
            try:
                choice = input("Enter card to play (e.g., 'A♠' or '10♦'): ").strip()
                for card in valid_plays:
                    if str(card) == choice:
                        return card
                print("Invalid card. Please choose from valid plays.")
            except (ValueError, KeyboardInterrupt):
                print("Please enter a valid card")