"""
Player module for Estimation card game.
Defines player state and strategy interface with proper rule enforcement.
"""

from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from estimation_bot.card import Card, Suit


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
        self.hand = cards.copy()  # Make a copy to avoid reference issues
        
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
            raise ValueError(f"Card {card} not in hand {self.hand}")
        self.hand.remove(card)
        return card
    
    def has_suit(self, suit: Suit) -> bool:
        """Check if player has any cards of given suit."""
        return any(card.suit == suit for card in self.hand)
    
    def get_valid_plays(self, led_suit: Optional[Suit]) -> List[Card]:
        """
        Get cards that can legally be played following Estimation rules.
        
        Args:
            led_suit: Suit that was led (None if leading)
            
        Returns:
            List of valid cards to play
        """
        if led_suit is None:
            # Leading - can play any card
            return self.hand.copy()
        
        # Must follow suit if possible
        following_suit = [card for card in self.hand if card.suit == led_suit]
        if following_suit:
            return following_suit
        
        # Cannot follow suit - can play any remaining card
        return self.hand.copy()
    
    def reset_round(self):
        """Reset player state for new round."""
        self.hand = []
        self.bid = None
        self.estimation = None
        self.tricks_won = 0
        self.is_declarer = False
        self.is_with = False
        self.is_dash = False
        self.has_avoid = False
    
    def add_score(self, points: int):
        """Add points to player's total score."""
        self.score += points
    
    def __str__(self):
        return f"{self.name} (Score: {self.score})"


class BotInterface(ABC):
    """Abstract interface that all bots must implement."""
    
    @abstractmethod
    def make_bid(self, hand: List[Card], other_bids: List[Optional[tuple]], 
                 is_speed_round: bool = False) -> Optional[Tuple[int, Optional[Suit]]]:
        """
        Make a bid for the round.
        
        Args:
            hand: Current hand of cards
            other_bids: List of (amount, trump_suit) tuples from other players
            is_speed_round: Whether this is a speed round (no bidding)
            
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
                   trump_suit: Optional[Suit], led_suit: Optional[Suit],
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
    
    def make_bid_interactive(self, other_bids: List[Optional[tuple]]) -> Optional[Tuple[int, Optional[Suit]]]:
        """Get bid from human player via console input."""
        print(f"\n{self.name}'s turn to bid")
        print(f"Your hand: {self._format_hand()}")
        print(f"Other bids: {self._format_other_bids(other_bids)}")
        
        # Determine minimum bid
        valid_bids = [bid for bid in other_bids if bid is not None]
        min_bid = 4
        if valid_bids:
            highest_bid = max(bid[0] for bid in valid_bids)
            min_bid = max(4, highest_bid + 1)
        
        print(f"Minimum bid: {min_bid}, Maximum: {len(self.hand)}")
        
        while True:
            try:
                choice = input("Enter bid amount (or 'p' to pass): ").strip().lower()
                if choice == 'p':
                    return None
                
                bid_amount = int(choice)
                if bid_amount < min_bid or bid_amount > len(self.hand):
                    print(f"Bid must be between {min_bid} and {len(self.hand)}")
                    continue
                
                # Choose trump suit
                print("Choose trump suit:")
                print("1. Spades ♠")
                print("2. Hearts ♥") 
                print("3. Diamonds ♦")
                print("4. Clubs ♣")
                print("5. No Trump")
                
                trump_choice = input("Enter choice (1-5): ").strip()
                trump_map = {
                    '1': Suit.SPADES, '2': Suit.HEARTS, '3': Suit.DIAMONDS,
                    '4': Suit.CLUBS, '5': None
                }
                
                if trump_choice not in trump_map:
                    print("Invalid trump choice")
                    continue
                
                return (bid_amount, trump_map[trump_choice])
                
            except ValueError:
                print("Please enter a valid number or 'p' to pass")
    
    def make_estimation_interactive(self, trump_suit: Optional[Suit], declarer_bid: int,
                                  other_estimations: List[Optional[int]], 
                                  is_last_estimator: bool = False) -> int:
        """Get estimation from human player."""
        print(f"\n{self.name}'s turn to estimate")
        print(f"Trump: {trump_suit or 'No Trump'}")
        print(f"Declarer bid: {declarer_bid}")
        print(f"Your hand: {self._format_hand()}")
        print(f"Other estimations: {other_estimations}")
        
        if is_last_estimator:
            total_so_far = sum(e for e in other_estimations if e is not None)
            print(f"WARNING: You are last estimator. Total so far: {total_so_far}")
            print(f"Cannot estimate {13 - total_so_far} (would make total = 13, Risk rule)")
        
        while True:
            try:
                estimation = int(input(f"Enter estimation (0-{declarer_bid}): "))
                if estimation < 0 or estimation > declarer_bid:
                    print(f"Estimation must be between 0 and {declarer_bid}")
                    continue
                
                if is_last_estimator:
                    total_so_far = sum(e for e in other_estimations if e is not None)
                    if total_so_far + estimation == 13:
                        print("Cannot make total equal 13 (Risk rule)")
                        continue
                
                return estimation
                
            except ValueError:
                print("Please enter a valid number")
    
    def choose_card_interactive(self, valid_plays: List[Card], 
                               trump_suit: Optional[Suit], led_suit: Optional[Suit]) -> Card:
        """Get card choice from human player via console input."""
        print(f"\n{self.name}'s turn to play")
        print(f"Trump: {trump_suit or 'No Trump'}, Led: {led_suit or 'Leading'}")
        print(f"Your hand: {self._format_hand()}")
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
    
    def _format_hand(self) -> str:
        """Format hand for display."""
        if not self.hand:
            return "Empty"
        
        # Group by suit
        by_suit = {}
        for card in self.hand:
            suit = card.suit
            if suit not in by_suit:
                by_suit[suit] = []
            by_suit[suit].append(card)
        
        # Sort within each suit
        for suit in by_suit:
            by_suit[suit].sort(key=lambda c: c.rank.value)
        
        # Format by suit
        suit_strings = []
        for suit in [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
            if suit in by_suit:
                cards_str = ' '.join(str(card) for card in by_suit[suit])
                suit_strings.append(f"{suit.name[0]}: {cards_str}")
        
        return ' | '.join(suit_strings)
    
    def _format_other_bids(self, other_bids: List[Optional[tuple]]) -> str:
        """Format other players' bids for display."""
        formatted = []
        for i, bid in enumerate(other_bids):
            if bid is None:
                formatted.append(f"P{i}: Pass")
            else:
                amount, trump = bid
                trump_str = trump.name if trump else "No Trump"
                formatted.append(f"P{i}: {amount} {trump_str}")
        return ", ".join(formatted)