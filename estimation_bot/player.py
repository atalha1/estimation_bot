"""
Player module for Estimation card game.
Defines player state and strategy interface.
"""

from typing import List, Optional, Union, Tuple
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
        self.hand = cards

    def make_dash_choice(self) -> bool:
    """Ask human player if they want to make a dash call."""
    print(f"\n{self.name}, do you want to make a DASH CALL (0 tricks)?")
    print("This is decided BEFORE seeing other bids - it's risky!")
    
    while True:
        try:
            choice = input("Make DASH CALL? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            print("Please enter 'y' or 'n'")
        except (ValueError, KeyboardInterrupt):
            print("Please enter 'y' or 'n'")
        
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
        self.is_declarer = False
        self.is_with = False
        self.is_dash = False
        self.estimation = None
    
    def add_score(self, points: int):
        """Add points to player's total score."""
        self.score += points
    
    def __str__(self):
        return f"{self.name} (Score: {self.score})"


class BotInterface(ABC):
    """Abstract interface that all bots must implement."""
    
    @abstractmethod
    def make_bid(self, hand: List[Card], other_bids: List[Optional[Union[Tuple[int, Optional[Suit]], str]]], 
                 is_speed_round: bool = False, can_dash: bool = True) -> Optional[Union[Tuple[int, Optional[Suit]], str]]:
        """
        Make a bid for the round.
        
        Args:
            hand: Current hand of cards
            other_bids: List of bids from other players (None, (amount, trump_suit), or "DASH")
            is_speed_round: Whether this is a speed round
            can_dash: Whether dash calls are still available
            
        Returns:
            Tuple of (bid_amount, trump_suit), "DASH", or None for pass
        """
        pass
    
    @abstractmethod
    def make_estimation(self, hand: List[Card], trump_suit: Optional[Suit],
                       declarer_bid: int, other_estimations: List[Optional[int]],
                       is_last_estimator: bool = False, can_dash: bool = True) -> Union[int, str]:
        """
        Make estimation after trump is determined.
        
        Args:
            hand: Current hand
            trump_suit: Determined trump suit (None for No Trump)
            declarer_bid: Declarer's winning bid amount
            other_estimations: Estimations from other players
            is_last_estimator: Whether this player is last to estimate (Risk)
            can_dash: Whether dash estimations are still available
            
        Returns:
            Estimation (0-13), "DASH", or declarer_bid for "WITH"
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
    
    def make_bid_interactive(self, other_bids: List[Optional[Union[Tuple[int, Optional[Suit]], str]]], can_dash: bool = True) -> Optional[Union[Tuple[int, Optional[Suit]], str]]:
        """Get bid from human player via console input."""
        print(f"\n{self.name}'s turn to bid")
        print(f"Your hand: {self._format_hand()}")
        
        # Show other bids
        bid_summary = []
        for i, bid in enumerate(other_bids):
            if bid is None:
                bid_summary.append(f"Player {i}: Pass")
            elif bid == "DASH":
                bid_summary.append(f"Player {i}: DASH CALL")
            elif isinstance(bid, tuple):
                amount, trump = bid
                trump_str = trump.name if trump else "No Trump"
                bid_summary.append(f"Player {i}: {amount} {trump_str}")
            else:
                bid_summary.append(f"Player {i}: Unknown bid")
                
                if any(b is not None for b in other_bids):
                    print(f"Previous bids: {', '.join(bid_summary)}")
        
        print("\nOptions:")
        print("1. Make a regular bid (4-13 tricks + trump suit)")
        if can_dash:
            print("2. Make a DASH CALL (0 tricks)")
        print("3. Pass")
        
        while True:
            try:
                choice = input("Enter your choice (1/2/3): ").strip()
                
                if choice == "3":
                    return None  # Pass
                
                elif choice == "2" and can_dash:
                    return "DASH"
                
                elif choice == "1":
                    # Regular bid
                    while True:
                        try:
                            amount = int(input("Enter bid amount (4-13): "))
                            if 4 <= amount <= 13:
                                break
                            print("Bid must be between 4 and 13")
                        except ValueError:
                            print("Please enter a valid number")
                    
                    print("Choose trump suit:")
                    print("1. No Trump")
                    print("2. Spades ♠")
                    print("3. Hearts ♥") 
                    print("4. Diamonds ♦")
                    print("5. Clubs ♣")
                    
                    while True:
                        try:
                            trump_choice = int(input("Enter trump choice (1-5): "))
                            trump_map = {
                                1: None,
                                2: Suit.SPADES,
                                3: Suit.HEARTS,
                                4: Suit.DIAMONDS,
                                5: Suit.CLUBS
                            }
                            if trump_choice in trump_map:
                                return (amount, trump_map[trump_choice])
                            print("Please enter 1-5")
                        except ValueError:
                            print("Please enter a valid number")
                
                else:
                    print("Invalid choice")
                    
            except (ValueError, KeyboardInterrupt):
                print("Please enter a valid choice")

    def make_dash_choice(self) -> bool:
        """Ask human player if they want to make a dash call."""
        print(f"\n{self.name}, do you want to make a DASH CALL (0 tricks)?")
        print("This is decided BEFORE seeing other bids - it's risky!")
        
        while True:
            try:
                choice = input("Make DASH CALL? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                print("Please enter 'y' or 'n'")
            except (ValueError, KeyboardInterrupt):
                print("Please enter 'y' or 'n'")

    def make_estimation_interactive(self, trump_suit: Optional[Suit], declarer_bid: int,
                                  other_estimations: List[int], is_last_estimator: bool = False,
                                  can_dash: bool = True) -> Union[int, str]:
        """Get estimation from human player via console input."""
        print(f"\n{self.name}'s turn to estimate")
        print(f"Trump suit: {trump_suit.name if trump_suit else 'No Trump'}")
        print(f"Declarer bid: {declarer_bid}")
        print(f"Your hand: {self._format_hand()}")
        
        # Show other estimations
        if other_estimations:
            current_total = sum(v for v in round_obj.estimations.values() if v is not None) + (estimation if estimation is not None else 0)            
            print(f"Other estimations so far: {other_estimations} (Total: {current_total})")
        
        print(f"\nOptions:")
        if is_last_estimator:
            current_total = sum(other_estimations)
            forbidden = 13 - current_total
            print(f"⚠️  You are the Risk player! Cannot estimate {forbidden} (total would be 13)")

        # Only show WITH option if player can actually go WITH (same bid as declarer)
        print(f"1. Regular estimation (0-{declarer_bid})")
        if can_dash:
            print("2. DASH (estimate 0 tricks)")
    
    def choose_card_interactive(self, valid_plays: List[Card], trump_suit: Optional[Suit], 
                               led_suit: Optional[Suit], cards_on_table: List[str] = None) -> Card:
        """Get card choice from human player via console input."""
        print(f"\n{self.name}'s turn to play")
        print(f"Trump: {trump_suit.name if trump_suit else 'No Trump'}")
        print(f"Led suit: {led_suit.name if led_suit else 'Leading'}")
        
        if cards_on_table:
            print(f"Cards on table: {', '.join(cards_on_table)}")
        
        print(f"Your hand: {self._format_hand()}")
        print(f"Valid plays: {', '.join(str(card) for card in valid_plays)}")
        
        while True:
            try:
                choice = input("Enter your choice: ").strip()
                
                if choice == "2" and can_dash:
                    return "DASH"
                elif choice == "1":
                    while True:
                        try:
                            estimation = int(input(f"Enter estimation (0-{declarer_bid}): "))
                            if 0 <= estimation <= declarer_bid:
                                # Check risk constraint
                                if is_last_estimator:
                                    current_total = sum(other_estimations)
                                    if current_total + estimation == 13:
                                        print(f"Cannot estimate {estimation} - total would be exactly 13 (Risk rule)")
                                        continue
                                return estimation
                            print(f"Estimation must be between 0 and {declarer_bid}")
                        except ValueError:
                            print("Please enter a valid number")
                else:
                    print("Invalid choice")
            except (ValueError, KeyboardInterrupt):
                print("Please enter a valid choice")
                
                    
    def _format_hand(self) -> str:
        """Format hand for display grouped by suit."""
        if not self.hand:
            return "Empty hand"
        
        # Group by suit
        by_suit = {}
        for card in self.hand:
            suit_name = card.suit
            if suit_name not in by_suit:
                by_suit[suit_name] = []
            by_suit[suit_name].append(card)
        
        # Sort within each suit
        for suit in by_suit:
            by_suit[suit].sort(key=lambda c: c.rank.value, reverse=True)  # High to low
        
        # Format by suit
        suit_strings = []
        for suit in [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
            if suit in by_suit:
                cards_str = ' '.join(str(card.rank) for card in by_suit[suit])
                suit_strings.append(f"{suit.value}: {cards_str}")
        
        return ' | '.join(suit_strings)