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
    
    def make_bid_interactive(self, bid_history: List, can_dash: bool = False) -> Optional[Union[Tuple[int, Optional[Suit]], str]]:
        """Get bid from human player with proper trump hierarchy."""
        print(f"\n{self.name}'s turn to bid")
        print(f"Your hand: {self._format_hand()}")
        
        # Show bid history
        if bid_history:
            print("Previous bids:")
            for player_id, amount, trump_suit in bid_history:
                trump_name = trump_suit.name if trump_suit else "No Trump"
                print(f"  Player {player_id}: {amount} {trump_name}")
        
        # Determine current minimum bid
        if bid_history:
            # Find actual highest bid by comparing all bids
            highest_bid_entry = None
            highest_value = 0
            suit_ranks = {Suit.CLUBS: 1, Suit.DIAMONDS: 2, Suit.HEARTS: 3, Suit.SPADES: 4, None: 5}
            
            for player_id, amount, trump_suit in bid_history:
                trump_rank = suit_ranks.get(trump_suit, 0)
                if amount > highest_value or (amount == highest_value and trump_rank > suit_ranks.get(highest_bid_entry[2] if highest_bid_entry else None, 0)):
                    highest_value = amount
                    highest_bid_entry = (player_id, amount, trump_suit)
            
            if highest_bid_entry:
                min_bid = highest_bid_entry[1]
                current_trump = highest_bid_entry[2]
                print(f"Current high bid: {min_bid} {current_trump.name if current_trump else 'No Trump'}")
            else:
                min_bid = 4
                current_trump = None
                print("No bids yet. Minimum bid: 4")
        else:
            min_bid = 4
            current_trump = None
            print("No bids yet. Minimum bid: 4")
        
        print("\nOptions: 1. Make bid  2. Pass")
        
        while True:
            try:
                choice = input("Enter choice (1/2): ").strip()
                
                if choice == "2":
                    return None  # Pass
                
                elif choice == "1":
                    # Get bid amount
                    while True:
                        try:
                            amount = int(input(f"Enter bid amount ({min_bid}-13): "))
                            if amount < min_bid or amount > 13:
                                print(f"Must bid at least {min_bid}")
                                continue
                            break
                        except ValueError:
                            print("Please enter a valid number")
                    
                    # Get trump suit
                    print("\nChoose trump suit:")
                    print("1. Clubs ♣ (weakest)")
                    print("2. Diamonds ♦")
                    print("3. Hearts ♥")
                    print("4. Spades ♠") 
                    print("5. No Trump (strongest)")
                    
                    # Show which suits are valid for current bid amount
                    # Validate trump hierarchy
                    if amount == min_bid and current_trump is not None:
                        suit_ranks = {Suit.CLUBS: 1, Suit.DIAMONDS: 2, Suit.HEARTS: 3, Suit.SPADES: 4, None: 5}
                        current_rank = suit_ranks.get(current_trump, 0)
                        new_rank = suit_ranks.get(trump_suit, 0)
                        if new_rank < current_rank:
                            print(f"Trump suit must be equal or stronger than current ({current_trump.name if current_trump else 'No Trump'})")
                            continue
                    
                    while True:
                        try:
                            trump_choice = int(input("Enter trump choice (1-5): "))
                            trump_map = {1: Suit.CLUBS, 2: Suit.DIAMONDS, 3: Suit.HEARTS, 4: Suit.SPADES, 5: None}
                            
                            if trump_choice not in trump_map:
                                print("Please enter 1-5")
                                continue
                            
                            trump_suit = trump_map[trump_choice]
                            
                            # Validate trump hierarchy
                            if amount == min_bid and current_trump is not None:
                                suit_ranks = {Suit.CLUBS: 1, Suit.DIAMONDS: 2, Suit.HEARTS: 3, Suit.SPADES: 4, None: 5}
                                if suit_ranks[trump_suit] < suit_ranks[current_trump]:
                                    print("Trump suit must be equal or stronger than current")
                                    continue
                            
                            return (amount, trump_suit)
                            
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
                                        can_dash: bool = False, is_with: bool = False, 
                                        min_with_bid: int = 0) -> Union[int, str]:
        """Get estimation with WITH constraints and proper risk handling."""
        print(f"\n{self.name}'s turn to estimate")
        print(f"Trump suit: {trump_suit.name if trump_suit else 'No Trump'}")
        print(f"Declarer bid: {declarer_bid}")
        print(f"Your hand: {self._format_hand()}")
        
        # Show other estimations
        if other_estimations:
            current_total = sum(other_estimations)
            print(f"Other estimations so far: {other_estimations} (Total: {current_total})")
        
        # Show constraints
        min_est = min_with_bid if is_with else 0
        max_est = declarer_bid
        
        print(f"\nEstimation range: {min_est}-{max_est}")
        
        if is_with:
            print(f"⚠️  You are WITH - minimum estimate: {min_with_bid}")
        
        if is_last_estimator:
            current_total = sum(other_estimations)
            forbidden = 13 - current_total
            if 0 <= forbidden <= max_est:
                print(f"⚠️  You are at RISK! Cannot estimate {forbidden} (total would be 13)")
        
        while True:
            try:
                estimation_str = input(f"Enter estimation ({min_est}-{max_est}): ").strip()
                estimation = int(estimation_str)
                
                if estimation < min_est or estimation > max_est:
                    print(f"Estimation must be between {min_est} and {max_est}")
                    continue
                
                # Check risk constraint
                if is_last_estimator:
                    current_total = sum(other_estimations)
                    if current_total + estimation == 13:
                        print(f"Cannot estimate {estimation} - total would be exactly 13 (Risk rule)")
                        continue
                
                return estimation
                
            except ValueError:
                print("Please enter a valid number")
    
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
                choice = input("Choose a card to play: ").strip()
                
                # Try multiple matching strategies
                for card in valid_plays:
                    card_str = str(card)
                    # Direct match
                    if card_str == choice:
                        return card
                    # Case insensitive match
                    if card_str.lower() == choice.lower():
                        return card
                    # Match just the rank if suits match context
                    if str(card.rank) == choice and led_suit and card.suit == led_suit:
                        return card
                    # Match rank + suit symbol
                    rank_str = str(card.rank)
                    suit_str = str(card.suit)
                    if f"{rank_str}{suit_str}" == choice:
                        return card
                    # Try without suit
                    if rank_str == choice and len([c for c in valid_plays if str(c.rank) == choice]) == 1:
                        return card
                
                # If no match, show what we're looking for
                print(f"'{choice}' is not a valid card.")
                print(f"Try typing exactly: {', '.join(str(card) for card in valid_plays)}")
                print("Or just the rank if unique (e.g., 'A', '10', 'K')")
                
            except (ValueError, KeyboardInterrupt):
                print("Please enter a valid card")  
                
                    
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