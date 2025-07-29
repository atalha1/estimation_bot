"""
Rules module for Estimation card game.
Contains game rules, validation logic, and rule constants.
"""

from typing import List, Optional
from estimation_bot.card import Card, Suit


# Game constants
NUM_PLAYERS = 4
CARDS_PER_PLAYER = 13
TOTAL_CARDS = 52

# Game modes (Bolas)
FULL_BOLA_ROUNDS = 18
MINI_BOLA_ROUNDS = 10
MICRO_BOLA_ROUNDS = 5

# Round types
NORMAL_ROUNDS = 13
SPEED_ROUNDS_START = 14

# Bidding constants
MIN_BID = 4
MAX_DASH_CALLS = 2

# Scoring constants
BASE_POINTS = 10
CALL_BONUS = 10
SOLE_WINNER_BONUS = 10
RISK_BONUS = 10
DASH_OVER_POINTS = 25
DASH_UNDER_POINTS = 33

# Suit rankings (high to low): No Trump, Spades, Hearts, Diamonds, Clubs
SUIT_RANKINGS = {
    None: 5,  # No Trump
    Suit.SPADES: 4,
    Suit.HEARTS: 3,
    Suit.DIAMONDS: 2,
    Suit.CLUBS: 1
}


def validate_bid(bid: int, hand_size: int) -> bool:
    """
    Validate that a bid is legal.
    
    Args:
        bid: Proposed bid
        hand_size: Number of cards in hand
        
    Returns:
        True if bid is valid
    """
    return 0 <= bid <= hand_size


def validate_card_play(card: Card, hand: List[Card], led_suit: Optional[Suit]) -> bool:
    """
    Validate that a card play is legal.
    
    Args:
        card: Card being played
        hand: Player's current hand
        led_suit: Suit that was led (None if leading)
        
    Returns:
        True if play is valid
    """
    if card not in hand:
        return False
    
    # If leading, any card is valid
    if led_suit is None:
        return True
    
    # Must follow suit if possible
    has_led_suit = any(c.suit == led_suit for c in hand)
    if has_led_suit:
        return card.suit == led_suit
    
    # If can't follow suit, any card is valid
    return True


def calculate_round_score(bid: int, tricks_won: int) -> int:
    """
    Calculate score for a round based on Estimation rules.
    
    Args:
        bid: Player's bid
        tricks_won: Actual tricks won
        
    Returns:
        Points earned (positive or negative)
    """
    if bid == tricks_won:
        # Made bid exactly: +10 + bid
        return EXACT_BID_BONUS + bid
    else:
        # Missed bid: -10 - difference
        difference = abs(bid - tricks_won)
        return MISSED_BID_PENALTY - difference


def get_trump_rotation() -> List[Suit]:
    """Get the standard trump suit rotation."""
    return [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]


def is_valid_hand_size(hand_size: int, round_number: int) -> bool:
    """
    Check if hand size is correct for round number.
    
    Args:
        hand_size: Current hand size
        round_number: Current round (1-13)
        
    Returns:
        True if hand size is valid
    """
    expected_size = CARDS_PER_PLAYER
    return hand_size <= expected_size


def can_make_bid(hand: List[Card], trump_suit: Suit) -> range:
    """
    Get range of reasonable bids for a hand.
    
    Args:
        hand: Player's cards
        trump_suit: Trump suit for round
        
    Returns:
        Range of valid bid values
    """
    return range(0, len(hand) + 1)


def must_follow_suit(hand: List[Card], led_suit: Suit) -> bool:
    """
    Check if player must follow suit.
    
    Args:
        hand: Player's cards
        led_suit: Suit that was led
        
    Returns:
        True if player has cards of led suit and must follow
    """
    return any(card.suit == led_suit for card in hand)


def get_legal_plays(hand: List[Card], led_suit: Optional[Suit]) -> List[Card]:
    """
    Get all legal card plays for current situation.
    
    Args:
        hand: Player's current hand
        led_suit: Suit led in trick (None if leading)
        
    Returns:
        List of cards that can legally be played
    """
    if led_suit is None:
        # Leading - can play any card
        return hand.copy()
    
    # Must follow suit if possible
    following_suit = [card for card in hand if card.suit == led_suit]
    if following_suit:
        return following_suit
    
    # Can't follow suit - can play any card
    return hand.copy()


def determine_trick_winner(cards_played: List[tuple], trump_suit: Suit, led_suit: Suit) -> int:
    """
    Determine winner of a trick.
    
    Args:
        cards_played: List of (player_id, card) tuples in play order
        trump_suit: Current trump suit
        led_suit: Suit that was led
        
    Returns:
        Player ID of trick winner
    """
    if not cards_played:
        raise ValueError("No cards played")
    
    winning_player_id = cards_played[0][0]
    winning_card = cards_played[0][1]
    
    for player_id, card in cards_played[1:]:
        if card.beats(winning_card, trump_suit, led_suit):
            winning_player_id = player_id
            winning_card = card
    
    return winning_player_id


def validate_game_state(players: List, round_number: int) -> bool:
    """
    Validate current game state is legal.
    
    Args:
        players: List of player objects
        round_number: Current round number
        
    Returns:
        True if game state is valid
    """
    # Check player count
    if len(players) != NUM_PLAYERS:
        return False
    
    # Check round number
    if not (1 <= round_number <= NUM_ROUNDS):
        return False
    
    # Check hand sizes
    for player in players:
        if len(player.hand) > CARDS_PER_PLAYER:
            return False
    
    return True


class GameRules:
    """Container class for game rule constants and methods."""
    
    NUM_PLAYERS = NUM_PLAYERS
    NUM_ROUNDS = NUM_ROUNDS
    CARDS_PER_PLAYER = CARDS_PER_PLAYER
    EXACT_BID_BONUS = EXACT_BID_BONUS
    MISSED_BID_PENALTY = MISSED_BID_PENALTY
    
    @staticmethod
    def score_round(bid: int, tricks_won: int) -> int:
        """Calculate round score."""
        return calculate_round_score(bid, tricks_won)
    
    @staticmethod
    def valid_bid(bid: int, hand_size: int) -> bool:
        """Check if bid is valid."""
        return validate_bid(bid, hand_size)
    
    @staticmethod
    def legal_plays(hand: List[Card], led_suit: Optional[Suit]) -> List[Card]:
        """Get legal plays."""
        return get_legal_plays(hand, led_suit)