"""
Random bot implementation for Estimation card game.
Provides a baseline bot that makes random legal moves while respecting all game rules.
"""

import random
from typing import List, Optional, Tuple
from estimation_bot.card import Card, Suit, Rank
from estimation_bot.player import BotInterface


class RandomBot(BotInterface):
    """Bot that makes completely random legal moves while respecting game rules."""

    def __init__(self, name: str = "RandomBot"):
        self.name = name

    def make_bid(self, hand: List[Card], other_bids: List[Optional[tuple]], 
                 is_speed_round: bool = False) -> Optional[Tuple[int, Optional[Suit]]]:
        """
        Make a random bid following Estimation rules.
        
        Args:
            hand: Current hand of cards
            other_bids: List of (amount, trump_suit) tuples from other players
            is_speed_round: Whether this is a speed round (no bidding allowed)
            
        Returns:
            Tuple of (bid_amount, trump_suit) or None for pass
        """
        if is_speed_round:
            # Speed rounds have predetermined trump, no bidding
            return None
            
        # Filter out None bids to get actual bids
        valid_bids = [bid for bid in other_bids if bid is not None]
        
        # Determine minimum bid required
        min_bid = 4  # Base minimum bid
        if valid_bids:
            highest_bid = max(bid[0] for bid in valid_bids)
            min_bid = max(4, highest_bid + 1)
        
        # Can't bid more than hand size
        max_possible_bid = len(hand)
        
        # If minimum required bid exceeds hand size, must pass
        if min_bid > max_possible_bid:
            return None
            
        # Random decision to bid or pass (30% chance to pass if possible)
        if random.random() < 0.3 and len(valid_bids) > 0:
            return None
            
        # Choose random bid amount between min_bid and hand size
        bid_amount = random.randint(min_bid, max_possible_bid)
        
        # Choose random trump suit (including None for No Trump)
        trump_choices = list(Suit) + [None]  # None represents No Trump
        trump_suit = random.choice(trump_choices)
        
        return (bid_amount, trump_suit)

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
            is_last_estimator: Whether this player is last to estimate (Risk rule)
            
        Returns:
            Estimation (0-13)
        """
        # Basic heuristic for estimation
        high_ranks = {Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK}
        estimated_tricks = 0.0

        for card in hand:
            if trump_suit and card.suit == trump_suit:
                # Trump cards are more valuable
                if card.rank in high_ranks:
                    estimated_tricks += 1.0
                elif card.rank.value >= 10:
                    estimated_tricks += 0.7
                else:
                    estimated_tricks += 0.4
            else:
                # Non-trump cards
                if card.rank == Rank.ACE:
                    estimated_tricks += 0.8
                elif card.rank == Rank.KING:
                    estimated_tricks += 0.5
                elif card.rank == Rank.QUEEN:
                    estimated_tricks += 0.3
                elif card.rank == Rank.JACK:
                    estimated_tricks += 0.2

        # Convert to integer estimate
        estimate = max(0, min(13, round(estimated_tricks)))
        
        # Cannot estimate more than declarer's bid
        estimate = min(estimate, declarer_bid)
        
        # Handle Risk rule - last estimator cannot make total equal exactly 13
        if is_last_estimator:
            total_so_far = sum(e for e in other_estimations if e is not None)
            if total_so_far + estimate == 13:
                # Adjust estimate to avoid Risk
                if estimate > 0:
                    estimate -= 1
                else:
                    estimate += 1
                # Still can't exceed declarer's bid
                estimate = min(estimate, declarer_bid)

        return max(0, estimate)

    def choose_card(self, hand: List[Card], valid_plays: List[Card],
                    trump_suit: Optional[Suit], led_suit: Optional[Suit],
                    trick_cards: List[Card]) -> Card:
        """
        Choose which card to play from valid options.
        
        Args:
            hand: Current hand
            valid_plays: Cards that can legally be played
            trump_suit: Trump suit for this round
            led_suit: Suit led in current trick (None if leading)
            trick_cards: Cards already played in current trick
            
        Returns:
            Card to play
        """
        if not valid_plays:
            raise ValueError("No valid plays available")

        # Simply choose randomly from valid plays
        return random.choice(valid_plays)

    def __str__(self):
        return self.name