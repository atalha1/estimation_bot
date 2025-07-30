"""
Random bot implementation for Estimation card game.
Provides a baseline bot that makes random legal moves.
"""

import random
from typing import List, Optional, Union, Tuple
from estimation_bot.card import Card, Suit, Rank
from estimation_bot.player import BotInterface


class RandomBot(BotInterface):
    """Bot that makes completely random legal moves."""

    def __init__(self, name: str = "RandomBot"):
        self.name = name

    def make_bid(self, hand: List[Card], other_bids: List[Optional[Union[Tuple[int, Optional[Suit]], str]]], 
                 is_speed_round: bool = False, can_dash: bool = True) -> Optional[Union[Tuple[int, Optional[Suit]], str]]:
        """Random bid with some basic logic."""
        if is_speed_round:
            return None  # No bidding in speed rounds
        
        # Random choice between bidding and passing
        if random.random() < 0.3:  # 30% chance to pass
            return None
        
        # Small chance for dash call
        if can_dash and random.random() < 0.1:  # 10% chance
            return "DASH"
        
        # Regular bid
        amount = random.randint(4, min(10, len(hand)))  # Conservative random bid
        trump_suit = random.choice([None, Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS])
        
        return (amount, trump_suit)

    def make_estimation(self, hand: List[Card], trump_suit: Optional[Suit],
                       declarer_bid: int, other_estimations: List[Optional[int]],
                       is_last_estimator: bool = False, can_dash: bool = True) -> Union[int, str]:
        """
        Random estimation with basic hand analysis.
        """
        # Small chance for special moves
        if can_dash and random.random() < 0.05:  # 5% chance for dash
            return "DASH"
        
        if random.random() < 0.1:  # 10% chance for WITH
            return declarer_bid
        
        # Basic estimation based on hand strength
        high_ranks = {Rank.ACE, Rank.KING, Rank.QUEEN}
        estimated_tricks = 0.0

        for card in hand:
            if trump_suit and card.suit == trump_suit:
                if card.rank in high_ranks:
                    estimated_tricks += 1.0
                else:
                    estimated_tricks += 0.5
            else:
                if card.rank == Rank.ACE:
                    estimated_tricks += 1.0
                elif card.rank == Rank.KING:
                    estimated_tricks += 0.6
                elif card.rank == Rank.QUEEN:
                    estimated_tricks += 0.4
                elif card.rank == Rank.JACK:
                    estimated_tricks += 0.3

        estimate = int(round(estimated_tricks))
        estimate = max(0, min(declarer_bid, estimate))
        
        # Handle risk constraint
        if is_last_estimator:
            current_total = sum(e for e in other_estimations if e is not None)
            if current_total + estimate == 13:
                # Adjust to avoid exact 13
                if estimate > 0:
                    estimate -= 1
                else:
                    estimate += 1
                estimate = max(0, min(declarer_bid, estimate))

        return estimate

    def choose_card(self, hand: List[Card], valid_plays: List[Card],
                    trump_suit: Optional[Suit], led_suit: Optional[Suit],
                    trick_cards: List[Card]) -> Card:
        """Choose a random valid card to play."""
        if not valid_plays:
            raise ValueError("No valid plays available")

        return random.choice(valid_plays)

    def __str__(self):
        return self.name


class WeightedRandomBot(BotInterface):
    """Slightly smarter random bot with basic preferences."""
    
    def __init__(self, name: str = "WeightedRandomBot"):
        self.name = name
    
    def make_bid(self, hand: List[Card], other_bids: List[Optional[Union[Tuple[int, Optional[Suit]], str]]], 
                 is_speed_round: bool = False, can_dash: bool = True) -> Optional[Union[Tuple[int, Optional[Suit]], str]]:
        """
        Make a bid with slight preference for conservative bids.
        """
        if is_speed_round:
            return None  # No bidding in speed rounds
        
        hand_strength = self._estimate_hand_strength(hand, None)  # Rough estimate
        
        # Decide whether to bid based on hand strength
        bid_probability = 0.4 + hand_strength * 0.4  # 40-80% chance based on strength
        
        if random.random() > bid_probability:
            return None  # Pass
        
        # Small chance for dash call with weak hands
        if can_dash and hand_strength < 0.2 and random.random() < 0.15:
            return "DASH"
        
        # Regular bid
        if hand_strength < 0.3:
            amount = random.randint(4, 6)
        elif hand_strength > 0.7:
            amount = random.randint(7, min(10, len(hand)))
        else:
            amount = random.randint(4, 8)
        
        # Choose trump suit - prefer suits we have strength in
        trump_options = [None, Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]
        
        # Weight trump choices based on suit strength
        suit_weights = []
        for trump_opt in trump_options:
            weight = 1.0
            if trump_opt is not None:
                trump_cards = [c for c in hand if c.suit == trump_opt]
                if trump_cards:
                    # Prefer suits we have cards in, especially high cards
                    weight += len(trump_cards) * 0.5
                    weight += sum(1 for c in trump_cards if c.rank.value >= 11) * 0.5
            suit_weights.append(weight)
        
        trump_suit = random.choices(trump_options, weights=suit_weights)[0]
        
        return (amount, trump_suit)
    
    def make_estimation(self, hand: List[Card], trump_suit: Optional[Suit],
                       declarer_bid: int, other_estimations: List[Optional[int]],
                       is_last_estimator: bool = False, can_dash: bool = True) -> Union[int, str]:
        """
        Make estimation with basic hand analysis.
        """
        hand_strength = self._estimate_hand_strength(hand, trump_suit)
        
        # Small chance for dash with very weak hands
        if can_dash and hand_strength < 0.1 and random.random() < 0.1:
            return "DASH"
        
        # Small chance for WITH with strong hands and high declarer bid
        if declarer_bid >= 7 and hand_strength > 0.8 and random.random() < 0.15:
            return declarer_bid
        
        # Estimate based on hand analysis
        high_ranks = {Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK}
        estimated_tricks = 0.0

        for card in hand:
            if trump_suit and card.suit == trump_suit:
                if card.rank in high_ranks:
                    estimated_tricks += 0.9
                else:
                    estimated_tricks += 0.4
            else:
                if card.rank == Rank.ACE:
                    estimated_tricks += 0.8
                elif card.rank == Rank.KING:
                    estimated_tricks += 0.5
                elif card.rank == Rank.QUEEN:
                    estimated_tricks += 0.3
                elif card.rank == Rank.JACK:
                    estimated_tricks += 0.2

        estimate = int(round(estimated_tricks))
        estimate = max(0, min(declarer_bid, estimate))
        
        # Handle risk constraint
        if is_last_estimator:
            current_total = sum(e for e in other_estimations if e is not None)
            if current_total + estimate == 13:
                # Prefer going under rather than over
                if estimate > 1:
                    estimate -= 1
                else:
                    estimate += 1
                estimate = max(0, min(declarer_bid, estimate))

        return estimate
    
    def choose_card(self, hand: List[Card], valid_plays: List[Card],
                   trump_suit: Optional[Suit], led_suit: Optional[Suit],
                   trick_cards: List[Card]) -> Card:
        """
        Choose card with slight strategic preference.
        """
        if not valid_plays:
            raise ValueError("No valid plays available")
        
        if len(valid_plays) == 1:
            return valid_plays[0]
        
        # Calculate weights for each valid play
        weights = []
        for card in valid_plays:
            weight = 1.0
            
            # Prefer high cards when leading
            if led_suit is None:
                weight += (card.rank.value - 7) / 10
            
            # Prefer trump cards when not leading trump
            if trump_suit and card.suit == trump_suit and led_suit != trump_suit:
                weight += 1.5
            
            # Slightly prefer high cards in general
            weight += (card.rank.value - 7) / 20
            
            weights.append(max(weight, 0.1))  # Ensure positive weight
        
        return random.choices(valid_plays, weights=weights)[0]
    
    def _estimate_hand_strength(self, hand: List[Card], trump_suit: Optional[Suit]) -> float:
        """
        Estimate hand strength (0-1).
        
        Args:
            hand: Cards in hand
            trump_suit: Current trump suit
            
        Returns:
            Strength estimate between 0 and 1
        """
        if not hand:
            return 0.0
        
        total_strength = 0.0
        for card in hand:
            # Base strength from rank
            rank_strength = (card.rank.value - 2) / 12  # 2->0, Ace->1
            
            # Trump bonus
            if trump_suit and card.suit == trump_suit:
                rank_strength *= 1.3
            
            total_strength += rank_strength
        
        # Normalize by hand size
        return min(total_strength / len(hand), 1.0)
    
    def __str__(self):
        return self.name