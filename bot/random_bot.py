"""
Random bot implementation for Estimation card game.
Provides a baseline bot that makes random legal moves.
"""

import random
from typing import List, Optional
from ..game.card import Card, Suit
from ..game.player import BotInterface


class RandomBot(BotInterface):
    """Bot that makes completely random legal moves."""
    
    def __init__(self, name: str = "RandomBot"):
        self.name = name
    
    def make_bid(self, hand: List[Card], trump_suit: Suit, 
                 other_bids: List[Optional[int]]) -> int:
        """
        Make a random bid between 0 and hand size.
        
        Args:
            hand: Current hand of cards
            trump_suit: Trump suit for this round  
            other_bids: Bids made by other players
            
        Returns:
            Random bid (0 to hand_size)
        """
        return random.randint(0, len(hand))
    
    def choose_card(self, hand: List[Card], valid_plays: List[Card],
                   trump_suit: Suit, led_suit: Optional[Suit],
                   trick_cards: List[Card]) -> Card:
        """
        Choose a random valid card to play.
        
        Args:
            hand: Current hand
            valid_plays: Cards that can legally be played
            trump_suit: Trump suit for this round
            led_suit: Suit led in current trick
            trick_cards: Cards already played in current trick
            
        Returns:
            Randomly selected valid card
        """
        if not valid_plays:
            raise ValueError("No valid plays available")
        
        return random.choice(valid_plays)
    
    def __str__(self):
        return self.name


class WeightedRandomBot(BotInterface):
    """Slightly smarter random bot with basic preferences."""
    
    def __init__(self, name: str = "WeightedRandomBot"):
        self.name = name
    
    def make_bid(self, hand: List[Card], trump_suit: Suit, 
                 other_bids: List[Optional[int]]) -> int:
        """
        Make a bid with slight preference for conservative bids.
        
        Args:
            hand: Current hand of cards
            trump_suit: Trump suit for this round
            other_bids: Bids made by other players
            
        Returns:
            Weighted random bid
        """
        hand_strength = self._estimate_hand_strength(hand, trump_suit)
        max_bid = len(hand)
        
        # Weight towards lower bids for weak hands, higher for strong hands
        if hand_strength < 0.3:
            # Weak hand - prefer 0-3 bids
            weights = [3, 2, 2, 1] + [0.5] * (max_bid - 3)
        elif hand_strength > 0.7:
            # Strong hand - prefer higher bids
            low_weight = 0.5
            high_weight = 2
            weights = [low_weight] * 3 + [high_weight] * (max_bid - 2)
        else:
            # Medium hand - fairly uniform
            weights = [1] * (max_bid + 1)
        
        # Ensure weights list matches bid range
        weights = weights[:max_bid + 1]
        if len(weights) < max_bid + 1:
            weights.extend([0.5] * (max_bid + 1 - len(weights)))
        
        return random.choices(range(max_bid + 1), weights=weights)[0]
    
    def choose_card(self, hand: List[Card], valid_plays: List[Card],
                   trump_suit: Suit, led_suit: Optional[Suit],
                   trick_cards: List[Card]) -> Card:
        """
        Choose card with slight strategic preference.
        
        Args:
            hand: Current hand
            valid_plays: Cards that can legally be played  
            trump_suit: Trump suit for this round
            led_suit: Suit led in current trick
            trick_cards: Cards already played in current trick
            
        Returns:
            Weighted random card selection
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
            if card.suit == trump_suit and led_suit != trump_suit:
                weight += 1.5
            
            # Slightly prefer high cards in general
            weight += (card.rank.value - 7) / 20
            
            weights.append(max(weight, 0.1))  # Ensure positive weight
        
        return random.choices(valid_plays, weights=weights)[0]
    
    def _estimate_hand_strength(self, hand: List[Card], trump_suit: Suit) -> float:
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
            if card.suit == trump_suit:
                rank_strength *= 1.3
            
            total_strength += rank_strength
        
        # Normalize by hand size
        return min(total_strength / len(hand), 1.0)
    
    def __str__(self):
        return self.name