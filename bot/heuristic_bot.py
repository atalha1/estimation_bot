"""
Heuristic bot implementation for Estimation card game.
Uses rule-based strategy with card analysis.
"""

import random
from typing import List, Optional, Union, Tuple, Dict
from estimation_bot.card import Card, Suit, Rank
from estimation_bot.player import BotInterface


class HeuristicBot(BotInterface):
    """Rule-based bot with strategic heuristics."""

    def __init__(self, name: str = "HeuristicBot"):
        self.name = name

    def make_bid(self, hand: List[Card], other_bids: List[Optional[Union[Tuple[int, Optional[Suit]], str]]], 
                 is_speed_round: bool = False, can_dash: bool = True) -> Optional[Union[Tuple[int, Optional[Suit]], str]]:
        """Strategic bidding based on hand analysis."""
        if is_speed_round:
            return None  # No bidding in speed rounds
        
        # Analyze hand strength for different trump suits
        suit_analysis = self._analyze_hand_by_suit(hand)
        
        # Find best trump option
        best_trump = None
        best_strength = 0
        best_bid = 0
        
        for trump_suit in [None, Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
            strength = self._calculate_hand_strength(hand, trump_suit)
            estimated_tricks = self._estimate_tricks(hand, trump_suit)
            
            if estimated_tricks >= 4 and strength > best_strength:
                best_strength = strength
                best_trump = trump_suit
                best_bid = min(estimated_tricks, 13)
        
        # Decide whether to bid
        if best_strength < 0.4:  # Weak hand
            if can_dash and best_strength < 0.2:
                return "DASH"
            return None  # Pass
        
        # Check if we can outbid others
        highest_bid = 0
        for bid in (other_bids.values() if hasattr(other_bids, 'values') else other_bids):
            if bid and bid != "DASH" and isinstance(bid, tuple):
                amount, trump = bid
                if amount > highest_bid:
                    highest_bid = amount
        
        if best_bid <= highest_bid:
            return None  # Can't outbid
        
        return (best_bid, best_trump)

    def make_estimation(self, hand: List[Card], trump_suit: Optional[Suit],
                       declarer_bid: int, other_estimations: List[Optional[int]],
                       is_last_estimator: bool = False, can_dash: bool = True) -> Union[int, str]:
        """Strategic estimation based on hand analysis."""
        
        hand_strength = self._calculate_hand_strength(hand, trump_suit)
        estimated_tricks = self._estimate_tricks(hand, trump_suit)
        
        # Consider dash for very weak hands
        if can_dash and hand_strength < 0.15 and estimated_tricks == 0:
            return "DASH"
        
        # Consider WITH for strong hands that match declarer bid
        if estimated_tricks >= declarer_bid * 0.8 and hand_strength > 0.7:
            return declarer_bid  # WITH
        
        # Regular estimation
        estimate = min(estimated_tricks, declarer_bid)
        
        # Handle risk constraint
        if is_last_estimator:
            current_total = sum(e for e in other_estimations if e is not None)
            if current_total + estimate == 13:
                # Prefer conservative adjustment
                if estimate > 0 and current_total < 13:
                    estimate -= 1
                else:
                    estimate += 1
                estimate = max(0, min(declarer_bid, estimate))

        return estimate

    def choose_card(self, hand: List[Card], valid_plays: List[Card],
                    trump_suit: Optional[Suit], led_suit: Optional[Suit],
                    trick_cards: List[Card]) -> Card:
        """Strategic card selection."""
        if not valid_plays:
            raise ValueError("No valid plays available")
        
        if len(valid_plays) == 1:
            return valid_plays[0]
        
        # Analyze trick situation
        if not trick_cards:  # Leading
            return self._choose_lead_card(valid_plays, trump_suit)
        else:  # Following
            return self._choose_follow_card(valid_plays, trump_suit, led_suit, trick_cards)

    def _choose_lead_card(self, valid_plays: List[Card], trump_suit: Optional[Suit]) -> Card:
        """Choose card when leading the trick."""
        # Prefer high non-trump cards to start tricks
        non_trump = [c for c in valid_plays if c.suit != trump_suit]
        if non_trump:
            return max(non_trump, key=lambda c: c.rank.value)
        
        # If only trump, play lowest trump
        return min(valid_plays, key=lambda c: c.rank.value)

    def _choose_follow_card(self, valid_plays: List[Card], trump_suit: Optional[Suit], 
                           led_suit: Optional[Suit], trick_cards: List[Card]) -> Card:
        """Choose card when following in a trick."""
        
        # Find highest card played so far
        highest_card = max(trick_cards, key=lambda c: self._card_trick_value(c, trump_suit, led_suit))
        
        # Try to win with lowest possible card
        winning_cards = [c for c in valid_plays 
                        if self._card_trick_value(c, trump_suit, led_suit) > 
                           self._card_trick_value(highest_card, trump_suit, led_suit)]
        
        if winning_cards:
            return min(winning_cards, key=lambda c: self._card_trick_value(c, trump_suit, led_suit))
        
        # Can't win, play lowest card
        return min(valid_plays, key=lambda c: self._card_trick_value(c, trump_suit, led_suit))

    def _card_trick_value(self, card: Card, trump_suit: Optional[Suit], led_suit: Optional[Suit]) -> int:
        """Calculate card's value in current trick context."""
        if trump_suit and card.suit == trump_suit:
            return 1000 + card.rank.value  # Trump cards always higher
        elif card.suit == led_suit:
            return 100 + card.rank.value   # Led suit cards
        else:
            return card.rank.value         # Off-suit cards

    def _analyze_hand_by_suit(self, hand: List[Card]) -> Dict[Suit, Dict]:
        """Analyze hand composition by suit."""
        analysis = {}
        
        for suit in Suit:
            suit_cards = [c for c in hand if c.suit == suit]
            analysis[suit] = {
                'count': len(suit_cards),
                'high_cards': len([c for c in suit_cards if c.rank.value >= 11]),
                'honors': len([c for c in suit_cards if c.rank.value >= 13]),  # A, K
                'strength': sum(c.rank.value for c in suit_cards) / max(len(suit_cards), 1)
            }
        
        return analysis

    def _calculate_hand_strength(self, hand: List[Card], trump_suit: Optional[Suit]) -> float:
        """Calculate overall hand strength for given trump."""
        if not hand:
            return 0.0
        
        total_strength = 0.0
        
        for card in hand:
            # Base strength from rank
            rank_strength = (card.rank.value - 2) / 12
            
            # Trump bonus
            if trump_suit and card.suit == trump_suit:
                rank_strength *= 1.5
            
            # Honor bonus
            if card.rank.value >= 11:  # J, Q, K, A
                rank_strength += 0.2
            
            total_strength += rank_strength
        
        return min(total_strength / len(hand), 1.0)

    def _estimate_tricks(self, hand: List[Card], trump_suit: Optional[Suit]) -> int:
        """Estimate likely tricks to be won."""
        estimated = 0.0
        
        for card in hand:
            if trump_suit and card.suit == trump_suit:
                # Trump cards
                if card.rank.value >= 13:  # A, K
                    estimated += 0.9
                elif card.rank.value >= 11:  # Q, J
                    estimated += 0.7
                else:
                    estimated += 0.4
            else:
                # Non-trump cards
                if card.rank == Rank.ACE:
                    estimated += 0.8
                elif card.rank == Rank.KING:
                    estimated += 0.5
                elif card.rank == Rank.QUEEN:
                    estimated += 0.3
                elif card.rank == Rank.JACK:
                    estimated += 0.2
        
        return int(round(estimated))

    def __str__(self):
        return self.name


class AdvancedHeuristicBot(HeuristicBot):
    """Enhanced heuristic bot with more sophisticated analysis."""
    
    def __init__(self, name: str = "AdvancedHeuristicBot"):
        super().__init__(name)
    
    def make_bid(self, hand: List[Card], other_bids: List[Optional[Union[Tuple[int, Optional[Suit]], str]]], 
                 is_speed_round: bool = False, can_dash: bool = True) -> Optional[Union[Tuple[int, Optional[Suit]], str]]:
        """Advanced bidding with opponent modeling."""
        if is_speed_round:
            return None
        
        # Analyze competition
        active_bidders = sum(1 for bid in (other_bids.values() if hasattr(other_bids, 'values') else other_bids) if bid is not None)
        highest_bid = 0

        for bid in (other_bids.values() if hasattr(other_bids, 'values') else other_bids):
            if bid and bid != "DASH" and isinstance(bid, tuple):
                amount, trump = bid
                highest_bid = max(highest_bid, amount)
        # More conservative if many active bidders
        competition_factor = active_bidders * 0.1
        
        # Analyze our hand
        suit_analysis = self._analyze_hand_by_suit(hand)
        best_trump = None
        best_strength = 0
        best_bid = 0
        
        for trump_suit in [None, Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
            strength = self._calculate_hand_strength(hand, trump_suit) - competition_factor
            estimated_tricks = self._estimate_tricks_advanced(hand, trump_suit)
            
            # Consider distribution bonuses
            if trump_suit:
                trump_count = suit_analysis[trump_suit]['count']
                if trump_count >= 5:  # Long trump suit
                    strength += 0.2
                elif trump_count <= 1:  # Short trump suit
                    strength -= 0.3
            
            if estimated_tricks >= 4 and strength > best_strength:
                best_strength = strength
                best_trump = trump_suit
                best_bid = min(estimated_tricks, 12)  # Cap at 12 for safety
        
        # Bidding decision
        if best_strength < 0.3:
            if can_dash and best_strength < 0.15:
                return "DASH"
            return None
        
        # Must outbid to win
        if best_bid <= highest_bid:
            if best_strength > 0.8:  # Very strong hand, try to outbid
                best_bid = min(highest_bid + 1, 12)
            else:
                return None
        
        return (best_bid, best_trump)
    
    def _estimate_tricks_advanced(self, hand: List[Card], trump_suit: Optional[Suit]) -> int:
        """Advanced trick estimation with suit distribution."""
        estimated = 0.0
        suit_counts = {}
        
        # Count cards by suit
        for suit in Suit:
            suit_counts[suit] = len([c for c in hand if c.suit == suit])
        
        for card in hand:
            if trump_suit and card.suit == trump_suit:
                # Trump cards - consider length
                trump_length = suit_counts[trump_suit]
                base_value = 0.9 if card.rank.value >= 13 else 0.7 if card.rank.value >= 11 else 0.4
                
                # Length bonus
                if trump_length >= 5:
                    base_value += 0.2
                elif trump_length <= 2:
                    base_value -= 0.1
                
                estimated += base_value
            else:
                # Non-trump cards - consider void possibilities
                suit_length = suit_counts[card.suit]
                
                if card.rank == Rank.ACE:
                    estimated += 0.8 if suit_length >= 3 else 0.6
                elif card.rank == Rank.KING:
                    estimated += 0.5 if suit_length >= 2 else 0.3
                elif card.rank == Rank.QUEEN:
                    estimated += 0.3 if suit_length >= 2 else 0.1
                elif card.rank == Rank.JACK:
                    estimated += 0.2 if suit_length >= 2 else 0.1
        
        return int(round(estimated))
    
    def __str__(self):
        return self.name