"""
Heuristic bot implementation for Estimation card game.
Uses rule-based strategy with card counting and basic tactics.
"""

from typing import List, Optional, Dict
from ..game.card import Card, Suit, Rank
from ..game.player import BotInterface


class HeuristicBot(BotInterface):
    """Rule-based bot with strategic heuristics."""
    
    def __init__(self, name: str = "HeuristicBot"):
        self.name = name
    
    def make_bid(self, hand: List[Card], trump_suit: Suit, 
                 other_bids: List[Optional[int]]) -> int:
        """
        Make strategic bid based on hand analysis.
        
        Args:
            hand: Current hand of cards
            trump_suit: Trump suit for this round
            other_bids: Bids made by other players
            
        Returns:
            Strategic bid
        """
        # Analyze hand strength
        trump_cards = [c for c in hand if c.suit == trump_suit]
        high_cards = [c for c in hand if c.rank.value >= 11]  # J, Q, K, A
        aces = [c for c in hand if c.rank.value == 14]
        
        # Count likely tricks
        likely_tricks = 0
        
        # Trump aces are almost guaranteed tricks
        trump_aces = [c for c in trump_cards if c.rank.value == 14]
        likely_tricks += len(trump_aces)
        
        # High trump cards are likely tricks
        high_trump = [c for c in trump_cards if c.rank.value >= 12]
        likely_tricks += len(high_trump) * 0.8
        
        # Non-trump aces are likely tricks if we have length in suit
        for card in aces:
            if card.suit != trump_suit:
                suit_length = len([c for c in hand if c.suit == card.suit])
                if suit_length >= 3:  # Good chance ace will hold up
                    likely_tricks += 0.7
        
        # Kings in long suits
        kings = [c for c in hand if c.rank.value == 13 and c.suit != trump_suit]
        for king in kings:
            suit_length = len([c for c in hand if c.suit == king.suit])
            if suit_length >= 4:
                likely_tricks += 0.5
        
        # Medium trump cards if we have many trump
        if len(trump_cards) >= 4:
            medium_trump = [c for c in trump_cards if 9 <= c.rank.value <= 11]
            likely_tricks += len(medium_trump) * 0.3
        
        # Conservative adjustment - bid slightly under estimate
        base_bid = int(likely_tricks * 0.9)
        
        # Adjust based on position and other bids
        total_other_bids = sum(bid for bid in other_bids if bid is not None)
        remaining_tricks = 13 - total_other_bids
        
        # If others have bid very low, we might be able to bid higher
        if total_other_bids < 8:
            base_bid = min(base_bid + 1, len(hand))
        
        # If others have bid high, be more conservative
        if total_other_bids > 10:
            base_bid = max(base_bid - 1, 0)
        
        return max(0, min(base_bid, len(hand)))
    
    def choose_card(self, hand: List[Card], valid_plays: List[Card],
                   trump_suit: Suit, led_suit: Optional[Suit],
                   trick_cards: List[Card]) -> Card:
        """
        Choose card using strategic heuristics.
        
        Args:
            hand: Current hand
            valid_plays: Cards that can legally be played
            trump_suit: Trump suit for this round
            led_suit: Suit led in current trick
            trick_cards: Cards already played in current trick
            
        Returns:
            Strategically chosen card
        """
        if not valid_plays:
            raise ValueError("No valid plays available")
        
        if len(valid_plays) == 1:
            return valid_plays[0]
        
        # Leading a trick
        if led_suit is None:
            return self._choose_lead_card(valid_plays, trump_suit, hand)
        
        # Following in a trick
        return self._choose_follow_card(valid_plays, trump_suit, led_suit, 
                                      trick_cards, hand)
    
    def _choose_lead_card(self, valid_plays: List[Card], trump_suit: Suit, 
                         hand: List[Card]) -> Card:
        """Choose best card to lead."""
        # Prefer leading aces (likely to win)
        aces = [c for c in valid_plays if c.rank.value == 14]
        if aces:
            # Lead non-trump ace if we have one
            non_trump_aces = [c for c in aces if c.suit != trump_suit]
            if non_trump_aces:
                return non_trump_aces[0]
            return aces[0]
        
        # Lead kings from long suits
        kings = [c for c in valid_plays if c.rank.value == 13]
        for king in kings:
            suit_length = len([c for c in hand if c.suit == king.suit])
            if suit_length >= 4:
                return king
        
        # Lead low cards from short suits (to get rid of them)
        suit_lengths = {}
        for card in hand:
            suit_lengths[card.suit] = suit_lengths.get(card.suit, 0) + 1
        
        short_suit_cards = []
        for card in valid_plays:
            if suit_lengths[card.suit] <= 2 and card.rank.value <= 9:
                short_suit_cards.append(card)
        
        if short_suit_cards:
            return min(short_suit_cards, key=lambda c: c.rank.value)
        
        # Default: lead lowest card
        return min(valid_plays, key=lambda c: c.rank.value)
    
    def _choose_follow_card(self, valid_plays: List[Card], trump_suit: Suit,
                           led_suit: Suit, trick_cards: List[Card], 
                           hand: List[Card]) -> Card:
        """Choose best card when following suit."""
        # Determine if we can win the trick
        winning_card = self._get_current_winning_card(trick_cards, trump_suit, led_suit)
        
        # Cards that can beat the current winner
        winning_cards = [c for c in valid_plays 
                        if c.beats(winning_card, trump_suit, led_suit)]
        
        if winning_cards:
            # We can win - choose lowest winning card to conserve high cards
            return min(winning_cards, key=lambda c: c.rank.value)
        
        # We can't win - play lowest card to conserve high cards
        if led_suit and any(c.suit == led_suit for c in valid_plays):
            # Must follow suit - play lowest in suit
            following_cards = [c for c in valid_plays if c.suit == led_suit]
            return min(following_cards, key=lambda c: c.rank.value)
        
        # Can't follow suit - discard lowest card (unless it's trump)
        non_trump = [c for c in valid_plays if c.suit != trump_suit]
        if non_trump:
            return min(non_trump, key=lambda c: c.rank.value)
        
        # Only trump cards available - play lowest trump
        return min(valid_plays, key=lambda c: c.rank.value)
    
    def _get_current_winning_card(self, trick_cards: List[Card], 
                                trump_suit: Suit, led_suit: Suit) -> Card:
        """Determine which card is currently winning the trick."""
        if not trick_cards:
            return None
        
        winning_card = trick_cards[0]
        for card in trick_cards[1:]:
            if card.beats(winning_card, trump_suit, led_suit):
                winning_card = card
        
        return winning_card
    
    def __str__(self):
        return self.name


class AdvancedHeuristicBot(HeuristicBot):
    """More sophisticated heuristic bot with card counting."""
    
    def __init__(self, name: str = "AdvancedHeuristicBot"):
        super().__init__(name)
        self.cards_seen = set()
        self.cards_played_by_suit = {suit: [] for suit in Suit}
    
    def make_bid(self, hand: List[Card], trump_suit: Suit, 
                 other_bids: List[Optional[int]]) -> int:
        """Enhanced bidding with card counting."""
        # Reset card tracking for new round
        self.cards_seen = set(hand)
        self.cards_played_by_suit = {suit: [] for suit in Suit}
        
        # Use parent bidding logic as base
        base_bid = super().make_bid(hand, trump_suit, other_bids)
        
        # Adjust based on card counting knowledge
        # (In a real implementation, we'd track cards from previous tricks)
        
        return base_bid
    
    def choose_card(self, hand: List[Card], valid_plays: List[Card],
                   trump_suit: Suit, led_suit: Optional[Suit],
                   trick_cards: List[Card]) -> Card:
        """Enhanced card selection with memory of played cards."""
        # Track cards we've seen
        for card in trick_cards:
            self.cards_seen.add(card)
            self.cards_played_by_suit[card.suit].append(card)
        
        # Use parent logic but with card counting awareness
        return super().choose_card(hand, valid_plays, trump_suit, led_suit, trick_cards)
    
    def _estimate_remaining_high_cards(self, suit: Suit) -> int:
        """Estimate how many high cards remain in a suit."""
        high_ranks = [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK]
        played_high = len([c for c in self.cards_played_by_suit[suit] 
                          if c.rank in high_ranks])
        return len(high_ranks) - played_high