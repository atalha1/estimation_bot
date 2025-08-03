#!/usr/bin/env python3
"""
Enhanced Heuristics Module - Advanced evaluation functions for ISMCTS
Location: estimation_bot/bot/enhanced_heuristics.py

Sophisticated heuristic evaluation functions for bidding, play, and nil estimation.
"""

import random
import math
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict, Counter

from ..card import Card, Suit, Rank


class HandAnalyzer:
    """Advanced hand analysis for strategic decision making."""
    
    @staticmethod
    def analyze_hand_strength(hand: List[Card], trump_suit: Optional[Suit]) -> Dict[str, float]:
        """Comprehensive hand strength analysis."""
        
        if not hand:
            return {'total_strength': 0.0, 'suit_strengths': {}, 'high_card_points': 0}
        
        suit_cards = defaultdict(list)
        for card in hand:
            suit_cards[card.suit].append(card)
        
        # Sort cards in each suit by rank
        for suit in suit_cards:
            suit_cards[suit].sort(key=lambda c: c.rank.value, reverse=True)
        
        suit_strengths = {}
        total_strength = 0.0
        hcp = 0  # High card points
        
        for suit, cards in suit_cards.items():
            strength = HandAnalyzer._analyze_suit_strength(cards, suit == trump_suit)
            suit_strengths[suit.name] = strength
            total_strength += strength
            
            # Calculate high card points (A=4, K=3, Q=2, J=1)
            for card in cards:
                if card.rank == Rank.ACE:
                    hcp += 4
                elif card.rank == Rank.KING:
                    hcp += 3
                elif card.rank == Rank.QUEEN:
                    hcp += 2
                elif card.rank == Rank.JACK:
                    hcp += 1
        
        return {
            'total_strength': total_strength,
            'suit_strengths': suit_strengths,
            'high_card_points': hcp,
            'trump_length': len(suit_cards.get(trump_suit, [])) if trump_suit else 0,
            'longest_suit': max(len(cards) for cards in suit_cards.values()),
            'voids': [suit.name for suit in Suit if suit not in suit_cards],
            'singletons': [suit.name for suit, cards in suit_cards.items() if len(cards) == 1]
        }
    
    @staticmethod
    def _analyze_suit_strength(cards: List[Card], is_trump: bool) -> float:
        """Analyze strength of cards in a single suit."""
        if not cards:
            return 0.0
        
        strength = 0.0
        length = len(cards)
        
        # Length bonus/penalty
        if length >= 5:
            strength += 1.5  # Long suit bonus
        elif length == 4:
            strength += 0.5
        elif length == 1:
            strength -= 0.3  # Singleton penalty
        
        # Analyze each card
        for i, card in enumerate(cards):
            rank_val = card.rank.value
            
            # Base strength by rank
            if rank_val == 14:  # Ace
                strength += 1.2
            elif rank_val == 13:  # King
                # Protected vs unprotected king
                if i < length - 1:  # Has cards behind it
                    strength += 0.9
                else:
                    strength += 0.4  # Unprotected
            elif rank_val == 12:  # Queen
                strength += 0.6 if i < length - 1 else 0.3
            elif rank_val == 11:  # Jack
                strength += 0.4 if i < length - 1 else 0.2
            elif rank_val == 10:
                strength += 0.3
            elif rank_val >= 8:
                strength += 0.1
            else:
                strength += 0.05  # Low cards have minimal value
        
        # Trump bonus
        if is_trump:
            strength *= 1.3
            # Extra bonus for trump length
            if length >= 4:
                strength += 0.5
        
        return strength


class NilBidEvaluator:
    """Specialized evaluator for nil (zero) bid decisions."""
    
    @staticmethod
    def evaluate_nil_probability(hand: List[Card], trump_suit: Optional[Suit], 
                                opponents_info: Dict = None) -> Dict[str, float]:
        """Comprehensive nil bid evaluation."""
        
        if not hand:
            return {'probability': 1.0, 'confidence': 1.0, 'risk_factors': []}
        
        # Basic danger analysis
        danger_score = 0.0
        risk_factors = []
        
        suit_cards = defaultdict(list)
        for card in hand:
            suit_cards[card.suit].append(card)
        
        # Analyze each suit for danger
        for suit, cards in suit_cards.items():
            cards.sort(key=lambda c: c.rank.value, reverse=True)
            is_trump = suit == trump_suit
            
            suit_danger = NilBidEvaluator._evaluate_suit_danger(cards, is_trump)
            danger_score += suit_danger
            
            # Identify specific risk factors
            if suit_danger > 0.7:
                risk_factors.append(f"High danger in {suit.name}")
            
            # Check for unprotected high cards
            if cards and cards[0].rank.value >= 12 and len(cards) <= 2:
                risk_factors.append(f"Unprotected {cards[0].rank.name} of {suit.name}")
        
        # Trump-specific risks
        if trump_suit and trump_suit in suit_cards:
            trump_cards = suit_cards[trump_suit]
            if len(trump_cards) >= 4:
                risk_factors.append("Long trump suit")
                danger_score += 0.3
        
        # Void analysis (good for nil)
        voids = [suit for suit in Suit if suit not in suit_cards]
        void_bonus = len(voids) * 0.2
        danger_score -= void_bonus
        
        if voids:
            risk_factors.append(f"Voids help: {[s.name for s in voids]}")
        
        # Calculate final probability
        base_prob = max(0.1, min(0.9, 1.0 - (danger_score / 4.0)))
        
        # Confidence based on hand clarity
        confidence = NilBidEvaluator._calculate_confidence(hand, trump_suit)
        
        return {
            'probability': base_prob,
            'confidence': confidence,
            'danger_score': danger_score,
            'risk_factors': risk_factors,
            'void_count': len(voids)
        }
    
    @staticmethod
    def _evaluate_suit_danger(cards: List[Card], is_trump: bool) -> float:
        """Evaluate danger level for a single suit."""
        if not cards:
            return 0.0  # Void is safe
        
        danger = 0.0
        length = len(cards)
        
        # High card danger
        for card in cards:
            rank_val = card.rank.value
            if rank_val == 14:  # Ace
                danger += 0.9
            elif rank_val == 13:  # King
                danger += 0.7 if length <= 2 else 0.4
            elif rank_val == 12:  # Queen
                danger += 0.5 if length <= 2 else 0.2
            elif rank_val == 11:  # Jack
                danger += 0.3 if length <= 1 else 0.1
        
        # Length protection
        if length >= 4:
            danger *= 0.6  # Long suits offer protection
        elif length == 1:
            danger *= 1.5  # Singletons are dangerous
        
        # Trump danger multiplier
        if is_trump:
            danger *= 1.4
        
        return min(1.0, danger)
    
    @staticmethod
    def _calculate_confidence(hand: List[Card], trump_suit: Optional[Suit]) -> float:
        """Calculate confidence in nil bid assessment."""
        confidence = 0.7  # Base confidence
        
        # More confidence with extreme hands
        hcp = sum(4 if c.rank == Rank.ACE else 
                 3 if c.rank == Rank.KING else
                 2 if c.rank == Rank.QUEEN else
                 1 if c.rank == Rank.JACK else 0 
                 for c in hand)
        
        if hcp <= 2:  # Very weak hand
            confidence += 0.2
        elif hcp >= 12:  # Very strong hand
            confidence += 0.2
        
        # Void suits increase confidence
        suit_count = len(set(card.suit for card in hand))
        if suit_count <= 3:
            confidence += 0.1
        
        return min(1.0, confidence)


class PlayHeuristics:
    """Advanced heuristics for card play decisions."""
    
    @staticmethod
    def evaluate_card_play(card: Card, game_state: Dict, hand: List[Card]) -> float:
        """Evaluate the quality of playing a specific card."""
        
        trump_suit = game_state.get('trump_suit')
        led_suit = game_state.get('led_suit')
        cards_played = game_state.get('cards_played', [])
        my_estimation = game_state.get('my_estimation', 0)
        tricks_won = game_state.get('tricks_won', 0)
        
        score = 0.0
        
        # Leading logic
        if led_suit is None:
            score += PlayHeuristics._evaluate_lead(card, hand, trump_suit, my_estimation, tricks_won)
        else:
            # Following logic
            score += PlayHeuristics._evaluate_follow(card, hand, led_suit, trump_suit, 
                                                   cards_played, my_estimation, tricks_won)
        
        return score
    
    @staticmethod
    def _evaluate_lead(card: Card, hand: List[Card], trump_suit: Optional[Suit],
                      estimation: int, tricks_won: int) -> float:
        """Evaluate leading a specific card."""
        score = 0.0
        
        is_trump = card.suit == trump_suit
        rank_val = card.rank.value
        
        # Need tricks vs want to avoid tricks
        need_tricks = tricks_won < estimation
        avoid_tricks = tricks_won >= estimation or estimation == 0
        
        if need_tricks:
            # Lead high cards to win tricks
            if rank_val >= 12:  # Q, K, A
                score += 0.8
            elif rank_val >= 10:
                score += 0.4
            
            # Trump leads are strong
            if is_trump:
                score += 0.5
        
        elif avoid_tricks:
            # Lead low cards to avoid winning
            if rank_val <= 8:
                score += 0.6
            elif rank_val >= 13:  # K, A
                score -= 0.8
            
            # Avoid trump leads if possible
            if is_trump:
                score -= 0.3
        
        # Suit preference (lead from length)
        suit_length = sum(1 for c in hand if c.suit == card.suit)
        if suit_length >= 4:
            score += 0.3  # Lead from long suit
        elif suit_length == 1:
            score += 0.2  # Lead singleton to eliminate suit
        
        return score
    
    @staticmethod
    def _evaluate_follow(card: Card, hand: List[Card], led_suit: Suit, 
                        trump_suit: Optional[Suit], cards_played: List[Card],
                        estimation: int, tricks_won: int) -> float:
        """Evaluate following with a specific card."""
        score = 0.0
        
        is_trump = card.suit == trump_suit
        follows_suit = card.suit == led_suit
        rank_val = card.rank.value
        
        need_tricks = tricks_won < estimation
        avoid_tricks = tricks_won >= estimation or estimation == 0
        
        # Analyze trick situation
        can_win = PlayHeuristics._can_win_trick(card, cards_played, led_suit, trump_suit)
        
        if need_tricks and can_win:
            score += 1.0  # Win tricks we need
        elif avoid_tricks and can_win:
            score -= 0.8  # Avoid winning unwanted tricks
        elif need_tricks and not can_win:
            score -= 0.2  # Not helping our goal
        elif avoid_tricks and not can_win:
            score += 0.4  # Good, avoiding trick
        
        # Following suit preferences
        if follows_suit:
            if need_tricks:
                # Play high to win
                if rank_val >= 11:
                    score += 0.3
            else:
                # Play low to lose
                if rank_val <= 9:
                    score += 0.3
        
        # Trump play logic
        elif is_trump:
            if need_tricks:
                score += 0.5  # Trump to win
            else:
                score -= 0.6  # Avoid trumping unless necessary
        
        # Discard logic (can't follow, not trump)
        else:
            score += 0.2  # Safe discard
            # Prefer discarding from short suits
            suit_length = sum(1 for c in hand if c.suit == card.suit)
            if suit_length == 1:
                score += 0.3
        
        return score
    
    @staticmethod
    def _can_win_trick(card: Card, cards_played: List[Card], led_suit: Suit, 
                      trump_suit: Optional[Suit]) -> bool:
        """Determine if playing this card would win the trick."""
        if not cards_played:
            return True  # Leading always "wins" initially
        
        current_winner = cards_played[0]
        
        # Check each played card to find current winner
        for played_card in cards_played:
            if played_card.beats(current_winner, trump_suit, led_suit):
                current_winner = played_card
        
        # Can our card beat the current winner?
        return card.beats(current_winner, trump_suit, led_suit)


class BiddingStrategy:
    """Advanced bidding strategy with competitive analysis."""
    
    @staticmethod
    def evaluate_bid_decision(hand: List[Card], other_bids: Dict[int, tuple],
                             trump_suit: Optional[Suit] = None) -> Dict[str, Any]:
        """Comprehensive bid evaluation."""
        
        # Hand analysis
        analyzer = HandAnalyzer()
        hand_analysis = analyzer.analyze_hand_strength(hand, trump_suit)
        
        # Calculate base trick-taking potential
        base_estimate = BiddingStrategy._calculate_base_estimate(hand_analysis, trump_suit)
        
        # Competitive analysis
        competition_analysis = BiddingStrategy._analyze_competition(other_bids)
        
        # Risk assessment
        risk_assessment = BiddingStrategy._assess_bidding_risk(hand, other_bids)
        
        # Final bid recommendation
        recommended_bid = BiddingStrategy._determine_final_bid(
            base_estimate, competition_analysis, risk_assessment, hand
        )
        
        return {
            'recommended_bid': recommended_bid,
            'base_estimate': base_estimate,
            'hand_strength': hand_analysis['total_strength'],
            'competition_level': competition_analysis['highest_bid'],
            'risk_level': risk_assessment['risk_score'],
            'confidence': risk_assessment['confidence']
        }
    
    @staticmethod
    def _calculate_base_estimate(hand_analysis: Dict, trump_suit: Optional[Suit]) -> int:
        """Calculate base trick estimate from hand analysis."""
        strength = hand_analysis['total_strength']
        hcp = hand_analysis['high_card_points']
        
        # Convert strength to trick estimate
        base_tricks = min(13, max(0, round(strength)))
        
        # Adjust based on distribution
        if hand_analysis['voids']:
            base_tricks += len(hand_analysis['voids']) * 0.5  # Void bonus
        
        if hand_analysis['longest_suit'] >= 6:
            base_tricks += 1  # Very long suit bonus
        
        return int(base_tricks)
    
    @staticmethod
    def _analyze_competition(other_bids: Dict[int, tuple]) -> Dict[str, Any]:
        """Analyze competitive bidding situation."""
        if not other_bids:
            return {'highest_bid': 0, 'num_bidders': 0, 'competition_level': 'none'}
        
        if isinstance(other_bids, dict):
            active_bids = [bid for bid in other_bids.values() if bid is not None]
        else:
            active_bids = [bid for bid in other_bids if bid is not None]
        
        if not active_bids:
            return {'highest_bid': 0, 'num_bidders': 0, 'competition_level': 'none'}
        
        bid_amounts = [bid[0] for bid in active_bids if bid is not None and len(bid) >= 1]
        highest_bid = max(bid_amounts) if bid_amounts else 0
        num_bidders = len(active_bids)
        
        if highest_bid >= 8:
            competition_level = 'high'
        elif highest_bid >= 6:
            competition_level = 'medium'
        else:
            competition_level = 'low'
        
        return {
            'highest_bid': highest_bid,
            'num_bidders': num_bidders,
            'competition_level': competition_level
        }
    
    @staticmethod
    def _assess_bidding_risk(hand: List[Card], other_bids: Dict[int, tuple]) -> Dict[str, float]:
        """Assess risk of various bidding decisions."""
        
        # Calculate hand reliability
        suit_distribution = defaultdict(int)
        for card in hand:
            suit_distribution[card.suit] += 1
        
        # Balanced hands are more reliable
        suit_lengths = list(suit_distribution.values())
        balance_score = 1.0 - (max(suit_lengths) - min(suit_lengths)) / 13
        
        # High card concentration
        hcp_in_longest = 0
        longest_suit = max(suit_distribution.keys(), key=lambda s: suit_distribution[s])
        for card in hand:
            if card.suit == longest_suit and card.rank.value >= 11:
                hcp_in_longest += 1
        
        concentration_risk = hcp_in_longest / max(1, suit_distribution[longest_suit])
        
        risk_score = (1.0 - balance_score) + (concentration_risk * 0.5)
        confidence = max(0.3, 1.0 - risk_score)
        
        return {
            'risk_score': risk_score,
            'confidence': confidence,
            'balance_score': balance_score
        }
    
    @staticmethod
    def _determine_final_bid(base_estimate: int, competition: Dict, 
                           risk: Dict, hand: List[Card]) -> Optional[Tuple[int, Optional[Suit]]]:
        """Determine final bid based on all factors."""
        
        highest_bid = competition['highest_bid']
        confidence = risk['confidence']
        
        # Conservative adjustment based on confidence
        if confidence < 0.6:
            adjusted_estimate = max(0, base_estimate - 1)
        else:
            adjusted_estimate = base_estimate
        
        # Competitive considerations
        if highest_bid > 0:
            # Need to bid higher to compete
            min_competitive_bid = highest_bid + 1
            
            if adjusted_estimate >= min_competitive_bid and confidence > 0.5:
                # We can compete
                bid_amount = min_competitive_bid
            else:
                # Can't compete reliably, pass
                return None
        else:
            # First to bid
            bid_amount = max(4, adjusted_estimate)  # Minimum opening bid
        
        # Select trump suit
        trump_suit = BiddingStrategy._select_optimal_trump(hand)
        
        return (bid_amount, trump_suit)
    
    @staticmethod 
    def _select_optimal_trump(hand: List[Card]) -> Optional[Suit]:
        """Select optimal trump suit for the hand."""
        suit_scores = {}
        
        for suit in Suit:
            suit_cards = [c for c in hand if c.suit == suit]
            if len(suit_cards) < 3:  # Need minimum length
                continue
            
            # Score this suit as trump
            score = len(suit_cards) * 0.5  # Length bonus
            
            for card in suit_cards:
                if card.rank.value >= 11:  # J, Q, K, A
                    score += card.rank.value - 10
            
            suit_scores[suit] = score
        
        if not suit_scores:
            return None  # No trump
        
        return max(suit_scores.keys(), key=lambda s: suit_scores[s])


class FastSimulator:
    """Fast game simulation for MCTS rollouts."""
    
    @staticmethod
    def simulate_trick_outcome(my_card: Card, opponent_cards: List[Card],
                              trump_suit: Optional[Suit], led_suit: Optional[Suit]) -> bool:
        """Quickly simulate if our card wins the trick."""
        
        if not opponent_cards:
            return True
        
        # Find current best card
        best_card = my_card
        for opp_card in opponent_cards:
            if opp_card.beats(best_card, trump_suit, led_suit):
                best_card = opp_card
        
        return best_card == my_card
    
    @staticmethod
    def simulate_hand_outcome(hand: List[Card], trump_suit: Optional[Suit],
                             estimation: int, num_simulations: int = 30) -> float:
        """Simulate multiple hands to estimate success probability."""
        
        if not hand:
            return 1.0 if estimation == 0 else 0.0
        
        successes = 0
        
        for _ in range(num_simulations):
            # Simple simulation: estimate tricks based on card strength
            estimated_tricks = 0
            
            for card in hand:
                # Probability this card wins a trick
                win_prob = FastSimulator._estimate_card_win_probability(
                    card, trump_suit, len(hand)
                )
                
                if random.random() < win_prob:
                    estimated_tricks += 1
            
            if estimated_tricks == estimation:
                successes += 1
        
        return successes / num_simulations
    
    @staticmethod
    def _estimate_card_win_probability(card: Card, trump_suit: Optional[Suit], 
                                     hand_size: int) -> float:
        """Estimate probability that a card wins a trick."""
        
        is_trump = card.suit == trump_suit
        rank_val = card.rank.value
        
        # Base probability by rank
        if rank_val == 14:  # Ace
            base_prob = 0.9
        elif rank_val == 13:  # King
            base_prob = 0.7
        elif rank_val == 12:  # Queen
            base_prob = 0.5
        elif rank_val == 11:  # Jack
            base_prob = 0.3
        elif rank_val == 10:
            base_prob = 0.2
        else:
            base_prob = 0.1
        
        # Trump bonus
        if is_trump:
            base_prob = min(0.95, base_prob * 1.4)
        
        # Adjust for game phase (more tricks available early)
        phase_multiplier = min(1.0, hand_size / 8.0)
        
        return base_prob * phase_multiplier


# Integration class for ISMCTS agent
class EnhancedHeuristicEvaluator:
    """Enhanced evaluator that integrates all heuristic modules."""
    
    def __init__(self):
        self.hand_analyzer = HandAnalyzer()
        self.nil_evaluator = NilBidEvaluator()
        self.play_heuristics = PlayHeuristics()
        self.bidding_strategy = BiddingStrategy()
        self.simulator = FastSimulator()
    
    def comprehensive_bid_evaluation(self, hand: List[Card], other_bids: Dict[int, tuple],
                                   trump_suit: Optional[Suit] = None) -> Dict[str, Any]:
        """Get comprehensive bidding recommendation."""
        
        # Hand strength analysis
        hand_analysis = self.hand_analyzer.analyze_hand_strength(hand, trump_suit)
        
        # Nil bid evaluation
        nil_evaluation = self.nil_evaluator.evaluate_nil_probability(hand, trump_suit)
        
        # Regular bid evaluation
        bid_evaluation = self.bidding_strategy.evaluate_bid_decision(hand, other_bids, trump_suit)
        
        return {
            'hand_analysis': hand_analysis,
            'nil_evaluation': nil_evaluation,
            'bid_evaluation': bid_evaluation,
            'recommendation': self._make_final_recommendation(nil_evaluation, bid_evaluation)
        }
    
    def _make_final_recommendation(self, nil_eval: Dict, bid_eval: Dict) -> str:
        """Make final bidding recommendation."""
        
        nil_prob = nil_eval['probability']
        nil_confidence = nil_eval['confidence']
        
        # Strong nil candidate
        if nil_prob > 0.7 and nil_confidence > 0.6:
            return "STRONG_NIL"
        elif nil_prob > 0.5 and nil_confidence > 0.7:
            return "CONSIDER_NIL"
        else:
            return f"BID_{bid_eval['recommended_bid']}"
    
    def evaluate_card_plays(self, hand: List[Card], valid_plays: List[Card],
                           game_state: Dict) -> List[Tuple[Card, float]]:
        """Evaluate all valid card plays and return ranked list."""
        
        evaluations = []
        
        for card in valid_plays:
            score = self.play_heuristics.evaluate_card_play(card, game_state, hand)
            evaluations.append((card, score))
        
        # Sort by score (best first)
        evaluations.sort(key=lambda x: x[1], reverse=True)
        
        return evaluations


if __name__ == "__main__":
    # Quick test of enhanced heuristics
    from ..card import Card, Suit, Rank
    
    evaluator = EnhancedHeuristicEvaluator()
    
    # Test hand
    test_hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.SPADES, Rank.KING),
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.CLUBS, Rank.THREE),
        Card(Suit.DIAMONDS, Rank.FOUR)
    ]
    
    # Test bid evaluation
    bid_result = evaluator.comprehensive_bid_evaluation(test_hand, {}, Suit.SPADES)
    
    print("Enhanced Heuristics Test:")
    print(f"Hand strength: {bid_result['hand_analysis']['total_strength']:.2f}")
    print(f"Nil probability: {bid_result['nil_evaluation']['probability']:.2%}")
    print(f"Recommendation: {bid_result['recommendation']}")
    print("✅ Enhanced heuristics module ready!")