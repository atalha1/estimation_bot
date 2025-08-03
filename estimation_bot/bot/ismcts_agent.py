#!/usr/bin/env python3
"""
ISMCTS Agent - Information Set Monte Carlo Tree Search for Estimation
Location: estimation_bot/bot/ismcts_agent.py

A competitive AI agent using hybrid ISMCTS + heuristics approach.
"""

import random
import math
import time
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy

from ..card import Card, Suit, Rank
from ..player import BotInterface



@dataclass 
class MCTSNode:
    """MCTS tree node for decision tracking."""
    state_key: str
    parent: Optional['MCTSNode'] = None
    children: Dict[str, 'MCTSNode'] = None
    visits: int = 0
    wins: float = 0.0
    untried_moves: List = None
    player_id: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.untried_moves is None:
            self.untried_moves = []
    
    def uct_value(self, c_param: float = 1.4) -> float:
        """Calculate UCT value for node selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self) -> 'MCTSNode':
        """Select best child using UCT."""
        return max(self.children.values(), key=lambda c: c.uct_value())
    
    def add_child(self, move, state_key: str, player_id: int) -> 'MCTSNode':
        """Add new child node."""
        child = MCTSNode(state_key, parent=self, player_id=player_id)
        self.children[str(move)] = child
        if move in self.untried_moves:
            self.untried_moves.remove(move)
        return child
    
    def update(self, result: float):
        """Backpropagate result through tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.update(result)


class HandReconstructor:
    """Reconstructs unknown opponent hands using available information."""
    
    def __init__(self):
        self.known_cards = set()
        self.void_suits = defaultdict(set)  # player_id -> set of void suits
        
    def update_void(self, player_id: int, suit: Suit):
        """Record that player is void in suit."""
        self.void_suits[player_id].add(suit)
    
    def update_played_cards(self, cards: List[Card]):
        """Update known played cards."""
        self.known_cards.update(cards)
    
    def generate_hands(self, my_hand: List[Card], num_players: int = 4) -> List[List[Card]]:
        """Generate plausible opponent hands."""
        # Get remaining cards
        all_cards = set()
        for suit in Suit:
            for rank in Rank:
                all_cards.add(Card(suit, rank))
        
        remaining_cards = list(all_cards - self.known_cards - set(my_hand))
        random.shuffle(remaining_cards)
        
        # Distribute to other players
        hand_size = len(my_hand)
        hands = [[] for _ in range(num_players)]
        hands[0] = my_hand  # Assume we're player 0
        
        card_idx = 0
        for player_id in range(1, num_players):
            player_hand = []
            cards_needed = hand_size
            
            # Respect void constraints
            valid_cards = []
            for card in remaining_cards[card_idx:]:
                if card.suit not in self.void_suits[player_id]:
                    valid_cards.append(card)
                if len(valid_cards) >= cards_needed:
                    break
            
            # Fill hand
            player_hand = valid_cards[:cards_needed]
            hands[player_id] = player_hand
            card_idx += cards_needed
        
        return hands


class HeuristicEvaluator:
    """Fast heuristic evaluation for MCTS simulations."""
    
    @staticmethod
    def heuristic_bid(hand: List[Card], trump_suit: Optional[Suit]) -> int:
        """Estimate reasonable trick count for hand."""
        if not hand:
            return 0
            
        score = 0.0
        suit_counts = defaultdict(int)
        
        # Count cards by suit
        for card in hand:
            suit_counts[card.suit] += 1
        
        # Evaluate each card
        for card in hand:
            rank_value = card.rank.value
            is_trump = card.suit == trump_suit
            suit_length = suit_counts[card.suit]
            
            # High cards
            if rank_value == 14:  # Ace
                score += 1.0 if not is_trump else 1.2
            elif rank_value == 13:  # King
                if suit_length >= 2:  # Protected King
                    score += 0.8 if not is_trump else 1.0
                else:
                    score += 0.3  # Singleton King
            elif rank_value == 12:  # Queen
                score += 0.5 if not is_trump else 0.7
            elif rank_value == 11:  # Jack
                score += 0.3 if not is_trump else 0.5
            
            # Trump bonus
            if is_trump and rank_value >= 9:
                score += 0.3
        
        # Suit length bonuses
        for suit, count in suit_counts.items():
            if count >= 5:
                score += 1.0  # Long suit
            elif count == 0 and suit != trump_suit:
                score -= 1.0  # Void penalty (except trump)
        
        return max(0, min(len(hand), round(score)))
    
    @staticmethod
    def zero_bid_probability(hand: List[Card], trump_suit: Optional[Suit]) -> float:
        """Calculate probability of making zero tricks."""
        if not hand:
            return 1.0
        
        # Multiplicative model: P(success) = ∏(1 - danger[suit])
        prob_success = 1.0
        suit_cards = defaultdict(list)
        
        # Group by suit
        for card in hand:
            suit_cards[card.suit].append(card)
        
        for suit, cards in suit_cards.items():
            is_trump = suit == trump_suit
            
            # Count low cards (≤ 8)
            low_cards = sum(1 for c in cards if c.rank.value <= 8)
            
            if low_cards >= 3:
                danger = 0.0  # Safe with multiple low cards
            else:
                # Estimate danger based on high cards remaining
                high_cards_in_suit = max(0, 6 - low_cards)  # Rough estimate
                unseen_cards = 13 - len(cards)  # Cards we don't have
                
                if unseen_cards > 0:
                    danger = min(1.0, high_cards_in_suit / unseen_cards)
                else:
                    danger = 1.0
                
                # Trump suits are more dangerous
                if is_trump:
                    danger *= 1.5
            
            prob_success *= (1.0 - danger)
        
        return max(0.0, min(1.0, prob_success))
    
    @staticmethod
    def simulate_zero_bid(hand: List[Card], trump_suit: Optional[Suit], 
                         num_sims: int = 50) -> float:
        """Run fast simulations to evaluate nil success rate."""
        if not hand:
            return 1.0
        
        successes = 0
        
        for _ in range(num_sims):
            # Simple simulation: check if we have all low cards
            tricks_won = 0
            
            # Count potential tricks (simplified)
            for card in hand:
                if card.rank.value >= 12:  # Q, K, A
                    if card.suit == trump_suit:
                        tricks_won += 1  # Trump high cards likely win
                    elif card.rank.value == 14:  # Aces often win
                        tricks_won += 1
            
            # Add randomness
            if random.random() < 0.3:  # 30% chance of unexpected trick
                tricks_won += 1
            
            if tricks_won == 0:
                successes += 1
        
        return successes / num_sims


class ISMCTSAgent(BotInterface):
    """Information Set Monte Carlo Tree Search agent with enhanced heuristics."""
    
    def __init__(self, name: str, simulations_per_move: int = 150):
        self.name = name
        self.simulations_per_move = simulations_per_move
        self.reconstructor = HandReconstructor()
        self.evaluator = HeuristicEvaluator()
        self.root = None
        
        # Enhanced heuristics (will be imported when available)
        try:
            from .enhanced_heuristics import EnhancedHeuristicEvaluator
            self.enhanced_evaluator = EnhancedHeuristicEvaluator()
            self.use_enhanced = True
        except ImportError:
            self.enhanced_evaluator = None
            self.use_enhanced = False
        
        # Adaptive parameters
        self.nil_threshold = 0.4
        self.endgame_threshold = 5  # Switch to full MCTS when ≤ 5 tricks left
        self.uct_c_param = 1.4
        
        # Performance tracking
        self.move_times = []
        self.total_simulations = 0
        self.nil_attempts = 0
        self.nil_successes = 0
        
    def make_bid(self, hand: List[Card], other_bids: Dict[int, tuple], 
                 is_last_bidder: bool, can_bid_dash: bool) -> Optional[Tuple[int, Optional[Suit]]]:
        """Make bidding decision using enhanced heuristics + ISMCTS."""
        
        if self.use_enhanced:
            # Use comprehensive evaluation
            evaluation = self.enhanced_evaluator.comprehensive_bid_evaluation(
                hand, other_bids, None
            )
            
            nil_eval = evaluation['nil_evaluation']
            bid_eval = evaluation['bid_evaluation']
            
            # Check for strong nil bid
            if can_bid_dash and nil_eval['probability'] > self.nil_threshold:
                confidence = nil_eval['confidence']
                if confidence > 0.6:
                    self.nil_attempts += 1
                    return "DASH"
            
            # Regular competitive bid
            recommended = bid_eval['recommended_bid']
            if recommended:
                return recommended
            else:
                return None  # Pass
        
        else:
            # Fallback to basic heuristics
            heuristic_estimate = self.evaluator.heuristic_bid(hand, None)
            
            # Consider zero bid
            zero_prob = self.evaluator.zero_bid_probability(hand, None)
            if zero_prob > self.nil_threshold and can_bid_dash:
                sim_prob = self.evaluator.simulate_zero_bid(hand, None)
                if sim_prob > 0.6:
                    self.nil_attempts += 1
                    return "DASH"
            
            # Competitive bidding
            highest_bid = 0
            if isinstance(other_bids, dict):
                bid_values = other_bids.values()
            else:
                bid_values = other_bids  # It's already a list
                
            for bid_data in bid_values:
                if bid_data and len(bid_data) >= 1:
                    highest_bid = max(highest_bid, bid_data[0])
            
            if highest_bid > 0:
                my_bid = max(highest_bid + 1, heuristic_estimate)
                if my_bid <= len(hand):
                    trump_suit = self._select_trump_suit(hand)
                    return (my_bid, trump_suit)
                return None
            else:
                bid_amount = max(4, heuristic_estimate)
                trump_suit = self._select_trump_suit(hand)
                return (bid_amount, trump_suit)
    
    def _select_trump_suit(self, hand: List[Card]) -> Optional[Suit]:
        """Select best trump suit based on hand strength."""
        suit_scores = {}
        suit_counts = defaultdict(int)
        
        # Count cards and high cards by suit
        for card in hand:
            suit_counts[card.suit] += 1
            if card.suit not in suit_scores:
                suit_scores[card.suit] = 0
            
            # Score high cards more in potential trump
            if card.rank.value >= 11:  # J, Q, K, A
                suit_scores[card.suit] += card.rank.value - 10
        
        # Prefer suits with good length + high cards
        best_suit = None
        best_score = 0
        
        for suit, count in suit_counts.items():
            if count >= 3:  # Need reasonable length
                total_score = suit_scores[suit] + (count - 3) * 2  # Length bonus
                if total_score > best_score:
                    best_score = total_score
                    best_suit = suit
        
        return best_suit
    
    def _mcts_bid(self, hand: List[Card], other_bids: Dict[int, tuple],
                  is_last_bidder: bool, can_bid_dash: bool) -> Optional[Tuple[int, Optional[Suit]]]:
        """Use MCTS for competitive bidding decisions."""
        
        # For now, fallback to heuristic (MCTS bidding is complex)
        heuristic_estimate = self.evaluator.heuristic_bid(hand, None)
        
        # Analyze competition
        highest_bid = 0
        bid_values = other_bids.values() if isinstance(other_bids, dict) else other_bids
        for bid_data in bid_values:
            if bid_data and len(bid_data) >= 1:
                highest_bid = max(highest_bid, bid_data[0])
        
        # Competitive strategy
        if highest_bid > 0:
            my_bid = max(highest_bid + 1, heuristic_estimate)
            if my_bid <= len(hand):
                trump_suit = self._select_trump_suit(hand)
                return (my_bid, trump_suit)
        
        # Pass if can't compete
        return None
    
    def make_estimation(self, hand: List[Card], trump_suit: Optional[Suit], 
                       declarer_bid: int, current_estimations: List[int],
                       is_last_estimator: bool, can_dash: bool) -> int:
        """Make estimation using heuristics."""
        
        heuristic_estimate = self.evaluator.heuristic_bid(hand, trump_suit)
        
        # Consider zero bid
        if can_dash:
            zero_prob = self.evaluator.zero_bid_probability(hand, trump_suit)
            if zero_prob > self.nil_threshold:
                sim_prob = self.evaluator.simulate_zero_bid(hand, trump_suit)
                if sim_prob > 0.6:
                    return "DASH"
        
        # Risk player logic
        if is_last_estimator:
            total_so_far = sum(est for est in current_estimations if est is not None)
            if total_so_far + heuristic_estimate == 13:
                # Avoid exact 13 (Risk)
                return max(0, heuristic_estimate - 1)
        
        return max(0, heuristic_estimate)
    
    def choose_card(self, hand: List[Card], valid_plays: List[Card], 
                   trump_suit: Optional[Suit], led_suit: Optional[Suit],
                   cards_played: List[Card]) -> Card:
        """Choose card using enhanced heuristics + ISMCTS based on game phase."""
        
        start_time = time.time()
        
        # Early game: use enhanced heuristics
        if len(hand) > self.endgame_threshold:
            if self.use_enhanced:
                chosen_card = self._enhanced_heuristic_play(hand, valid_plays, trump_suit, 
                                                          led_suit, cards_played)
            else:
                chosen_card = self._heuristic_play(hand, valid_plays, trump_suit, 
                                                 led_suit, cards_played)
        else:
            # Endgame: use MCTS
            chosen_card = self._mcts_play(hand, valid_plays, trump_suit, led_suit, cards_played)
        
        # Track performance
        move_time = time.time() - start_time
        self.move_times.append(move_time)
        
        return chosen_card
    
    def _enhanced_heuristic_play(self, hand: List[Card], valid_plays: List[Card],
                               trump_suit: Optional[Suit], led_suit: Optional[Suit],
                               cards_played: List[Card]) -> Card:
        """Enhanced heuristic card selection using comprehensive evaluation."""
        
        if not valid_plays:
            return hand[0]
        
        # Build game state for evaluation
        game_state = {
            'trump_suit': trump_suit,
            'led_suit': led_suit,
            'cards_played': cards_played,
            'my_estimation': getattr(self, 'my_estimation', 3),  # Default estimation
            'tricks_won': getattr(self, 'tricks_won', 0)
        }
        
        # Get evaluations for all valid plays
        evaluations = self.enhanced_evaluator.evaluate_card_plays(hand, valid_plays, game_state)
        
        # Return best card
        return evaluations[0][0] if evaluations else valid_plays[0]
    
    def _heuristic_play(self, hand: List[Card], valid_plays: List[Card],
                       trump_suit: Optional[Suit], led_suit: Optional[Suit],
                       cards_played: List[Card]) -> Card:
        """Fast heuristic card selection."""
        
        if not valid_plays:
            return hand[0]
        
        # Leading
        if led_suit is None:
            # Lead lowest safe card
            non_trump = [c for c in valid_plays if c.suit != trump_suit]
            if non_trump:
                return min(non_trump, key=lambda c: c.rank.value)
            else:
                return min(valid_plays, key=lambda c: c.rank.value)
        
        # Following suit
        else:
            following_cards = [c for c in valid_plays if c.suit == led_suit]
            
            if following_cards:
                # Try to play low if we can't win
                return min(following_cards, key=lambda c: c.rank.value)
            else:
                # Can't follow - trump or discard
                trumps = [c for c in valid_plays if c.suit == trump_suit]
                if trumps:
                    return min(trumps, key=lambda c: c.rank.value)  # Low trump
                else:
                    return min(valid_plays, key=lambda c: c.rank.value)  # Low discard
    
    def _mcts_play(self, hand: List[Card], valid_plays: List[Card],
                  trump_suit: Optional[Suit], led_suit: Optional[Suit],
                  cards_played: List[Card]) -> Card:
        """MCTS card selection for endgame."""
        
        if len(valid_plays) == 1:
            return valid_plays[0]
        
        # Update reconstructor with new information
        self.reconstructor.update_played_cards(cards_played)
        
        # Initialize root node
        state_key = self._get_state_key(hand, cards_played, trump_suit)
        self.root = MCTSNode(state_key, untried_moves=valid_plays.copy())
        
        # Run MCTS simulations
        for _ in range(self.simulations_per_move):
            self._mcts_simulate()
            self.total_simulations += 1
        
        # Select best move
        if self.root.children:
            best_child = max(self.root.children.values(), 
                           key=lambda c: c.visits)  # Most visited = most promising
            
            # Find corresponding card
            for move_str, child in self.root.children.items():
                if child == best_child:
                    # Parse move back to card
                    for card in valid_plays:
                        if str(card) == move_str:
                            return card
        
        # Fallback to heuristic
        return self._heuristic_play(hand, valid_plays, trump_suit, led_suit, cards_played)
    
    def _mcts_simulate(self):
        """Run single MCTS simulation."""
        node = self.root
        path = [node]
        
        # Selection phase
        while node.untried_moves == [] and node.children:
            node = node.select_child()
            path.append(node)
        
        # Expansion phase
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            state_key = self._get_state_key([], [], None)  # Simplified
            node = node.add_child(move, state_key, 0)
            path.append(node)
        
        # Simulation phase (rollout)
        result = self._rollout()
        
        # Backpropagation
        for node in path:
            node.update(result)
    
    def _rollout(self) -> float:
        """Fast rollout simulation."""
        # Simplified: random outcome for now
        return random.random()
    
    def _get_state_key(self, hand: List[Card], played: List[Card], trump: Optional[Suit]) -> str:
        """Generate unique state key."""
        hand_str = ','.join(sorted(str(c) for c in hand))
        played_str = ','.join(sorted(str(c) for c in played))
        trump_str = str(trump) if trump else 'None'
        return f"{hand_str}|{played_str}|{trump_str}"
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get agent performance statistics."""
        if not self.move_times:
            return {}
        
        nil_success_rate = self.nil_successes / max(1, self.nil_attempts)
        
        return {
            'avg_move_time': sum(self.move_times) / len(self.move_times),
            'max_move_time': max(self.move_times),
            'total_simulations': self.total_simulations,
            'simulations_per_move': self.simulations_per_move,
            'nil_attempts': self.nil_attempts,
            'nil_successes': self.nil_successes,
            'nil_success_rate': nil_success_rate,
            'moves_played': len(self.move_times)
        }
    
    def update_nil_result(self, succeeded: bool):
        """Update nil bid tracking."""
        if succeeded:
            self.nil_successes += 1


# Factory function for integration with existing training system
def create_ismcts_agent(name: str, **kwargs) -> ISMCTSAgent:
    """Factory function to create ISMCTS agent."""
    simulations = kwargs.get('simulations_per_move', 150)
    return ISMCTSAgent(name, simulations)


if __name__ == "__main__":
    # Quick test
    agent = ISMCTSAgent("Test Agent")
    
    # Test heuristic evaluation
    test_hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.SPADES, Rank.KING),  
        Card(Suit.HEARTS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.JACK),
        Card(Suit.DIAMONDS, Rank.TWO)
    ]
    
    bid_estimate = agent.evaluator.heuristic_bid(test_hand, Suit.SPADES)
    zero_prob = agent.evaluator.zero_bid_probability(test_hand, Suit.SPADES)
    
    print(f"Test hand bid estimate: {bid_estimate}")
    print(f"Zero bid probability: {zero_prob:.2%}")
    print("ISMCTS Agent initialized successfully!")