"""
Enhanced data collection for neural network training.
Location: training/data_collector.py
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import h5py
from datetime import datetime

from estimation_bot.card import Card, Suit, Rank
from estimation_bot.player import Player
from estimation_bot.round import Round, Trick


@dataclass
class GameState:
    """Complete game state at any point in time."""
    # Meta info
    game_id: str
    round_number: int
    trick_number: int
    timestamp: float
    
    # Game state
    trump_suit: Optional[Suit]
    is_speed_round: bool
    current_scores: List[int]  # Score for each player
    
    # Round state
    declarer_id: Optional[int]
    declarer_bid: Optional[int]
    estimations: Dict[int, int]
    tricks_won: List[int]  # Tricks won so far per player
    
    # Current trick
    led_suit: Optional[Suit]
    cards_played: List[Optional[Card]]  # Cards played in current trick
    current_player: int
    
    # Player hands (encoded)
    hands_encoded: List[List[int]]  # Binary encoding of each player's hand
    
    def to_vector(self) -> np.ndarray:
        """Convert game state to feature vector for neural network."""
        features = []
        
        # Basic features
        features.extend([
            self.round_number / 18,  # Normalize
            self.trick_number / 13,
            float(self.is_speed_round),
            self.current_player / 3
        ])
        
        # Trump suit (one-hot: None, S, H, D, C)
        trump_encoding = [0] * 5
        if self.trump_suit is None:
            trump_encoding[0] = 1
        else:
            trump_encoding[list(Suit).index(self.trump_suit) + 1] = 1
        features.extend(trump_encoding)
        
        # Scores (normalized)
        max_score = max(self.current_scores) if max(self.current_scores) > 0 else 1
        features.extend([s / max_score for s in self.current_scores])
        
        # Estimations and tricks
        for i in range(4):
            features.extend([
                self.estimations.get(i, 0) / 13,
                self.tricks_won[i] / 13,
                float(i == self.declarer_id)
            ])
        
        # Current trick state
        features.extend([
            float(self.led_suit is not None),
            len([c for c in self.cards_played if c is not None]) / 4
        ])
        
        # Flatten hands encoding
        for hand in self.hands_encoded:
            features.extend(hand)
        
        return np.array(features, dtype=np.float32)


@dataclass 
class DecisionPoint:
    """A decision point in the game requiring action."""
    state: GameState
    decision_type: str  # 'bid', 'estimation', 'play_card'
    player_id: int
    
    # Context
    valid_actions: List[Any]  # Valid bids, estimations, or cards
    
    # Outcome (filled after decision)
    action_taken: Any
    immediate_reward: float = 0.0
    
    # For card play decisions
    trick_winner: Optional[int] = None
    trick_points: Optional[int] = None


class EnhancedDataCollector:
    """Collects detailed game data for neural network training."""
    
    def __init__(self):
        self.current_game_states: List[GameState] = []
        self.decision_points: List[DecisionPoint] = []
        self.completed_games: List[Dict] = []
        
    def encode_hand(self, hand: List[Card]) -> List[int]:
        """Encode hand as 52-bit vector."""
        encoding = [0] * 52
        for card in hand:
            # Map card to index (0-51)
            suit_offset = list(Suit).index(card.suit) * 13
            rank_offset = card.rank.value - 2  # 2 is lowest rank
            encoding[suit_offset + rank_offset] = 1
        return encoding
    
    def encode_card(self, card: Card) -> int:
        """Encode single card as index."""
        suit_offset = list(Suit).index(card.suit) * 13
        rank_offset = card.rank.value - 2
        return suit_offset + rank_offset
    
    def capture_game_state(self, game, round_obj: Round, trick: Optional[Trick] = None) -> GameState:
        """Capture current game state."""
        # Encode hands
        hands_encoded = []
        for player in game.players:
            hands_encoded.append(self.encode_hand(player.hand))
        
        # Current trick info
        cards_played = [None] * 4
        led_suit = None
        if trick:
            led_suit = trick.led_suit
            for pid, card in trick.cards.items():
                cards_played[pid] = card
        
        return GameState(
            game_id=f"game_{id(game)}",
            round_number=round_obj.round_number,
            trick_number=len(round_obj.tricks),
            timestamp=datetime.now().timestamp(),
            trump_suit=round_obj.trump_suit,
            is_speed_round=round_obj.is_speed_round,
            current_scores=[p.score for p in game.players],
            declarer_id=round_obj.declarer_id,
            declarer_bid=round_obj.declarer_bid,
            estimations=dict(round_obj.estimations),
            tricks_won=[p.tricks_won for p in game.players],
            led_suit=led_suit,
            cards_played=cards_played,
            current_player=round_obj.leader_id if trick else 0,
            hands_encoded=hands_encoded
        )
    
    def record_bid_decision(self, game, round_obj: Round, player_id: int,
                           valid_bids: List[Tuple[int, Optional[Suit]]],
                           bid_made: Optional[Tuple[int, Optional[Suit]]]):
        """Record a bidding decision."""
        state = self.capture_game_state(game, round_obj)
        
        decision = DecisionPoint(
            state=state,
            decision_type='bid',
            player_id=player_id,
            valid_actions=valid_bids,
            action_taken=bid_made
        )
        
        self.decision_points.append(decision)
    
    def record_estimation_decision(self, game, round_obj: Round, player_id: int,
                                  valid_estimations: List[int], estimation_made: int):
        """Record an estimation decision."""
        state = self.capture_game_state(game, round_obj)
        
        decision = DecisionPoint(
            state=state,
            decision_type='estimation',
            player_id=player_id,
            valid_actions=valid_estimations,
            action_taken=estimation_made
        )
        
        self.decision_points.append(decision)
    
    def record_card_decision(self, game, round_obj: Round, trick: Trick,
                           player_id: int, valid_cards: List[Card], card_played: Card):
        """Record a card play decision."""
        state = self.capture_game_state(game, round_obj, trick)
        
        decision = DecisionPoint(
            state=state,
            decision_type='play_card',
            player_id=player_id,
            valid_actions=[self.encode_card(c) for c in valid_cards],
            action_taken=self.encode_card(card_played)
        )
        
        self.decision_points.append(decision)
    
    def finalize_trick(self, trick_winner: int, points: int = 1):
        """Update the last card decisions with trick outcome."""
        # Find last 4 card decisions
        card_decisions = []
        for i in range(len(self.decision_points) - 1, -1, -1):
            if self.decision_points[i].decision_type == 'play_card':
                card_decisions.append(self.decision_points[i])
                if len(card_decisions) == 4:
                    break
        
        # Update with trick outcome
        for decision in card_decisions:
            decision.trick_winner = trick_winner
            decision.trick_points = points
            # Simple immediate reward
            if decision.player_id == trick_winner:
                decision.immediate_reward = 1.0
            else:
                decision.immediate_reward = 0.0
    
    def finalize_game(self, final_scores: Dict[int, int], winner_id: int):
        """Process completed game and calculate rewards."""
        # Calculate final rewards for all decisions
        max_score = max(final_scores.values())
        
        for decision in self.decision_points:
            player_score = final_scores[decision.player_id]
            # Normalize reward between -1 and 1
            decision.final_reward = (player_score - 0) / max(abs(max_score), 1)
            
            # Bonus for winning
            if decision.player_id == winner_id:
                decision.final_reward += 0.5
        
        # Store completed game
        game_data = {
            'decisions': self.decision_points,
            'final_scores': final_scores,
            'winner_id': winner_id,
            'num_decisions': len(self.decision_points)
        }
        
        self.completed_games.append(game_data)
        
        # Reset for next game
        self.decision_points = []
        self.current_game_states = []
    
    def save_to_hdf5(self, filepath: str):
        """Save collected data in HDF5 format for efficient loading."""
        with h5py.File(filepath, 'w') as f:
            # Create groups
            games_group = f.create_group('games')
            
            for i, game_data in enumerate(self.completed_games):
                game_group = games_group.create_group(f'game_{i}')
                
                # Save decision points
                decisions = game_data['decisions']
                if decisions:
                    # Stack state vectors
                    states = np.stack([d.state.to_vector() for d in decisions])
                    game_group.create_dataset('states', data=states)
                    
                    # Save actions and rewards
                    actions = np.array([d.action_taken for d in decisions])
                    rewards = np.array([getattr(d, 'final_reward', 0) for d in decisions])
                    
                    game_group.create_dataset('actions', data=actions)
                    game_group.create_dataset('rewards', data=rewards)
                    
                    # Save metadata
                    game_group.attrs['num_decisions'] = len(decisions)
                    game_group.attrs['winner_id'] = game_data['winner_id']
                    game_group.attrs['final_scores'] = json.dumps(game_data['final_scores'])
        
        print(f"Saved {len(self.completed_games)} games to {filepath}")
    
    def create_training_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a training batch from collected data."""
        all_decisions = []
        for game in self.completed_games:
            all_decisions.extend(game['decisions'])
        
        if len(all_decisions) < batch_size:
            return None, None, None
        
        # Random sample
        indices = np.random.choice(len(all_decisions), batch_size, replace=False)
        batch_decisions = [all_decisions[i] for i in indices]
        
        # Extract features
        states = np.stack([d.state.to_vector() for d in batch_decisions])
        actions = np.array([d.action_taken for d in batch_decisions])
        rewards = np.array([getattr(d, 'final_reward', 0) for d in batch_decisions])
        
        return states, actions, rewards


class DataCollectorMixin:
    """
    Mixin for game classes to enable data collection.
    Add this to EstimationGame for automatic data collection.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_collector = EnhancedDataCollector()
        self._collection_enabled = kwargs.get('collect_data', False)
    
    def enable_data_collection(self):
        """Enable detailed data collection."""
        self._collection_enabled = True
    
    def disable_data_collection(self):
        """Disable data collection."""
        self._collection_enabled = False
    
    def get_collected_data(self) -> EnhancedDataCollector:
        """Get the data collector instance."""
        return self.data_collector
    
    # Override methods to collect data
    def _collect_bid_decision(self, *args, **kwargs):
        if self._collection_enabled:
            self.data_collector.record_bid_decision(*args, **kwargs)
    
    def _collect_estimation_decision(self, *args, **kwargs):
        if self._collection_enabled:
            self.data_collector.record_estimation_decision(*args, **kwargs)
    
    def _collect_card_decision(self, *args, **kwargs):
        if self._collection_enabled:
            self.data_collector.record_card_decision(*args, **kwargs)