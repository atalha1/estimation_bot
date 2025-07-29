"""
Feature extraction module for Estimation RL bot.
Converts game state into numerical features for neural network input.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from estimation_bot.card import Card, Suit, Rank
from estimation_bot.player import Player


class FeatureExtractor:
    """Extracts numerical features from game state for RL training."""
    
    def __init__(self):
        self.suit_to_index = {suit: i for i, suit in enumerate(Suit)}
        self.rank_to_index = {rank: i for i, rank in enumerate(Rank)}
        
    def extract_bid_features(self, hand: List[Card], other_bids: List[Optional[tuple]],
                           player_id: int, is_speed_round: bool = False) -> np.ndarray:
        """
        Extract features for bidding decision.
        
        Args:
            hand: Player's cards
            other_bids: Other players' bids [(amount, trump_suit), ...]
            player_id: Current player ID
            is_speed_round: Whether this is a speed round
            
        Returns:
            Feature vector for neural network
        """
        features = []
        
        # Hand representation (52 binary features for each card)
        hand_vector = np.zeros(52)
        for card in hand:
            card_idx = self.suit_to_index[card.suit] * 13 + self.rank_to_index[card.rank]
            hand_vector[card_idx] = 1
        features.extend(hand_vector)
        
        # Hand statistics by suit
        for suit in Suit:
            suit_cards = [c for c in hand if c.suit == suit]
            features.append(len(suit_cards))  # Count
            if suit_cards:
                features.append(max(c.rank.value for c in suit_cards))  # Highest rank
                features.append(sum(c.rank.value for c in suit_cards) / len(suit_cards))  # Avg rank
            else:
                features.extend([0, 0])
        
        # High cards count
        high_cards = len([c for c in hand if c.rank.value >= 11])
        aces = len([c for c in hand if c.rank.value == 14])
        features.extend([high_cards, aces])
        
        # Other players' bids (encoded)
        for i, bid in enumerate(other_bids):
            if i == player_id:
                continue
            if bid is None:
                features.extend([0, 0, 0, 0, 0])  # [has_bid, amount, is_spades, is_hearts, etc.]
            else:
                amount, trump_suit = bid
                features.append(1)  # has bid
                features.append(amount / 13)  # normalized amount
                # Trump suit one-hot
                trump_features = [0, 0, 0, 0]  # [spades, hearts, diamonds, clubs]
                if trump_suit is not None:
                    trump_features[self.suit_to_index[trump_suit]] = 1
                features.extend(trump_features)
        
        # Game context
        features.append(1 if is_speed_round else 0)
        features.append(player_id / 3)  # Normalized position
        
        return np.array(features, dtype=np.float32)
    
    def extract_estimation_features(self, hand: List[Card], trump_suit: Optional[Suit],
                                  declarer_bid: int, other_estimations: List[Optional[int]],
                                  player_id: int, is_last_estimator: bool = False) -> np.ndarray:
        """
        Extract features for estimation decision.
        
        Args:
            hand: Player's cards
            trump_suit: Determined trump suit
            declarer_bid: Winning bid amount
            other_estimations: Other players' estimations
            player_id: Current player ID
            is_last_estimator: Whether this is the risk player
            
        Returns:
            Feature vector for neural network
        """
        features = []
        
        # Hand representation (same as bidding)
        hand_vector = np.zeros(52)
        for card in hand:
            card_idx = self.suit_to_index[card.suit] * 13 + self.rank_to_index[card.rank]
            hand_vector[card_idx] = 1
        features.extend(hand_vector)
        
        # Trump-specific analysis
        if trump_suit is not None:
            trump_cards = [c for c in hand if c.suit == trump_suit]
            features.append(len(trump_cards))
            if trump_cards:
                features.append(max(c.rank.value for c in trump_cards))
                features.append(sum(c.rank.value for c in trump_cards) / len(trump_cards))
            else:
                features.extend([0, 0])
            
            # Trump suit one-hot
            trump_onehot = [0, 0, 0, 0]
            trump_onehot[self.suit_to_index[trump_suit]] = 1
            features.extend(trump_onehot)
        else:
            # No trump
            features.extend([0, 0, 0, 0, 0, 0, 1])
        
        # Declarer bid context
        features.append(declarer_bid / 13)
        
        # Other estimations
        total_estimations = declarer_bid
        for est in other_estimations:
            if est is not None:
                features.append(est / 13)
                total_estimations += est
            else:
                features.append(-1)  # Not yet estimated
        
        # Risk analysis
        features.append(total_estimations / 13)
        features.append(1 if is_last_estimator else 0)
        remaining_total = 13 - total_estimations
        features.append(remaining_total / 13)
        
        return np.array(features, dtype=np.float32)
    
    def extract_play_features(self, hand: List[Card], valid_plays: List[Card],
                            trump_suit: Optional[Suit], led_suit: Optional[Suit],
                            trick_cards: List[Card], cards_seen: set,
                            player_estimation: int, tricks_won: int) -> np.ndarray:
        """
        Extract features for card play decision.
        
        Args:
            hand: Current hand
            valid_plays: Legal cards to play
            trump_suit: Trump suit for round
            led_suit: Suit led in current trick
            trick_cards: Cards already played in trick
            cards_seen: All cards seen so far
            player_estimation: Player's estimation
            tricks_won: Tricks won so far
            
        Returns:
            Feature vector for neural network
        """
        features = []
        
        # Hand representation
        hand_vector = np.zeros(52)
        for card in hand:
            card_idx = self.suit_to_index[card.suit] * 13 + self.rank_to_index[card.rank]
            hand_vector[card_idx] = 1
        features.extend(hand_vector)
        
        # Valid plays mask
        valid_plays_vector = np.zeros(52)
        for card in valid_plays:
            card_idx = self.suit_to_index[card.suit] * 13 + self.rank_to_index[card.rank]
            valid_plays_vector[card_idx] = 1
        features.extend(valid_plays_vector)
        
        # Trick context
        features.append(len(trick_cards))  # Position in trick
        features.append(1 if led_suit is None else 0)  # Leading
        
        # Current trick cards
        trick_vector = np.zeros(52)
        for card in trick_cards:
            card_idx = self.suit_to_index[card.suit] * 13 + self.rank_to_index[card.rank]
            trick_vector[card_idx] = 1
        features.extend(trick_vector)
        
        # Trump and led suit context
        if trump_suit is not None:
            trump_onehot = [0, 0, 0, 0]
            trump_onehot[self.suit_to_index[trump_suit]] = 1
            features.extend(trump_onehot)
        else:
            features.extend([0, 0, 0, 0])
        
        if led_suit is not None:
            led_onehot = [0, 0, 0, 0]
            led_onehot[self.suit_to_index[led_suit]] = 1
            features.extend(led_onehot)
        else:
            features.extend([0, 0, 0, 0])
        
        # Player state
        features.append(player_estimation / 13)
        features.append(tricks_won / 13)
        features.append((player_estimation - tricks_won) / 13)  # Tricks needed
        
        # Cards seen (for card counting)
        seen_vector = np.zeros(52)
        for card in cards_seen:
            card_idx = self.suit_to_index[card.suit] * 13 + self.rank_to_index[card.rank]
            seen_vector[card_idx] = 1
        features.extend(seen_vector)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of different feature vectors."""
        return {
            'bid_features': 52 + 16 + 2 + 15 + 2,  # hand + suit_stats + high_cards + other_bids + context
            'estimation_features': 52 + 7 + 1 + 3 + 3,  # hand + trump_analysis + declarer + estimations + risk
            'play_features': 52 * 4 + 16 + 3,  # hand + valid + trick + contexts + state + seen
        }


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to [0, 1] range."""
    # Most features are already normalized or binary
    # Could add more sophisticated normalization if needed
    return np.clip(features, 0, 1)


def encode_card_for_network(card: Card) -> int:
    """Encode a card as a single integer for network input."""
    suit_idx = list(Suit).index(card.suit)
    rank_idx = list(Rank).index(card.rank)
    return suit_idx * 13 + rank_idx


def decode_network_card(card_idx: int) -> Card:
    """Decode network output back to a card."""
    suit_idx = card_idx // 13
    rank_idx = card_idx % 13
    suit = list(Suit)[suit_idx]
    rank = list(Rank)[rank_idx]
    return Card(suit, rank)
    
"""
Feature extraction module for Estimation RL bot.
Converts game state into numerical features for neural network input.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from estimation_bot.card import Card, Suit, Rank
from estimation_bot.player import Player


class FeatureExtractor:
    """Extracts numerical features from game state for RL training."""
    
    def __init__(self):
        self.suit_to_index = {suit: i for i, suit in enumerate(Suit)}
        self.rank_to_index = {rank: i for i, rank in enumerate(Rank)}
        
    def extract_bid_features(self, hand: List[Card], trump_suit: Suit,
                           other_bids: List[Optional[int]], 
                           player_id: int) -> np.ndarray:
        """
        Extract