"""
Round module for Estimation card game.
Handles trick resolution, round progression, and proper scoring calculation.
"""

from typing import List, Dict, Optional, Tuple
from estimation_bot.card import Card, Suit
from estimation_bot.player import Player


class Trick:
    """Represents a single trick in the game."""
    
    def __init__(self, leader_id: int):
        self.leader_id = leader_id
        self.cards: Dict[int, Card] = {}  # player_id -> card
        self.play_order: List[int] = []
        self.winner_id: Optional[int] = None
        self.led_suit: Optional[Suit] = None
    
    def add_card(self, player_id: int, card: Card):
        """Add a card played by a player."""
        if player_id not in self.cards:
            self.play_order.append(player_id)
        self.cards[player_id] = card
        
        # First card determines led suit
        if self.led_suit is None:
            self.led_suit = card.suit
    
    def is_complete(self) -> bool:
        """Check if all 4 players have played."""
        return len(self.cards) == 4
    
    def determine_winner(self, trump_suit: Optional[Suit]) -> int:
        """
        Determine winner of the trick using proper card comparison.
        
        Args:
            trump_suit: Current trump suit (None for No Trump)
            
        Returns:
            Player ID of trick winner
        """
        if not self.is_complete():
            raise ValueError("Cannot determine winner of incomplete trick")
        
        winning_card = None
        winner_id = None
        
        for player_id in self.play_order:
            card = self.cards[player_id]
            
            if winning_card is None or card.beats(winning_card, trump_suit, self.led_suit):
                winning_card = card
                winner_id = player_id
        
        self.winner_id = winner_id
        return winner_id


class Round:
    """Manages a complete round of Estimation with proper scoring."""
    
    def __init__(self, round_number: int, trump_suit: Optional[Suit], players: List[Player]):
        self.round_number = round_number
        self.trump_suit = trump_suit
        self.players = {p.player_id: p for p in players}
        self.tricks: List[Trick] = []
        self.current_trick: Optional[Trick] = None
        self.leader_id = 0  # First player leads first trick
        self.bids: Dict[int, int] = {}
        self.estimations: Dict[int, int] = {}
        self.declarer_id: Optional[int] = None
        self.declarer_bid: Optional[int] = None
        
    def set_bid(self, player_id: int, bid: int):
        """Set a player's bid."""
        if 0 <= bid <= 13:
            self.bids[player_id] = bid
            self.players[player_id].bid = bid
    
    def set_estimation(self, player_id: int, estimation: int):
        """Set a player's estimation."""
        if 0 <= estimation <= 13:
            self.estimations[player_id] = estimation
            self.players[player_id].estimation = estimation
    
    def start_trick(self, leader_id: int):
        """Start a new trick with specified leader."""
        self.current_trick = Trick(leader_id)
        self.leader_id = leader_id
    
    def play_card(self, player_id: int, card: Card) -> bool:
        """
        Play a card to the current trick.
        
        Args:
            player_id: ID of player playing card
            card: Card being played
            
        Returns:
            True if trick is complete after this play
        """
        if self.current_trick is None:
            raise ValueError("No active trick")
        
        # Validate play
        player = self.players[player_id]
        valid_plays = player.get_valid_plays(self.current_trick.led_suit)
        
        if card not in valid_plays:
            raise ValueError(f"Invalid play: {card} not in valid plays {valid_plays}")
        
        # Play the card
        player.play_card(card)
        self.current_trick.add_card(player_id, card)
        
        # Check if trick is complete
        if self.current_trick.is_complete():
            winner_id = self.current_trick.determine_winner(self.trump_suit)
            self.players[winner_id].tricks_won += 1
            
            # Add to completed tricks
            self.tricks.append(self.current_trick)
            self.current_trick = None
            self.leader_id = winner_id
            
            return True
        
        return False
    
    def is_complete(self) -> bool:
        """Check if round is complete (all cards played)."""
        return len(self.tricks) == 13 and self.current_trick is None
    
    def get_next_player(self, current_player_id: int) -> int:
        """Get ID of next player in turn order."""
        return (current_player_id + 1) % len(self.players)
    
    def get_play_order(self, leader_id: int) -> List[int]:
        """Get play order starting from leader."""
        order = []
        current = leader_id
        for _ in range(len(self.players)):
            order.append(current)
            current = self.get_next_player(current)
        return order
    
    def calculate_scores(self) -> Dict[int, int]:
        """
        Calculate round scores based on Estimation rules.
        
        Returns:
            Dictionary of player_id -> points earned
        """
        scores = {}
        
        # Get all estimations and actual tricks
        all_estimations = []
        all_tricks = []
        
        for player_id in range(4):
            player = self.players[player_id]
            estimation = self.estimations.get(player_id, 0)
            tricks = player.tricks_won
            
            all_estimations.append(estimation)
            all_tricks.append(tricks)
        
        # Check for Risk situation (total estimations = 13)
        total_estimations = sum(all_estimations)
        is_risk = (total_estimations == 13)
        risk_level = 1 if is_risk else 0
        
        # Calculate scores for each player
        for player_id in range(4):
            player = self.players[player_id]
            estimation = all_estimations[player_id]
            tricks = all_tricks[player_id]
            
            # Base scoring
            if tricks == estimation:
                # Made estimation exactly
                points = 10 + estimation
                
                # Bonuses for making estimation
                if estimation == 0:  # Dash call
                    if self.round_number <= 13:  # Under rounds
                        points = 33
                    else:  # Over rounds  
                        points = 25
                
                # Risk bonus
                if is_risk:
                    points += 10 * risk_level
                
                # Check for sole winner (only player to make estimation)
                others_made = sum(1 for i in range(4) if i != player_id and 
                                all_tricks[i] == all_estimations[i])
                if others_made == 0:
                    points += 10
                    
            else:
                # Missed estimation
                difference = abs(tricks - estimation)
                points = -10 - difference
                
                # Risk penalty
                if is_risk:
                    points -= 10 * risk_level
                
                # Check for sole loser (only player to miss estimation)  
                others_missed = sum(1 for i in range(4) if i != player_id and 
                                  all_tricks[i] != all_estimations[i])
                if others_missed == 0:
                    points -= 10
                
                # Dash penalty
                if estimation == 0:
                    if self.round_number <= 13:  # Under rounds
                        points = -33
                    else:  # Over rounds
                        points = -25
            
            scores[player_id] = points
            
        return scores
    
    def get_status(self) -> Dict:
        """Get current round status for display."""
        return {
            'round_number': self.round_number,
            'trump_suit': str(self.trump_suit) if self.trump_suit else "No Trump",
            'tricks_completed': len(self.tricks),
            'estimations': self.estimations,
            'tricks_won': {pid: p.tricks_won for pid, p in self.players.items()},
            'current_leader': self.leader_id,
            'is_complete': self.is_complete(),
            'declarer_id': self.declarer_id,
            'declarer_bid': self.declarer_bid
        }