"""
Round module for Estimation card game.
Handles trick resolution and round progression with proper bidding flow.
"""

from typing import List, Dict, Optional, Tuple, Set
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
        Determine winner of the trick.
        
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
    """Manages a complete round of Estimation with proper bidding flow."""
    
    def __init__(self, round_number: int, trump_suit: Optional[Suit], players: List[Player], is_speed_round: bool = False):
        self.round_number = round_number
        self.trump_suit = trump_suit  # For speed rounds, this is predetermined
        self.players = {p.player_id: p for p in players}
        self.is_speed_round = is_speed_round
        
        self.tricks: List[Trick] = []
        self.current_trick: Optional[Trick] = None
        self.leader_id = 0  # First player leads first trick
        
        # Bidding phase data
        self.other_bids: Dict[int, Optional[Tuple[int, Optional[Suit]]]] = {i: None for i in range(4)}
        self.dash_players: Set[int] = set()  # Players who made dash calls
        
        # After declarer is determined
        self.declarer_id: Optional[int] = None
        self.declarer_bid: Optional[int] = None
        
        # Estimation phase data
        self.estimations: Dict[int, int] = {}
        self.with_players: Set[int] = set()  # Players who bid "with"
        
    def set_bid(self, player_id: int, amount: int, trump_suit: Optional[Suit]):
        """Set a player's bid during bidding phase."""
        if 4 <= amount <= 13:  # Valid bid range
            self.other_bids[player_id] = (amount, trump_suit)
            self.players[player_id].bid = amount
            self.players[player_id].trump_suit = trump_suit
    
    def set_dash_call(self, player_id: int):
        """Set a player's dash call."""
        self.dash_players.add(player_id)
        self.other_bids[player_id] = "DASH"
        self.players[player_id].is_dash = True
    
    def determine_declarer(self):
        """Determine the declarer (highest bidder) after bidding phase."""
        if self.is_speed_round:
            return  # No declarer in speed rounds, handled differently
        
        highest_bid = 0
        highest_trump_rank = 0
        declarer_candidates = []
        
        # Find highest bid(s)
        for player_id, bid_data in self.other_bids.items():
            if bid_data and bid_data != "DASH":
                amount, trump_suit = bid_data
                if amount > highest_bid:
                    highest_bid = amount
                    declarer_candidates = [(player_id, trump_suit)]
                elif amount == highest_bid:
                    declarer_candidates.append((player_id, trump_suit))
        
        if not declarer_candidates:
            # All passed or only dash calls
            self.declarer_id = None
            return
        
        # If tied, use trump suit ranking
        if len(declarer_candidates) > 1:
            # Trump suit rankings: No Trump > Spades > Hearts > Diamonds > Clubs
            suit_ranks = {None: 5, Suit.SPADES: 4, Suit.HEARTS: 3, Suit.DIAMONDS: 2, Suit.CLUBS: 1}
            
            best_candidate = max(declarer_candidates, key=lambda x: suit_ranks.get(x[1], 0))
            self.declarer_id, self.trump_suit = best_candidate
        else:
            self.declarer_id, self.trump_suit = declarer_candidates[0]
        
        self.declarer_bid = highest_bid
        self.players[self.declarer_id].is_declarer = True
    
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
            raise ValueError(f"Invalid play: {card}")
        
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
        Calculate round scores based on bids vs tricks won with full Estimation rules.
        
        Returns:
            Dictionary of player_id -> points earned
        """
        scores = {}
        total_estimations = sum(self.estimations.values())
        
        # Determine round type
        is_over_round = total_estimations > 13
        is_under_round = total_estimations < 13
        
        # Calculate risk level
        risk_level = abs(total_estimations - 13) // 2
        
        # Count winners and losers
        winners = []
        losers = []
        
        for player_id, player in self.players.items():
            estimation = self.estimations.get(player_id, 0)
            actual_tricks = player.tricks_won
            
            if actual_tricks == estimation:
                winners.append(player_id)
            else:
                losers.append(player_id)
        
        # Calculate scores for each player
        for player_id, player in self.players.items():
            estimation = self.estimations.get(player_id, 0)
            actual_tricks = player.tricks_won
            score = 0
            
            if actual_tricks == estimation:
                # Made estimation exactly
                score = 10 + estimation  # Base points
                
                # Call/With bonus
                if player_id == self.declarer_id or player_id in self.with_players:
                    score += 10
                
                # Sole winner bonus
                if len(winners) == 1:
                    score += 10
                
                # Risk bonus
                if risk_level > 0:
                    score += 10 * risk_level
                
                # Dash bonus
                if player_id in self.dash_players:
                    if is_over_round:
                        score = 25  # Override base calculation
                    elif is_under_round:
                        score = 33  # Override base calculation
                
            else:
                # Missed estimation
                difference = abs(actual_tricks - estimation)
                score = -difference  # Base penalty per trick difference
                
                # Additional penalties
                score -= 10  # Base penalty for missing
                
                # Call/With penalty
                if player_id == self.declarer_id or player_id in self.with_players:
                    score -= 10
                
                # Sole loser penalty
                if len(losers) == 1:
                    score -= 10
                
                # Risk penalty
                if risk_level > 0:
                    score -= 10 * risk_level
                
                # Dash penalty
                if player_id in self.dash_players:
                    if is_over_round:
                        score = -25  # Override base calculation
                    elif is_under_round:
                        score = -33  # Override base calculation
            
            scores[player_id] = score
        
        return scores
    
    def get_status(self) -> Dict:
        """Get current round status for display."""
        return {
            'round_number': self.round_number,
            'trump_suit': str(self.trump_suit) if self.trump_suit else 'No Trump',
            'is_speed_round': self.is_speed_round,
            'tricks_completed': len(self.tricks),
            'declarer_id': self.declarer_id,
            'declarer_bid': self.declarer_bid,
            'estimations': dict(self.estimations),
            'tricks_won': {pid: p.tricks_won for pid, p in self.players.items()},
            'current_leader': self.leader_id,
            'is_complete': self.is_complete(),
            'with_players': list(self.with_players),
            'dash_players': list(self.dash_players)
        }