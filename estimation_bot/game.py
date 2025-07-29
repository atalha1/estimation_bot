"""
Main game module for Estimation card game.
Manages overall game state, rounds, and progression.
"""

from typing import List, Dict, Optional, Tuple
from estimation_bot.card import Suit
from estimation_bot.deck import Deck
from estimation_bot.player import Player, BotInterface
from estimation_bot.round import Round


class EstimationGame:
    """Main game controller for Estimation."""
    
    def __init__(self, players: List[Player], game_mode: str = "FULL"):
        if len(players) != 4:
            raise ValueError("Estimation requires exactly 4 players")
        
        self.players = players
        self.deck = Deck()
        self.current_round = 0
        self.rounds: List[Round] = []
        self.game_complete = False
        self.game_mode = game_mode
        self.total_rounds = self._get_total_rounds()
        self.multiplier = 1  # For Sa'aydeh (doubled) rounds
        
    def _get_total_rounds(self) -> int:
        """Get total rounds based on game mode."""
        modes = {
            "FULL": 18,
            "MINI": 10, 
            "MICRO": 5
        }
        return modes.get(self.game_mode, 18)
    
    def is_speed_round(self, round_number: int) -> bool:
        """Check if round is a speed round."""
        if self.game_mode == "MICRO":
            return False
        speed_start = 14 if self.game_mode == "FULL" else 6
        return round_number >= speed_start

    def get_trump_suit(self, round_number):
        """
        Determines the trump suit for a given round.
        For now, let's assume the trump suit is always randomly chosen from suits.
        You can later replace this logic with actual Estimation rules.
        """
        import random
        return random.choice(['S', 'H', 'D', 'C'])  # Spades, Hearts, Diamonds, Clubs
    
    def get_speed_round_trump(self, round_number: int) -> Optional[Suit]:
        """Get predetermined trump for speed rounds."""
        if self.game_mode == "FULL":
            trump_map = {14: None, 15: Suit.SPADES, 16: Suit.HEARTS, 
                        17: Suit.DIAMONDS, 18: Suit.CLUBS}
        else:  # MINI
            trump_map = {6: None, 7: Suit.SPADES, 8: Suit.HEARTS,
                        9: Suit.DIAMONDS, 10: Suit.CLUBS}
        return trump_map.get(round_number)
    
    def start_new_round(self) -> Round:
        """Start a new round of the game."""
        if self.current_round >= 13:
            raise ValueError("Game already complete (13 rounds)")
        
        # Reset deck and deal cards
        self.deck.reset()
        hands = self.deck.deal_round(4)
        
        # Reset players and give them cards
        for i, player in enumerate(self.players):
            player.reset_round()
            player.receive_cards(hands[i])
        
        # Create new round
        trump_suit = self.get_trump_suit(self.current_round)
        round_obj = Round(self.current_round + 1, trump_suit, self.players)
        
        self.rounds.append(round_obj)
        self.current_round += 1
        
        return round_obj
    
    def collect_bids(self, round_obj: Round):
        """Collect bids from all players."""
        other_bids = [None] * 4
        
        # Each player bids in order
        for i in range(4):
            player = self.players[i]
            
            if hasattr(player, 'strategy') and isinstance(player.strategy, BotInterface):
                # Bot player
                bid = player.strategy.make_bid(
                    player.hand, round_obj.trump_suit, other_bids
                )
            elif hasattr(player, 'make_bid_interactive'):
                # Human player
                bid = player.make_bid_interactive(round_obj.trump_suit, other_bids)
            else:
                # Default to 0 bid
                bid = 0
            
            round_obj.set_bid(i, bid)
            other_bids[i] = bid
    
    def play_trick(self, round_obj: Round) -> int:
        """
        Play a single trick.
        
        Returns:
            Winner's player ID
        """
        round_obj.start_trick(round_obj.leader_id)
        play_order = round_obj.get_play_order(round_obj.leader_id)
        
        for player_id in play_order:
            player = self.players[player_id]
            valid_plays = player.get_valid_plays(round_obj.current_trick.led_suit)
            
            if hasattr(player, 'strategy') and isinstance(player.strategy, BotInterface):
                # Bot player
                card = player.strategy.choose_card(
                    player.hand, valid_plays, round_obj.trump_suit,
                    round_obj.current_trick.led_suit,
                    list(round_obj.current_trick.cards.values())
                )
            elif hasattr(player, 'choose_card_interactive'):
                # Human player
                card = player.choose_card_interactive(
                    valid_plays, round_obj.trump_suit, 
                    round_obj.current_trick.led_suit
                )
            else:
                # Default to first valid card
                card = valid_plays[0]
            
            is_complete = round_obj.play_card(player_id, card)
            
            if is_complete:
                return round_obj.leader_id
        
        return round_obj.leader_id
    
    def play_round(self) -> Dict[int, int]:
        """
        Play a complete round.
        
        Returns:
            Dictionary of player scores for this round
        """
        round_obj = self.start_new_round()
        
        # Collect bids
        self.collect_bids(round_obj)
        
        # Play all 13 tricks
        for _ in range(13):
            self.play_trick(round_obj)
        
        # Calculate and apply scores
        round_scores = round_obj.calculate_scores()
        for player_id, score in round_scores.items():
            self.players[player_id].add_score(score)
        
        return round_scores
    
    def play_game(self) -> Dict[int, int]:
        """
        Play a complete game (13 rounds).
        
        Returns:
            Final scores dictionary
        """
        for _ in range(13):
            self.play_round()
        
        self.game_complete = True
        return self.get_final_scores()
    
    def get_final_scores(self) -> Dict[int, int]:
        """Get final scores for all players."""
        return {i: player.score for i, player in enumerate(self.players)}
    
    def get_winner(self) -> Tuple[int, int]:
        """
        Get game winner.
        
        Returns:
            Tuple of (player_id, score)
        """
        if not self.game_complete:
            raise ValueError("Game not yet complete")
        
        scores = self.get_final_scores()
        winner_id = max(scores.keys(), key=lambda x: scores[x])
        return winner_id, scores[winner_id]
    
    def get_game_state(self) -> Dict:
        """Get current game state for logging/display."""
        return {
            'current_round': self.current_round,
            'game_complete': self.game_complete,
            'player_scores': {i: p.score for i, p in enumerate(self.players)},
            'rounds_played': len(self.rounds),
            'current_round_state': self.rounds[-1].get_status() if self.rounds else None
        }
    
    def reset_game(self):
        """Reset game state for new game."""
        self.current_round = 0
        self.rounds = []
        self.game_complete = False
        for player in self.players:
            player.score = 0
            player.reset_round()