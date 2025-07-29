"""
Main game module for Estimation card game.
Manages overall game state, rounds, and progression with proper rule enforcement.
"""

from typing import List, Dict, Optional, Tuple
from estimation_bot.card import Suit
from estimation_bot.deck import Deck
from estimation_bot.player import Player, BotInterface
from estimation_bot.round import Round


class EstimationGame:
    """Main game controller for Estimation with proper rule enforcement."""
    
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
        if self.current_round >= self.total_rounds:
            raise ValueError(f"Game already complete ({self.total_rounds} rounds)")
        
        # Reset deck and deal cards
        self.deck.reset()
        hands = self.deck.deal_round(4)
        
        # Reset players and give them cards
        for i, player in enumerate(self.players):
            player.reset_round()
            player.receive_cards(hands[i])
        
        # Determine trump suit
        round_number = self.current_round + 1
        if self.is_speed_round(round_number):
            trump_suit = self.get_speed_round_trump(round_number)
        else:
            trump_suit = None  # Will be determined by bidding
        
        # Create new round
        round_obj = Round(round_number, trump_suit, self.players)
        
        self.rounds.append(round_obj)
        self.current_round += 1
        
        return round_obj
    
    def conduct_bidding_phase(self, round_obj: Round) -> Tuple[int, int, Optional[Suit]]:
        """
        Conduct the bidding phase and return winner info.
        
        Returns:
            Tuple of (winner_player_id, winning_bid, trump_suit)
        """
        bids = [None] * 4  # Store (amount, trump_suit) tuples
        
        # Each player bids in order
        for i in range(4):
            player = self.players[i]
            
            if hasattr(player, 'strategy') and isinstance(player.strategy, BotInterface):
                # Bot player
                bid = player.strategy.make_bid(
                    player.hand, bids, self.is_speed_round(round_obj.round_number)
                )
            elif hasattr(player, 'make_bid_interactive'):
                # Human player
                bid = player.make_bid_interactive(bids)
            else:
                # Default to pass
                bid = None
            
            bids[i] = bid
        
        # Find winning bid
        winning_bid = 0
        winner_id = -1
        trump_suit = None
        
        for i, bid in enumerate(bids):
            if bid is not None:
                bid_amount, bid_trump = bid
                if bid_amount > winning_bid:
                    winning_bid = bid_amount
                    winner_id = i
                    trump_suit = bid_trump
        
        if winner_id == -1:
            # All players passed - shouldn't happen in normal game
            # Default to player 0 with minimum bid
            winner_id = 0
            winning_bid = 4
            trump_suit = Suit.SPADES
        
        # Set the round's trump suit and declarer
        round_obj.trump_suit = trump_suit
        round_obj.declarer_id = winner_id
        round_obj.declarer_bid = winning_bid
        self.players[winner_id].is_declarer = True
        
        return winner_id, winning_bid, trump_suit
    
    def conduct_estimation_phase(self, round_obj: Round, declarer_id: int, declarer_bid: int):
        """Conduct the estimation phase for non-declarer players."""
        estimations = [None] * 4
        estimations[declarer_id] = declarer_bid  # Declarer's estimation equals their bid
        
        # Other players estimate in order
        estimation_order = [(declarer_id + i) % 4 for i in range(1, 4)]
        
        for i, player_id in enumerate(estimation_order):
            player = self.players[player_id]
            is_last = (i == 2)  # Last of the 3 estimators
            
            if hasattr(player, 'strategy') and isinstance(player.strategy, BotInterface):
                # Bot player
                estimation = player.strategy.make_estimation(
                    player.hand, round_obj.trump_suit, declarer_bid, 
                    estimations, is_last
                )
            elif hasattr(player, 'make_estimation_interactive'):
                # Human player
                estimation = player.make_estimation_interactive(
                    round_obj.trump_suit, declarer_bid, estimations, is_last
                )
            else:
                # Default estimation
                estimation = min(3, len(player.hand) // 4)
            
            estimations[player_id] = estimation
            round_obj.set_estimation(player_id, estimation)
    
    def play_trick(self, round_obj: Round) -> int:
        """
        Play a single trick with proper rule enforcement.
        
        Returns:
            Winner's player ID
        """
        round_obj.start_trick(round_obj.leader_id)
        play_order = round_obj.get_play_order(round_obj.leader_id)
        
        for player_id in play_order:
            player = self.players[player_id]
            valid_plays = player.get_valid_plays(round_obj.current_trick.led_suit)
            
            # Ensure valid plays are actually valid
            if not valid_plays:
                raise ValueError(f"Player {player_id} has no valid plays")
            
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
            
            # Validate the chosen card
            if card not in valid_plays:
                raise ValueError(f"Player {player_id} chose invalid card: {card}")
            
            is_complete = round_obj.play_card(player_id, card)
            
            if is_complete:
                return round_obj.leader_id
        
        return round_obj.leader_id
    
    def play_round(self) -> Dict[int, int]:
        """
        Play a complete round with proper phases.
        
        Returns:
            Dictionary of player scores for this round
        """
        round_obj = self.start_new_round()
        
        # Phase 1: Bidding (unless speed round)
        if not self.is_speed_round(round_obj.round_number):
            declarer_id, declarer_bid, trump_suit = self.conduct_bidding_phase(round_obj)
        else:
            # Speed round - no bidding, predetermined trump
            declarer_id = 0  # Default declarer for speed rounds
            declarer_bid = 7   # Default bid for speed rounds
            trump_suit = round_obj.trump_suit
            round_obj.declarer_id = declarer_id
            round_obj.declarer_bid = declarer_bid
        
        # Phase 2: Estimation
        if not self.is_speed_round(round_obj.round_number):
            self.conduct_estimation_phase(round_obj, declarer_id, declarer_bid)
        else:
            # Speed rounds might have different estimation rules
            # For now, use same estimation logic
            self.conduct_estimation_phase(round_obj, declarer_id, declarer_bid)
        
        # Phase 3: Play all 13 tricks
        for _ in range(13):
            self.play_trick(round_obj)
        
        # Phase 4: Calculate and apply scores
        round_scores = round_obj.calculate_scores()
        for player_id, score in round_scores.items():
            self.players[player_id].add_score(score)
        
        return round_scores
    
    def play_game(self) -> Dict[int, int]:
        """
        Play a complete game.
        
        Returns:
            Final scores dictionary
        """
        for _ in range(self.total_rounds):
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