"""
Main game module for Estimation card game.
Manages overall game state, rounds, and progression.
"""

from typing import List, Dict, Optional, Tuple
from estimation_bot.card import Suit
from estimation_bot.deck import Deck
from estimation_bot.player import Player, BotInterface
from estimation_bot.round import Round
from estimation_bot.utils import GameLogger
from estimation_bot.player import Player, BotInterface, HumanPlayer
import json
import random
import os
from datetime import datetime


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
        self.logger = GameLogger()
        self.sa_aydeh_multiplier = 1  # Multiplier for failed rounds

        # Game data for comprehensive logging
        self.game_data = {
            'game_mode': game_mode,
            'players': [p.name for p in players],
            'rounds': [],
            'final_scores': {},
            'winner': None
        }
        
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
        is_speed = self.is_speed_round(self.current_round + 1)
        if is_speed:
            trump_suit = self.get_speed_round_trump(self.current_round + 1)
        else:
            trump_suit = None  # Will be determined by bidding
        
        # Create new round
        round_obj = Round(self.current_round + 1, trump_suit, self.players, is_speed)
        
        self.rounds.append(round_obj)
        self.current_round += 1
        
        return round_obj
    
    def collect_bids_and_estimations(self, round_obj: Round):
        """Collect bids and estimations from all players according to proper rules."""
        if round_obj.is_speed_round:
            # Speed rounds: No bidding, straight to estimation
            self._collect_estimations_speed_round(round_obj)
        else:
            # Normal rounds: Bidding phase then estimation phase
            self._collect_bids_normal_round(round_obj)
            self._collect_estimations_normal_round(round_obj)
        
        # Log the round type (Over/Under)
        total_estimations = sum(v for v in round_obj.estimations.values() if v is not None)
        round_type = "Over" if total_estimations > 13 else "Under" if total_estimations < 13 else "Exact"
        
        print(f"\n=== Round {round_obj.round_number} Setup Complete ===")
        print(f"Trump: {round_obj.trump_suit or 'No Trump'}")
        print(f"Round Type: {round_type} ({total_estimations} total tricks estimated)")
        
        if not round_obj.is_speed_round:
            declarer_name = self.players[round_obj.declarer_id].name if round_obj.declarer_id is not None else "None"
            print(f"Declarer: {declarer_name}")
        
        # Show all estimations
        print("Final Estimations:")
        for i, player in enumerate(self.players):
            estimation = round_obj.estimations.get(i, 0)
            tags = []
            if i == round_obj.declarer_id:
                tags.append("CALL")
            if round_obj.with_players and i in round_obj.with_players:
                tags.append("WITH")
            if round_obj.dash_players and i in round_obj.dash_players:
                tags.append("DASH")
            tag_str = f" ({', '.join(tags)})" if tags else ""
            print(f"  {player.name}: {estimation}{tag_str}")
        print()
    
    def _collect_bids_normal_round(self, round_obj: Round):
        """Collect bids for normal rounds with proper DASH logic."""
        print(f"\n=== BIDDING PHASE - Round {round_obj.round_number} ===")
        
        # Check for Avoid declarations first
        for i, player in enumerate(self.players):
            missing_suits = player.declare_avoid()
            if missing_suits:
                print(f"{player.name} declares AVOID: {', '.join(suit.name for suit in missing_suits)}")
        
        # First, ask all players about DASH intentions
        dash_intentions = {}
        
        print("\n--- DASH CALL PHASE ---")
        dash_count = 0
        for player_id in range(4):
            player = self.players[player_id]
            
            # Show hand to player before dash decision
            if isinstance(player, HumanPlayer):
                print(f"\n{player.name}'s hand: {player._format_hand()}")
            
            can_dash = dash_count < 2  # Only allow 2 dash calls maximum
            
            if hasattr(player, 'strategy') and isinstance(player.strategy, BotInterface):
                # Bot decides on dash
                dash_choice = random.choice([True, False]) if can_dash and hasattr(player.strategy, 'make_bid') else False
            elif hasattr(player, 'make_dash_choice') and can_dash:
                dash_choice = player.make_dash_choice()
            else:
                # Default: no dash or can't dash
                dash_choice = False
                
            dash_intentions[player_id] = dash_choice
            if dash_choice:
                print(f"{player.name} chooses DASH CALL (0 tricks)")
                round_obj.set_dash_call(player_id)
                dash_count += 1
            else:
                if not can_dash and hasattr(player, 'make_dash_choice'):
                    print(f"{player.name} cannot DASH (2 players already called DASH)")
                else:
                    print(f"{player.name} chooses to bid normally")
        
        # Now regular bidding for non-dash players
        bidding_order = list(range(4))  # 0, 1, 2, 3
        print("\n--- REGULAR BIDDING PHASE ---")
        for player_id in bidding_order:
            if dash_intentions[player_id]:
                continue  # Skip dash players
                
            player = self.players[player_id]
            
            if hasattr(player, 'strategy') and isinstance(player.strategy, BotInterface):
                # Bot player
                bid_result = player.strategy.make_bid(
                    player.hand, round_obj.other_bids, False, False  # No more dash calls
                )
            elif hasattr(player, 'make_bid_interactive'):
                # Human player
                bid_result = player.make_bid_interactive(round_obj.other_bids, False)
            else:
                # Default pass
                bid_result = None
            
            if bid_result is None:
                print(f"{player.name} passes")
                round_obj.other_bids[player_id] = None
            else:
                # Normal bid (amount, trump_suit)
                amount, trump_suit = bid_result
                print(f"{player.name} bids {amount} {trump_suit.name if trump_suit else 'No Trump'}")
                round_obj.set_bid(player_id, amount, trump_suit)
        
        # Determine winner
        round_obj.determine_declarer()
        
        if round_obj.declarer_id is None:
            print("All players passed! This becomes a Sa'aydeh round (doubled scoring).")
            self.sa_aydeh_multiplier *= 2
            print(f"Next successful round will have {self.sa_aydeh_multiplier}x scoring multiplier")
            
            # Reset and start a completely new round
            self.current_round -= 1  # Don't count the failed round
            return "RESTART_ROUND"  # Signal to restart
    
    def _collect_estimations_normal_round(self, round_obj: Round):
        """Collect estimations for normal rounds after declarer is determined."""
        if round_obj.declarer_id is None:
            return
        
        print(f"\n=== ESTIMATION PHASE - Round {round_obj.round_number} ===")
        declarer = self.players[round_obj.declarer_id]
        print(f"Declarer: {declarer.name} with {round_obj.declarer_bid} {round_obj.trump_suit.name if round_obj.trump_suit else 'No Trump'}")
        
        # Set declarer's estimation to their bid
        round_obj.estimations[round_obj.declarer_id] = round_obj.declarer_bid

        # Set the declarer as the leader for first trick
        round_obj.leader_id = round_obj.declarer_id
        
        # Other players estimate
        estimation_order = [i for i in range(4) if i != round_obj.declarer_id]
        dash_estimations_made = len(round_obj.dash_players) if round_obj.dash_players else 0
        
        for i, player_id in enumerate(estimation_order):
            player = self.players[player_id]
            is_last_estimator = i == len(estimation_order) - 1
            can_dash = dash_estimations_made < 2
            
            if player_id in round_obj.dash_players:
                # Already made dash call
                round_obj.estimations[player_id] = 0
                continue
            
            if hasattr(player, 'strategy') and isinstance(player.strategy, BotInterface):
                # Bot player
                estimation = player.strategy.make_estimation(
                    player.hand, round_obj.trump_suit, round_obj.declarer_bid,
                    list(round_obj.estimations.values()), is_last_estimator, can_dash
                )
            elif hasattr(player, 'make_estimation_interactive'):
                # Human player
                estimation = player.make_estimation_interactive(
                    round_obj.trump_suit, round_obj.declarer_bid,
                    list(round_obj.estimations.values()), is_last_estimator, can_dash
                )
            else:
                # Default estimation
                estimation = min(3, round_obj.declarer_bid)
            
            # Handle special estimations
            if estimation == "DASH":
                if can_dash:
                    estimation = 0
                    round_obj.dash_players.add(player_id)
                    dash_estimations_made += 1
                    print(f"{player.name} estimates DASH (0 tricks)")
                else:
                    estimation = 1  # Force to 1 if can't dash
                    print(f"{player.name} cannot DASH (limit reached), estimates {estimation}")
            elif estimation == round_obj.declarer_bid:
                # "With" call
                round_obj.with_players.add(player_id)
                print(f"{player.name} estimates WITH ({estimation} tricks)")
            else:
                print(f"{player.name} estimates {estimation} tricks")
            
            # Validate estimation
            if is_last_estimator:
                # Risk player - cannot make total = 13
                    current_total = sum(v for v in round_obj.estimations.values() if v is not None) + (estimation if estimation is not None else 0)
                    if estimation > 0:
                        estimation -= 1
                    else:
                        estimation += 1
                    print(f"  Adjusted to {estimation} to avoid Risk (total would be 13)")
            
            round_obj.estimations[player_id] = estimation
    
    def _collect_estimations_speed_round(self, round_obj: Round):
        """Collect estimations for speed rounds (no bidding phase)."""
        print(f"\n=== SPEED ROUND {round_obj.round_number} ===")
        print(f"Trump: {round_obj.trump_suit.name if round_obj.trump_suit else 'No Trump'}")
        
        dash_estimations_made = 0
        
        for i, player in enumerate(self.players):
            is_last_estimator = i == 3
            can_dash = dash_estimations_made < 2
            
            if hasattr(player, 'strategy') and isinstance(player.strategy, BotInterface):
                # Bot player
                estimation = player.strategy.make_estimation(
                    player.hand, round_obj.trump_suit, 13,  # No declarer bid in speed rounds
                    list(round_obj.estimations.values()), is_last_estimator, can_dash
                )
            elif hasattr(player, 'make_estimation_interactive'):
                # Human player
                estimation = player.make_estimation_interactive(
                    round_obj.trump_suit, 13, list(round_obj.estimations.values()),
                    is_last_estimator, can_dash
                )
            else:
                # Default estimation
                estimation = 3
            
            # Handle special estimations
            if estimation == "DASH":
                if can_dash:
                    estimation = 0
                    round_obj.dash_players.add(i)
                    dash_estimations_made += 1
                    print(f"{player.name} estimates DASH (0 tricks)")
                else:
                    estimation = 1  # Force to 1 if can't dash
                    print(f"{player.name} cannot DASH (limit reached), estimates {estimation}")
            else:
                print(f"{player.name} estimates {estimation} tricks")
            
            # Validate estimation for last player (Risk)
            if is_last_estimator:
                current_total = sum(round_obj.estimations.values()) + estimation
                if current_total == 13:
                    if estimation > 0:
                        estimation -= 1
                    else:
                        estimation += 1
                    print(f"  Adjusted to {estimation} to avoid Risk (total would be 13)")
            
            round_obj.estimations[i] = estimation
        
        # Determine declarer as highest estimator
        highest_estimation = max(round_obj.estimations.values())
        for player_id, estimation in round_obj.estimations.items():
            if estimation == highest_estimation:
                round_obj.declarer_id = player_id
                round_obj.declarer_bid = estimation
                break
    
    def play_trick(self, round_obj: Round, trick_number: int) -> int:
        """
        Play a single trick with enhanced display.
        
        Returns:
            Winner's player ID
        """
        print(f"\n--- Trick {trick_number} ---")
        
        # Show current trick counts
        print("Current Tricks Won:")
        for i, player in enumerate(self.players):
            estimation = round_obj.estimations.get(i, 0)
            print(f"  {player.name}: {player.tricks_won}/{estimation}")
        
        round_obj.start_trick(round_obj.leader_id)
        play_order = round_obj.get_play_order(round_obj.leader_id)
        
        cards_on_table = []  # Track cards played in order
        
        for player_id in play_order:
            player = self.players[player_id]
            valid_plays = player.get_valid_plays(round_obj.current_trick.led_suit)
            
            # Show current table state
            if cards_on_table:
                print(f"Cards on table: {', '.join(cards_on_table)}")
            
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
                    round_obj.current_trick.led_suit, cards_on_table
                )
            else:
                # Default to first valid card
                card = valid_plays[0]
            
            # Play the card
            is_complete = round_obj.play_card(player_id, card)
            cards_on_table.append(f"{player.name}: {card}")
            print(f"{player.name} plays {card}")
            
            if is_complete:
                winner_id = round_obj.leader_id
                winner_name = self.players[winner_id].name
                print(f">>> {winner_name} wins the trick! <<<")
                return winner_id
        
        return round_obj.leader_id
    
    def play_round(self) -> Dict[int, int]:
        """Play a complete round."""
        while True:
            round_obj = self.start_new_round()
            
            # Collect bids and estimations
            result = self.collect_bids_and_estimations(round_obj)
            
            if result == "RESTART_ROUND":
                continue  # Restart with a new round
            
            # Play all 13 tricks
            for trick_num in range(1, 14):
                self.play_trick(round_obj, trick_num)
            
            # Calculate and apply scores
            round_scores = round_obj.calculate_scores()
            
            # Apply Sa'aydeh multiplier if active
            if self.sa_aydeh_multiplier > 1:
                print(f"Applying Sa'aydeh {self.sa_aydeh_multiplier}x multiplier to scores!")
                for player_id in round_scores:
                    round_scores[player_id] *= self.sa_aydeh_multiplier
                self.sa_aydeh_multiplier = 1  # Reset multiplier after use
            
            for player_id, score in round_scores.items():
                self.players[player_id].add_score(score)
            
            # Log round completion
            self._log_round_completion(round_obj, round_scores)
            
            return round_scores
    
    def _log_round_completion(self, round_obj: Round, round_scores: Dict[int, int]):
        """Log detailed round completion data."""
        print(f"\n=== Round {round_obj.round_number} Complete ===")
        
        # Show final trick counts vs estimations
        print("Final Results:")
        for i, player in enumerate(self.players):
            estimation = round_obj.estimations.get(i, 0)
            actual = player.tricks_won
            score = round_scores.get(i, 0)
            status = "✓" if actual == estimation else "✗"
            
            tags = []
            if i == round_obj.declarer_id:
                tags.append("CALL")
            if round_obj.with_players and i in round_obj.with_players:
                tags.append("WITH")
            if round_obj.dash_players and i in round_obj.dash_players:
                tags.append("DASH")
            tag_str = f" ({', '.join(tags)})" if tags else ""
            
            print(f"  {player.name}: {actual}/{estimation} {status} -> {score:+d} points{tag_str}")
        
        # Store round data
        round_data = {
            'round_number': round_obj.round_number,
            'trump_suit': str(round_obj.trump_suit) if round_obj.trump_suit else 'No Trump',
            'is_speed_round': round_obj.is_speed_round,
            'declarer_id': round_obj.declarer_id,
            'declarer_name': self.players[round_obj.declarer_id].name if round_obj.declarer_id is not None else None,
            'estimations': dict(round_obj.estimations),
            'actual_tricks': {i: p.tricks_won for i, p in enumerate(self.players)},
            'scores': round_scores,
            'with_players': list(round_obj.with_players) if round_obj.with_players else [],
            'dash_players': list(round_obj.dash_players) if round_obj.dash_players else [],
            'round_type': 'Over' if sum(round_obj.estimations.values()) > 13 else 'Under' if sum(round_obj.estimations.values()) < 13 else 'Exact'
        }
        
        self.game_data['rounds'].append(round_data)
        
        print(f"Current Scores: {', '.join(f'{p.name}: {p.score}' for p in self.players)}")
        print()
    
    def play_game(self) -> Dict[int, int]:
        """
        Play a complete game.
        
        Returns:
            Final scores dictionary
        """
        self.logger.log_game_start(self.game_mode, [p.name for p in self.players])
        
        for _ in range(self.total_rounds):
            self.play_round()
        
        self.game_complete = True
        final_scores = self.get_final_scores()
        
        # Complete game data
        self.game_data['final_scores'] = {p.name: p.score for p in self.players}
        winner_id, winning_score = self.get_winner()
        self.game_data['winner'] = self.players[winner_id].name
        
        # Display final results
        self._display_final_results()
        
        # Save game data
        self._save_game_data()
        
        return final_scores
    
    def _display_final_results(self):
        """Display comprehensive final game results."""
        print("\n" + "="*60)
        print("FINAL GAME RESULTS")
        print("="*60)
        
        # Final standings
        sorted_players = sorted(self.players, key=lambda p: p.score, reverse=True)
        print("\nFinal Standings:")
        for i, player in enumerate(sorted_players, 1):
            print(f"{i}. {player.name}: {player.score} points")
        
        print(f"\n🏆 Winner: {sorted_players[0].name} 🏆")
        
        # Round-by-round breakdown
        print(f"\nRound-by-Round Breakdown:")
        print(f"{'Round':<6} {'Trump':<12} {'Type':<6} " + " ".join(f"{p.name:<12}" for p in self.players))
        print("-" * (30 + 13 * len(self.players)))
        
        for round_data in self.game_data['rounds']:
            round_num = round_data['round_number']
            trump = round_data['trump_suit'][:10] if len(round_data['trump_suit']) > 10 else round_data['trump_suit']
            round_type = round_data['round_type'][:6]
            
            score_str = " ".join(f"{round_data['scores'].get(i, 0):+12d}" for i in range(4))
            print(f"{round_num:<6} {trump:<12} {round_type:<6} {score_str}")
        
        print("-" * (30 + 13 * len(self.players)))
        total_str = " ".join(f"{p.score:+12d}" for p in self.players)
        print(f"{'TOTAL':<24} {total_str}")
    
    def _save_game_data(self):
        """Save comprehensive game data to file."""
        
        # Create directory if it doesn't exist
        dump_dir = "estimation_bot/game_dump"
        os.makedirs(dump_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        filename = f"{dump_dir}/estimation_game_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.game_data, f, indent=2)
        
        print(f"\nGame data saved to: {filename}")
    
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
        self.game_data = {
            'game_mode': self.game_mode,
            'players': [p.name for p in self.players],
            'rounds': [],
            'final_scores': {},
            'winner': None
        }
        for player in self.players:
            player.score = 0
            player.reset_round()