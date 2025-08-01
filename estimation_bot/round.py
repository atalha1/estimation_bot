"""
Round module for Estimation card game.
Handles Trick resolution and round progression with proper bidding flow.
"""

from typing import List, Dict, Optional, Tuple, Set
from estimation_bot.card import Card, Suit
from estimation_bot.player import Player, HumanPlayer


class Trick:
    """Represents a single trick in the card game."""
    
    def __init__(self, leader_id: int):
        self.leader_id = leader_id
        self.cards: Dict[int, Card] = {}
        self.led_suit: Optional[Suit] = None
    
    def add_card(self, player_id: int, card: Card):
        """Add a card to the trick."""
        if len(self.cards) == 0:  # First card sets led suit
            self.led_suit = card.suit
        self.cards[player_id] = card
    
    def determine_winner(self, trump_suit: Optional[Suit]) -> int:
        """Determine winner based on led suit and trump rules."""
        winner_id = next(iter(self.cards))  # Default to first player
        winning_card = self.cards[winner_id]
        
        for player_id, card in self.cards.items():
            if self._is_better_card(
                card, 
                winning_card, 
                self.led_suit, 
                trump_suit
            ):
                winner_id = player_id
                winning_card = card
        
        return winner_id
    
    def _is_better_card(
        self, 
        candidate: Card, 
        current_winner: Card, 
        led_suit: Suit, 
        trump_suit: Optional[Suit]
    ) -> bool:
        """Compare two cards following trick rules."""
        # Trump always beats non-trump
        if candidate.suit == trump_suit and current_winner.suit != trump_suit:
            return True
        if current_winner.suit == trump_suit and candidate.suit != trump_suit:
            return False
        
        # Both trump: higher rank wins
        if candidate.suit == trump_suit and current_winner.suit == trump_suit:
            return candidate.rank > current_winner.rank
        
        # Both non-trump: must follow led suit
        if candidate.suit == led_suit and current_winner.suit != led_suit:
            return True
        if current_winner.suit == led_suit and candidate.suit != led_suit:
            return False
        
        # Same suit: higher rank wins
        if candidate.suit == current_winner.suit:
            return candidate.rank > current_winner.rank
        
        return False  # Candidate doesn't beat current winner

class Round:
    """Manages a complete round of Estimation with proper 5-phase structure."""
    
    def __init__(self, round_number: int, trump_suit: Optional[Suit], players: List[Player], is_speed_round: bool = False):
        self.round_number = round_number
        self.predetermined_trump = trump_suit  # For speed rounds only
        self.players = {p.player_id: p for p in players}
        self.is_speed_round = is_speed_round
        
        # Phase tracking
        self.current_phase = 1  # 1=Void, 2=Dash, 3=Bidding, 4=Estimation, 5=Play
        
        # Phase 1: Void declarations
        self.void_declarations = {}  # player_id -> [suits]
        
        # Phase 2: Dash declarations  
        self.dash_players: Set[int] = set()
        
        # Phase 3: Trump bidding
        self.bidding_active = True
        self.current_bidder = 0  # Will be set randomly for first round
        self.highest_bid = 0
        self.highest_bidder = None
        self.trump_suit = None
        self.bid_history = []  # [(player_id, bid_amount, trump_suit), ...]
        self.passes = set()  # Players who have passed
        
        # Phase 4: Estimation
        self.estimations: Dict[int, int] = {}
        self.with_players: Set[int] = set()
        self.declarer_id: Optional[int] = None
        self.declarer_bid: Optional[int] = None
        
        # Phase 5: Card play
        self.Tricks: List[Trick] = []
        self.current_Trick: Optional[Trick] = None
        self.leader_id = 0
        
    def execute_phase_1_void_declarations(self):
        """Phase 1: Each player declares void suits."""
        print(f"\n=== PHASE 1: VOID DECLARATIONS - Round {self.round_number} ===")
        
        for player_id in range(4):
            player = self.players[player_id]
            void_suits = player.declare_avoid()
            if void_suits:
                self.void_declarations[player_id] = void_suits
                print(f"{player.name} declares VOID")
            else:
                print(f"{player.name} has cards in all suits")
        
        self.current_phase = 2
    
    def execute_phase_2_dash_declarations(self):
        """Phase 2: Players declare dash intentions."""
        if self.is_speed_round:
            print("Speed Round: No Dash declarations allowed")
            self.current_phase = 3
            return
            
        print(f"\n=== PHASE 2: DASH DECLARATIONS - Round {self.round_number} ===")
        
        dash_count = 0
        for player_id in range(4):
            player = self.players[player_id]
            
            if dash_count >= 2:
                print(f"{player.name} cannot declare Dash (2 players already declared)")
                continue
                
            # Show hand and get dash decision
            if isinstance(player, HumanPlayer):
                print(f"\n{player.name}'s hand: {player._format_hand()}")
                wants_dash = player.make_dash_choice()
            elif hasattr(player, 'strategy'):
                # Bot decision (simplified for now)
                wants_dash = False  # Bots don't dash in this implementation
            else:
                wants_dash = False
                
            if wants_dash:
                self.dash_players.add(player_id)
                dash_count += 1
                print(f"{player.name} declares DASH (aims for 0 Tricks)")
            else:
                print(f"{player.name} will participate in normal bidding")
        
        self.current_phase = 3
    
    def execute_phase_3_trump_bidding(self):
        """Phase 3: Trump bidding auction."""
        if self.is_speed_round:
            self._handle_speed_round_trump()
            self.current_phase = 4
            return
            
        print(f"\n=== PHASE 3: TRUMP BIDDING - Round {self.round_number} ===")
        
        eligible_bidders = [i for i in range(4) if i not in self.dash_players]
        
        if len(eligible_bidders) == 0:
            print("All players declared Dash! Round becomes Sa'aydeh (doubled)")
            return "SA_AYDEH"
        
        self.current_bidder = eligible_bidders[0]
        max_passes_in_row = 0
        total_passes = 0
        
        while total_passes < len(eligible_bidders) - 1 and max_passes_in_row < len(eligible_bidders):
            current_player = self.players[self.current_bidder]
            
            if self.current_bidder in self.passes:
                self.current_bidder = self._get_next_bidder(eligible_bidders)
                max_passes_in_row += 1
                continue
            
            # Get bid
            if isinstance(current_player, HumanPlayer):
                bid_result = current_player.make_bid_interactive(self._get_bidding_context(), False)
            elif hasattr(current_player, 'strategy'):
                bid_result = current_player.strategy.make_bid(
                    current_player.hand, self._get_bidding_context(), False, False
                )
            else:
                bid_result = None
            
            if bid_result is None:
                self.passes.add(self.current_bidder)
                print(f"{current_player.name} passes")
                total_passes += 1
                max_passes_in_row += 1
            else:
                bid_amount, trump_suit = bid_result
                if self._is_valid_bid(bid_amount, trump_suit):
                    self._record_bid(self.current_bidder, bid_amount, trump_suit)
                    max_passes_in_row = 0  # Reset consecutive passes
                else:
                    print(f"{current_player.name} invalid bid, must pass")
                    self.passes.add(self.current_bidder)
                    total_passes += 1
                    max_passes_in_row += 1
            
            self.current_bidder = self._get_next_bidder(eligible_bidders)
        
        self._determine_declarer_and_with()
        
        if self.declarer_id is None:
            print("All players passed! Round becomes Sa'aydeh (doubled)")
            return "SA_AYDEH"
        
        self.current_phase = 4
    
    def execute_phase_4_estimation(self):
        """Phase 4: Trick estimation by non-dash, non-declarer players."""
        print(f"\n=== PHASE 4: ESTIMATION - Round {self.round_number} ===")
        
        if self.is_speed_round:
            self._handle_speed_round_estimation()
            self.current_phase = 5
            return
        
        # Declarer's estimation is locked to their bid
        self.estimations[self.declarer_id] = self.declarer_bid
        print(f"Declarer {self.players[self.declarer_id].name}: {self.declarer_bid} Tricks (locked)")
        
        # Get estimation order (excluding declarer and dash players)
        estimation_order = self._get_estimation_order()
        
        for i, player_id in enumerate(estimation_order):
            player = self.players[player_id]
            is_last = (i == len(estimation_order) - 1)
            
            if isinstance(player, HumanPlayer):
                estimation = player.make_estimation_interactive(
                    self.trump_suit, self.declarer_bid,
                    list(self.estimations.values()), is_last, False
                )
            elif hasattr(player, 'strategy'):
                estimation = player.strategy.make_estimation(
                    player.hand, self.trump_suit, self.declarer_bid,
                    list(self.estimations.values()), is_last, False
                )
            else:
                estimation = min(3, self.declarer_bid)
            
            # Handle WITH detection
            if player_id in self.with_players:
                min_estimation = self._get_player_last_bid(player_id)
                max_estimation = self.declarer_bid
                if estimation < min_estimation:
                    estimation = min_estimation
                elif estimation > max_estimation:
                    estimation = max_estimation
                print(f"{player.name} (WITH): {estimation} Tricks")
            else:
                print(f"{player.name}: {estimation} Tricks")
            
            # Apply risk constraint for last estimator
            if is_last:
                estimation = self._apply_risk_constraint(estimation)
            
            self.estimations[player_id] = estimation
        
        # Set dash player estimations to 0
        for dash_player in self.dash_players:
            self.estimations[dash_player] = 0
        
        self.current_phase = 5
    
    def execute_phase_5_card_play(self):
        """Phase 5: Play all 13 Tricks."""
        print(f"\n=== PHASE 5: CARD PLAY - Round {self.round_number} ===")
        
        # Declarer leads first Trick (or highest estimator in speed rounds)
        if self.is_speed_round:
            if self.estimations:
                highest_est = max(self.estimations.values())
                for pid, est in self.estimations.items():
                    if est == highest_est:
                        self.leader_id = pid
                        break
            else:
                self.leader_id = 0  # Default to player 0
        else:
            self.leader_id = self.declarer_id if self.declarer_id is not None else 0
        
        # Play all 13 Tricks
        for Trick_num in range(1, 14):
            self._play_single_Trick(Trick_num)
        
        return True  # Round complete
    
    def _handle_speed_round_trump(self):
        """Handle trump determination for speed rounds."""
        trump_order = [None, Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]
        base_round = self.round_number - (14 if True else 6)  # Adjust based on game mode
        self.trump_suit = trump_order[base_round % 5]
        
        # Check for super calls (8+ Tricks)
        for player_id in range(4):
            player = self.players[player_id]
            if hasattr(player, 'strategy'):
                wants_super = False  # Simplified
            else:
                wants_super = False
            
            if wants_super:
                # Player can change trump with super call
                print(f"{player.name} requests Super Call - can change trump")
                # Implementation would allow trump selection here
        
        print(f"Speed Round Trump: {self.trump_suit.name if self.trump_suit else 'No Trump'}")
    
    def _handle_speed_round_estimation(self):
        """Handle estimation for speed rounds."""
        for player_id in range(4):
            player = self.players[player_id]
            
            if isinstance(player, HumanPlayer):
                estimation = player.make_estimation_interactive(
                    self.trump_suit, 13, list(self.estimations.values()), 
                    player_id == 3, False
                )
            elif hasattr(player, 'strategy'):
                estimation = player.strategy.make_estimation(
                    player.hand, self.trump_suit, 13,
                    list(self.estimations.values()), player_id == 3, False
                )
            else:
                estimation = 3
            
            # Speed round: WITH is only same number
            if estimation in self.estimations.values():
                for other_pid, other_est in self.estimations.items():
                    if other_est == estimation:
                        self.with_players.add(player_id)
                        self.with_players.add(other_pid)
            
            self.estimations[player_id] = estimation
            print(f"{player.name}: {estimation} Tricks")
    
    def _is_valid_bid(self, amount: int, trump_suit: Optional[Suit]) -> bool:
        """Check if bid is valid according to rules."""
        if amount < 4 or amount > 13:
            return False
        
        # Must be higher than current highest
        if amount < self.highest_bid:
            return False
        
        # If same amount, trump must be stronger
        if amount == self.highest_bid:
            suit_ranks = {Suit.CLUBS: 1, Suit.DIAMONDS: 2, Suit.HEARTS: 3, Suit.SPADES: 4, None: 5}
            current_rank = suit_ranks.get(self.trump_suit, 0)
            new_rank = suit_ranks.get(trump_suit, 0)
            return new_rank >= current_rank
        
        return True
    
    def _record_bid(self, player_id: int, amount: int, trump_suit: Optional[Suit]):
        """Record a valid bid and update highest."""
        self.bid_history.append((player_id, amount, trump_suit))
        
        # Update highest bid tracking
        suit_ranks = {Suit.CLUBS: 1, Suit.DIAMONDS: 2, Suit.HEARTS: 3, Suit.SPADES: 4, None: 5}
        current_rank = suit_ranks.get(self.trump_suit, 0)
        new_rank = suit_ranks.get(trump_suit, 0)
        
        if amount > self.highest_bid or (amount == self.highest_bid and new_rank > current_rank):
            self.highest_bid = amount
            self.highest_bidder = player_id
            self.trump_suit = trump_suit
        
        player_name = self.players[player_id].name
        trump_name = trump_suit.name if trump_suit else "No Trump"
        print(f"{player_name} bids {amount} {trump_name}")
    
    def _get_next_bidder(self, eligible_bidders: List[int]) -> int:
        """Get next bidder in counterclockwise order."""
        current_idx = eligible_bidders.index(self.current_bidder)
        # Counterclockwise: go backwards in list, wrap around
        next_idx = (current_idx - 1) % len(eligible_bidders)
        return eligible_bidders[next_idx]
    
    def _determine_declarer_and_with(self):
        """Determine declarer and WITH players."""
        if not self.bid_history:
            return
        
        # Declarer is highest bidder
        self.declarer_id = self.highest_bidder
        self.declarer_bid = self.highest_bid
        
        # Determine WITH players
        final_trump = self.trump_suit
        for player_id, bid_amount, trump_suit in self.bid_history:
            if player_id != self.declarer_id and trump_suit == final_trump:
                self.with_players.add(player_id)
    
    def _get_estimation_order(self) -> List[int]:
        """Get order for estimation phase."""
        if self.is_speed_round or self.declarer_id is None:
            # Speed rounds or no declarer: go counterclockwise from player 0
            order = []
            pos = 3  # Start counterclockwise from 0
            for _ in range(4):
                order.append(pos)
                pos = (pos - 1) % 4
            return order
        
        # Normal rounds: order by bid history then counterclockwise from declarer
        bidders_by_amount = {}
        for player_id, amount, trump_suit in self.bid_history:
            if amount not in bidders_by_amount:
                bidders_by_amount[amount] = []
            bidders_by_amount[amount].append(player_id)
        
        order = []
        sorted_amounts = sorted(bidders_by_amount.keys(), reverse=True)
        
        for amount in sorted_amounts[1:]:  # Skip highest (declarer)
            order.extend(bidders_by_amount[amount])
        
        remaining = [i for i in range(4) 
                    if i not in order and i != self.declarer_id and i not in self.dash_players]
        
        declarer_pos = self.declarer_id
        remaining_sorted = []
        pos = (declarer_pos - 1) % 4
        while len(remaining_sorted) < len(remaining):
            if pos in remaining:
                remaining_sorted.append(pos)
            pos = (pos - 1) % 4
        
        order.extend(remaining_sorted)
        return order
    
    def _apply_risk_constraint(self, estimation: int) -> int:
        """Apply risk constraint to last estimator."""
        current_total = sum(self.estimations.values())
        if current_total + estimation == 13:
            print(f"Risk constraint: Cannot estimate {estimation} (total would be 13)")
            if estimation > 0:
                estimation -= 1
            else:
                estimation += 1
            print(f"Adjusted to {estimation}")
        return estimation
    
    def _get_player_last_bid(self, player_id: int) -> int:
        """Get player's last recorded bid amount."""
        for pid, amount, trump in reversed(self.bid_history):
            if pid == player_id:
                return amount
        return 0
    
    def _play_single_Trick(self, Trick_num: int):
        """Play a single Trick."""
        print(f"\n--- Trick {Trick_num} ---")
        
        self.current_Trick = Trick(self.leader_id)
        play_order = self._get_counterclockwise_order(self.leader_id)
        
        for player_id in play_order:
            player = self.players[player_id]
            valid_plays = player.get_valid_plays(self.current_Trick.led_suit)
            
            if isinstance(player, HumanPlayer):
                card = player.choose_card_interactive(
                    valid_plays, self.trump_suit, self.current_Trick.led_suit, []
                )
            elif hasattr(player, 'strategy'):
                card = player.strategy.choose_card(
                    player.hand, valid_plays, self.trump_suit,
                    self.current_Trick.led_suit, list(self.current_Trick.cards.values())
                )
            else:
                card = valid_plays[0]
            
            player.play_card(card)
            self.current_Trick.add_card(player_id, card)
            print(f"{player.name} plays {card}")
        
        # Determine winner
        winner_id = self.current_Trick.determine_winner(self.trump_suit)
        self.players[winner_id].tricks_won += 1
        self.Tricks.append(self.current_Trick)
        self.leader_id = winner_id
        
        print(f">>> {self.players[winner_id].name} wins the Trick! <<<")
    
    def _get_counterclockwise_order(self, start_id: int) -> List[int]:
        """Get play order counterclockwise from start_id."""
        order = []
        pos = start_id
        for _ in range(4):
            order.append(pos)
            pos = (pos - 1) % 4  # Counterclockwise
        return order
    
    def _get_bidding_context(self) -> List:
        """Get current bidding context for players."""
        return [(pid, amt, trump) for pid, amt, trump in self.bid_history]
    
    def calculate_scores(self) -> Dict[int, int]:
        """Calculate round scores according to strict Estimation rules."""
        scores = {}
        total_estimations = sum(self.estimations.values())
        
        # Determine round type
        is_over_round = total_estimations > 13
        is_under_round = total_estimations < 13
        
        # Calculate risk level for last estimator
        risk_level = abs(total_estimations - 13) // 2
        last_estimator = self._get_last_estimator()
        
        # Count successful/failed players
        successful_players = []
        failed_players = []
        
        for player_id in range(4):
            estimation = self.estimations.get(player_id, 0)
            actual = self.players[player_id].tricks_won
            
            if actual == estimation:
                successful_players.append(player_id)
            else:
                failed_players.append(player_id)
        
        # Check for double WITH
        double_with = len(self.with_players) >= 2
        
        for player_id in range(4):
            estimation = self.estimations.get(player_id, 0)
            actual = self.players[player_id].tricks_won
            score = 0
            
            # Base scoring
            if actual == estimation:
                score += 10  # Success bonus
                
                # Super call scoring (8+ Tricks)
                if estimation >= 8:
                    score += estimation * estimation  # T*T
                else:
                    score += estimation  # Regular Trick points
                
            else:
                score -= 10  # Failure penalty
                
                # Super call penalty
                if estimation >= 8:
                    score -= (estimation * estimation) // 2  # T*T/2
                else:
                    score -= abs(actual - estimation)  # Difference penalty
            
            # Caller/WITH bonus/penalty
            is_caller_or_with = (player_id == self.declarer_id or player_id in self.with_players)
            if is_caller_or_with:
                if actual == estimation:
                    score += 10
                else:
                    score -= 10
            
            # Sole success/failure bonus/penalty
            if len(successful_players) == 1 and player_id in successful_players:
                score += 10
            elif len(failed_players) == 1 and player_id in failed_players:
                score -= 10
            
            # Risk scoring (only for last estimator)
            if player_id == last_estimator and risk_level > 0:
                if actual == estimation:
                    score += 10 * risk_level
                else:
                    score -= 10 * risk_level
            
            # Dash scoring
            if player_id in self.dash_players:
                if actual == 0:  # Dash success
                    if is_over_round:
                        score = 25
                    elif is_under_round:
                        score = 33
                    else:
                        score = 10  # Exact round
                else:  # Dash failure
                    if is_over_round:
                        score = -25
                    elif is_under_round:
                        score = -33
                    else:
                        score = -10  # Exact round
            
            # Double WITH multiplier
            if double_with:
                score *= 2
            
            scores[player_id] = score
        
        return scores
    
    def _get_last_estimator(self) -> int:
        """Get the player who estimated last (at risk)."""
        estimation_order = self._get_estimation_order()
        return estimation_order[-1] if estimation_order else 0