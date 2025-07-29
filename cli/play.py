"""
Command-line interface for playing Estimation interactively.
Handles human player input and game display.
"""

from typing import List, Optional, Tuple
from estimation_bot.card import Card, Suit
from estimation_bot.player import Player
from estimation_bot.round import Round
from estimation_bot.rules import SUIT_RANKINGS


def display_hand(hand: List[Card], title: str = "Your Hand"):
    """Display player's hand in organized format."""
    print(f"\n{title}:")
    
    # Group by suit
    by_suit = {}
    for card in hand:
        if card.suit not in by_suit:
            by_suit[card.suit] = []
        by_suit[card.suit].append(card)
    
    # Sort cards within each suit
    for suit in by_suit:
        by_suit[suit].sort(key=lambda c: c.rank.value, reverse=True)
    
    # Display each suit
    for suit in [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]:
        if suit in by_suit:
            cards_str = ' '.join(str(card) for card in by_suit[suit])
            print(f"  {suit}: {cards_str}")


def display_game_status(round_obj: Round, players: List[Player]):
    """Display current game status."""
    print(f"\n{'='*50}")
    print(f"Round {round_obj.round_number} - Trump: {round_obj.trump_suit or 'No Trump'}")
    print(f"{'='*50}")
    
    print("\nCurrent Standings:")
    for player in players:
        status = f"  {player.name}: {player.score} points"
        if hasattr(player, 'estimation') and player.estimation is not None:
            status += f" (Est: {player.estimation}, Won: {player.tricks_won})"
        print(status)


def get_human_bid(player: Player, other_bids: List[Optional[Tuple]], 
                  is_speed_round: bool = False) -> Optional[Tuple]:
    """Get bid from human player."""
    print(f"\n{player.name}'s turn to bid:")
    display_hand(player.hand)
    
    if not is_speed_round:
        print("\nOther bids:", [f"{bid[0]} {bid[1] or 'NT'}" if bid else "Pass" 
                               for bid in other_bids])
        
        print("\nOptions:")
        print("  0: Pass")
        print("  4-13: Bid with trump suit")
        print("  Format: 'amount suit' (e.g., '7 SPADES') or 'amount NT' for No Trump")
        print("  Dash: Enter '0 DASH' for Dash Call")
        
        while True:
            try:
                bid_input = input("Enter your bid: ").strip().upper()
                
                if bid_input == "0":
                    return None  # Pass
                
                parts = bid_input.split()
                if len(parts) != 2:
                    print("Invalid format. Use 'amount suit' or '0' to pass")
                    continue
                
                amount = int(parts[0])
                trump_str = parts[1]
                
                if amount == 0 and trump_str == "DASH":
                    return (0, None)  # Dash call
                
                if not (4 <= amount <= 13):
                    print("Bid amount must be 4-13 (or 0 for dash)")
                    continue
                
                if trump_str == "NT":
                    trump_suit = None
                else:
                    trump_suit = None
                    for suit in Suit:
                        if suit.name == trump_str:
                            trump_suit = suit
                            break
                    if trump_suit is None:
                        print("Invalid suit. Use SPADES, HEARTS, DIAMONDS, CLUBS, or NT")
                        continue
                
                return (amount, trump_suit)
                
            except ValueError:
                print("Invalid input. Please try again.")
    
    else:
        # Speed round - estimation only
        return None


def get_human_estimation(player: Player, trump_suit: Optional[Suit], 
                        declarer_bid: int, other_estimations: List[Optional[int]],
                        is_last: bool = False) -> int:
    """Get estimation from human player."""
    print(f"\n{player.name}'s estimation:")
    display_hand(player.hand)
    print(f"Trump: {trump_suit or 'No Trump'}")
    print(f"Declarer bid: {declarer_bid}")
    print(f"Other estimations: {other_estimations}")
    
    total_so_far = sum(est for est in other_estimations if est is not None) + declarer_bid
    
    if is_last:
        print(f"\n⚠️  You are the RISK player!")
        print(f"Total estimations so far: {total_so_far}")
        print(f"You CANNOT bid {13 - total_so_far} (would make total exactly 13)")
    
    while True:
        try:
            estimation = int(input(f"Enter estimation (0-{min(declarer_bid, 13)}): "))
            
            if not (0 <= estimation <= min(declarer_bid, 13)):
                print(f"Estimation must be 0-{min(declarer_bid, 13)}")
                continue
            
            if is_last and (total_so_far + estimation == 13):
                print("Cannot make total exactly 13! Choose different number.")
                continue
            
            return estimation
            
        except ValueError:
            print("Please enter a valid number.")


def get_human_card_play(player: Player, valid_plays: List[Card],
                       trump_suit: Optional[Suit], led_suit: Optional[Suit],
                       trick_cards: List[Card]) -> Card:
    """Get card play from human player."""
    print(f"\n{player.name}'s turn to play:")
    display_hand(player.hand)
    
    if led_suit:
        print(f"Led suit: {led_suit}")
    else:
        print("You are leading")
    
    print(f"Trump: {trump_suit or 'No Trump'}")
    
    if trick_cards:
        print("Cards played:", ', '.join(str(card) for card in trick_cards))
    
    print("\nValid plays:", ', '.join(str(card) for card in valid_plays))
    
    while True:
        try:
            card_input = input("Enter card to play: ").strip()
            
            for card in valid_plays:
                if str(card).upper() == card_input.upper():
                    return card
            
            print("Invalid card. Choose from valid plays.")
            
        except (ValueError, KeyboardInterrupt):
            print("Please enter a valid card.")


def display_trick_result(winner_name: str, trick_cards: List[Card], 
                        trump_suit: Optional[Suit]):
    """Display the result of a completed trick."""
    print(f"\nTrick won by {winner_name}")
    print(f"Cards: {', '.join(str(card) for card in trick_cards)}")
    print(f"Trump: {trump_suit or 'No Trump'}")


def display_round_result(round_obj: Round, players: List[Player], 
                        round_scores: dict):
    """Display results of completed round."""
    print(f"\n{'='*60}")
    print(f"Round {round_obj.round_number} Results:")
    print(f"Trump: {round_obj.trump_suit or 'No Trump'}")
    print(f"{'='*60}")
    
    for player_id, player in enumerate(players):
        score = round_scores.get(player_id, 0)
        status = "✓" if player.estimation == player.tricks_won else "✗"
        print(f"{status} {player.name}: Est {player.estimation}, "
              f"Won {player.tricks_won}, Score {score:+d}")
    
    print(f"\nRunning totals:")
    for player in sorted(players, key=lambda p: p.score, reverse=True):
        print(f"  {player.name}: {player.score}")


def confirm_continue() -> bool:
    """Ask player if they want to continue."""
    try:
        response = input("\nPress Enter to continue or 'q' to quit: ").strip().lower()
        return response != 'q'
    except KeyboardInterrupt:
        return False


def display_final_results(players: List[Player], winner_id: int):
    """Display final game results."""
    print(f"\n{'🎉'*20}")
    print(f"GAME COMPLETE!")
    print(f"{'🎉'*20}")
    
    winner = players[winner_id]
    print(f"\n🏆 Winner: {winner.name} with {winner.score} points! 🏆")
    
    print(f"\nFinal Standings:")
    for i, player in enumerate(sorted(players, key=lambda p: p.score, reverse=True)):
        medal = ["🥇", "🥈", "🥉", "4️⃣"][i]
        print(f"{medal} {player.name}: {player.score} points")