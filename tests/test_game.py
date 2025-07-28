"""
Unit tests for Estimation game logic.
Tests core game mechanics, rules, and edge cases.
"""

import pytest
from game.card import Card, Suit, Rank, create_deck
from game.deck import Deck
from game.player import Player
from game.rules import *
from game.game import EstimationGame
from bot.random_bot import RandomBot


class TestCard:
    """Test card functionality."""
    
    def test_card_creation(self):
        card = Card(Suit.SPADES, Rank.ACE)
        assert card.suit == Suit.SPADES
        assert card.rank == Rank.ACE
        assert str(card) == "A♠"
    
    def test_card_comparison(self):
        ace_spades = Card(Suit.SPADES, Rank.ACE)
        king_spades = Card(Suit.SPADES, Rank.KING)
        ace_hearts = Card(Suit.HEARTS, Rank.ACE)
        
        assert ace_spades > king_spades
        assert ace_spades == ace_hearts  # Same rank
    
    def test_card_beats(self):
        trump_suit = Suit.SPADES
        led_suit = Suit.HEARTS
        
        trump_2 = Card(Suit.SPADES, Rank.TWO)
        heart_ace = Card(Suit.HEARTS, Rank.ACE)
        club_king = Card(Suit.CLUBS, Rank.KING)
        
        # Trump beats non-trump
        assert trump_2.beats(heart_ace, trump_suit, led_suit)
        
        # Higher in led suit beats lower
        heart_king = Card(Suit.HEARTS, Rank.KING)
        assert heart_ace.beats(heart_king, trump_suit, led_suit)
        
        # Led suit beats off-suit
        assert heart_king.beats(club_king, trump_suit, led_suit)


class TestDeck:
    """Test deck functionality."""
    
    def test_deck_creation(self):
        deck = Deck()
        assert len(deck.cards) == 52
        
        # Check all cards are unique
        card_set = set(deck.cards)
        assert len(card_set) == 52
    
    def test_deck_dealing(self):
        deck = Deck()
        hand = deck.deal_hand(13)
        
        assert len(hand) == 13
        assert len(deck.cards) == 39
        
        # Cards should be unique
        assert len(set(hand)) == 13
    
    def test_round_dealing(self):
        deck = Deck()
        hands = deck.deal_round(4)
        
        assert len(hands) == 4
        for hand in hands.values():
            assert len(hand) == 13
        
        # All cards dealt
        total_cards = sum(len(hand) for hand in hands.values())
        assert total_cards == 52


class TestRules:
    """Test game rules and validation."""
    
    def test_bid_validation(self):
        # Normal rounds
        assert validate_bid(4, 13, False)
        assert validate_bid(13, 13, False)
        assert not validate_bid(3, 13, False)  # Too low
        assert not validate_bid(14, 13, False)  # Too high
        
        # Speed rounds
        assert validate_bid(0, 13, True)  # Dash allowed
        assert validate_bid(13, 13, True)
        
        # Dash calls
        assert validate_bid(0, 13, False)  # Dash call
    
    def test_score_calculation(self):
        # Made estimation exactly
        score = calculate_round_score(7, 7)
        assert score == 17  # 10 + 7
        
        # Missed estimation
        score = calculate_round_score(7, 5)
        assert score == -2  # -2 difference
        
        # Call bonus
        score = calculate_round_score(8, 8, is_call=True)
        assert score == 28  # 10 + 8 + 10
        
        # Dash call - over round
        score = calculate_round_score(0, 0, is_dash=True, is_over_round=True)
        assert score == 25
        
        # Dash call - under round
        score = calculate_round_score(0, 0, is_dash=True, is_over_round=False)
        assert score == 33
    
    def test_risk_calculation(self):
        assert calculate_risk_level(13) == 0  # Exact
        assert calculate_risk_level(15) == 1  # 2 over
        assert calculate_risk_level(11) == 1  # 2 under
        assert calculate_risk_level(17) == 2  # 4 over
        assert calculate_risk_level(9) == 2   # 4 under
    
    def test_bid_comparison(self):
        # Higher amount wins
        assert compare_bids((8, Suit.SPADES), (7, Suit.HEARTS)) == 1
        
        # Same amount, higher suit wins
        assert compare_bids((7, None), (7, Suit.SPADES)) == 1  # No Trump > Spades
        assert compare_bids((7, Suit.SPADES), (7, Suit.HEARTS)) == 1
        
        # Equal bids
        assert compare_bids((7, Suit.SPADES), (7, Suit.SPADES)) == 0


class TestPlayer:
    """Test player functionality."""
    
    def test_player_creation(self):
        player = Player(0, "TestPlayer")
        assert player.player_id == 0
        assert player.name == "TestPlayer"
        assert player.score == 0
        assert len(player.hand) == 0
    
    def test_card_management(self):
        player = Player(0)
        cards = [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.KING)]
        
        player.receive_cards(cards)
        assert len(player.hand) == 2
        
        played_card = player.play_card(cards[0])
        assert played_card == cards[0]
        assert len(player.hand) == 1
        assert cards[0] not in player.hand
    
    def test_valid_plays(self):
        player = Player(0)
        spade_ace = Card(Suit.SPADES, Rank.ACE)
        heart_king = Card(Suit.HEARTS, Rank.KING)
        spade_two = Card(Suit.SPADES, Rank.TWO)
        
        player.receive_cards([spade_ace, heart_king, spade_two])
        
        # Leading - can play any card
        valid = player.get_valid_plays(None)
        assert len(valid) == 3
        
        # Must follow spades if possible
        valid = player.get_valid_plays(Suit.SPADES)
        assert len(valid) == 2  # Both spades
        assert heart_king not in valid
        
        # Can't follow hearts - can play any
        valid = player.get_valid_plays(Suit.DIAMONDS)
        assert len(valid) == 3
    
    def test_avoid_declaration(self):
        player = Player(0)
        # Hand with no diamonds or clubs
        cards = [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.SPADES, Rank.KING), 
            Card(Suit.HEARTS, Rank.QUEEN)
        ]
        player.receive_cards(cards)
        
        missing = player.declare_avoid()
        assert Suit.DIAMONDS in missing
        assert Suit.CLUBS in missing
        assert len(missing) == 2
        assert player.has_avoid


class TestGame:
    """Test full game functionality."""
    
    def create_test_players(self):
        players = []
        for i in range(4):
            player = Player(i, f"Bot_{i}")
            player.strategy = RandomBot(f"RandomBot_{i}")
            players.append(player)
        return players
    
    def test_game_creation(self):
        players = self.create_test_players()
        game = EstimationGame(players, "FULL")
        
        assert len(game.players) == 4
        assert game.total_rounds == 18
        assert game.current_round == 0
        assert not game.game_complete
    
    def test_game_modes(self):
        players = self.create_test_players()
        
        full_game = EstimationGame(players, "FULL")
        assert full_game.total_rounds == 18
        
        mini_game = EstimationGame(players, "MINI")
        assert mini_game.total_rounds == 10
        
        micro_game = EstimationGame(players, "MICRO")
        assert micro_game.total_rounds == 5
    
    def test_speed_rounds(self):
        players = self.create_test_players()
        game = EstimationGame(players, "FULL")
        
        # Normal rounds
        assert not game.is_speed_round(1)
        assert not game.is_speed_round(13)
        
        # Speed rounds
        assert game.is_speed_round(14)
        assert game.is_speed_round(18)
        
        # Speed round trump suits
        assert game.get_speed_round_trump(14) is None  # No Trump
        assert game.get_speed_round_trump(15) == Suit.SPADES
        assert game.get_speed_round_trump(16) == Suit.HEARTS
        assert game.get_speed_round_trump(17) == Suit.DIAMONDS
        assert game.get_speed_round_trump(18) == Suit.CLUBS
    
    def test_micro_no_speed_rounds(self):
        players = self.create_test_players()
        game = EstimationGame(players, "MICRO")
        
        # No speed rounds in micro bola
        for round_num in range(1, 6):
            assert not game.is_speed_round(round_num)


if __name__ == "__main__":
    pytest.main([__file__])