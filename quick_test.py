#!/usr/bin/env python3
"""
Test script to validate the game fixes.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from estimation_bot.player import Player, HumanPlayer
from estimation_bot.game import EstimationGame
from bot.random_bot import RandomBot, WeightedRandomBot
from bot.heuristic_bot import HeuristicBot


def test_basic_game():
    """Test basic game functionality."""
    print("Testing basic game setup...")
    
    # Create 4 bot players
    players = []
    for i in range(4):
        player = Player(i, f"Bot_{i}")
        player.strategy = RandomBot(f"RandomBot_{i}")
        players.append(player)
    
    # Create game
    game = EstimationGame(players, "MICRO")  # Use MICRO for quick test
    
    print(f"Game created with {len(players)} players")
    print(f"Game mode: {game.game_mode}")
    print(f"Total rounds: {game.total_rounds}")
    
    return game


def test_bidding_system():
    """Test the bidding system."""
    print("\nTesting bidding system...")
    
    game = test_basic_game()
    round_obj = game.start_new_round()
    
    print(f"Round {round_obj.round_number} started")
    print(f"Is speed round: {round_obj.is_speed_round}")
    
    # Test bidding collection
    try:
        game.collect_bids_and_estimations(round_obj)
        print("✅ Bidding system working")
        return True
    except Exception as e:
        print(f"❌ Bidding system failed: {e}")
        return False


def test_human_interface():
    """Test human player interface setup."""
    print("\nTesting human player interface...")
    
    try:
        human = HumanPlayer(0, "TestHuman")
        print(f"Human player created: {human.name}")
        
        # Test method existence
        assert hasattr(human, 'make_bid_interactive')
        assert hasattr(human, 'make_estimation_interactive')
        assert hasattr(human, 'choose_card_interactive')
        
        print("✅ Human interface methods available")
        return True
    except Exception as e:
        print(f"❌ Human interface failed: {e}")
        return False


def test_bot_interface():
    """Test bot interfaces."""
    print("\nTesting bot interfaces...")
    
    try:
        bots = [
            RandomBot("TestRandom"),
            WeightedRandomBot("TestWeighted"),
            HeuristicBot("TestHeuristic")
        ]
        
        for bot in bots:
            # Test method existence
            assert hasattr(bot, 'make_bid')
            assert hasattr(bot, 'make_estimation')
            assert hasattr(bot, 'choose_card')
            print(f"✅ {bot.name} interface complete")
        
        return True
    except Exception as e:
        print(f"❌ Bot interface failed: {e}")
        return False


def run_quick_bot_game():
    """Run a quick bot-only game to test full flow."""
    print("\nRunning quick bot game...")
    
    try:
        # Create players
        players = []
        bot_types = [RandomBot, WeightedRandomBot, HeuristicBot, RandomBot]
        
        for i, bot_class in enumerate(bot_types):
            player = Player(i, f"Bot_{i}")
            player.strategy = bot_class(f"Bot_{i}")
            players.append(player)
        
        # Create and run game
        game = EstimationGame(players, "MICRO")
        final_scores = game.play_game()
        
        print("✅ Full game completed successfully!")
        print(f"Final scores: {final_scores}")
        return True
        
    except Exception as e:
        print(f"❌ Full game failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🎴 Testing Estimation Game Fixes 🎴\n")
    
    tests = [
        test_basic_game,
        test_bidding_system,
        test_human_interface,
        test_bot_interface,
        run_quick_bot_game
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result if isinstance(result, bool) else True)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Game fixes are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)