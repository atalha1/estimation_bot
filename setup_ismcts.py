#!/usr/bin/env python3
"""
ISMCTS Agent Setup Script
Location: estimation_bot/setup_ismcts.py

Easy setup and testing script for the ISMCTS competitive agent.
"""

import os
import sys
from pathlib import Path

def setup_ismcts():
    """Setup ISMCTS agent in the estimation bot system."""
    
    print("🤖 Setting up ISMCTS Competitive Agent")
    print("=" * 40)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "estimation_bot").exists():
        print("❌ Error: Run this script from the project root directory")
        print("   Expected structure: project_root/estimation_bot/")
        return False
    
    # Create necessary directories
    bot_dir = current_dir / "estimation_bot" / "bot"
    bot_dir.mkdir(exist_ok=True)
    
    training_dir = current_dir / "estimation_bot" / "training"
    training_dir.mkdir(exist_ok=True)
    
    # Check for required files
    required_files = [
        "estimation_bot/card.py",
        "estimation_bot/player.py", 
        "estimation_bot/game.py",
        "estimation_bot/training/trainer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (current_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ Directory structure validated")
    
    # Create __init__.py files if they don't exist
    init_files = [
        "estimation_bot/__init__.py",
        "estimation_bot/bot/__init__.py",
        "estimation_bot/training/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = current_dir / init_file
        if not init_path.exists():
            init_path.write_text("# Auto-generated __init__.py\n")
            print(f"✅ Created {init_file}")
    
    print("✅ ISMCTS Agent setup complete!")
    return True

def run_quick_test():
    """Run a quick test of the ISMCTS agent."""
    
    print("\n🧪 Running quick ISMCTS test...")
    
    try:
        # Add current directory to Python path
        sys.path.insert(0, str(Path.cwd()))
        
        # Import and test basic functionality
        from estimation_bot.bot.ismcts_agent import ISMCTSAgent
        from estimation_bot.card import Card, Suit, Rank
        
        # Create agent
        agent = ISMCTSAgent("TestAgent", simulations_per_move=50)
        
        # Test with sample hand
        test_hand = [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.CLUBS, Rank.QUEEN),
            Card(Suit.DIAMONDS, Rank.JACK),
            Card(Suit.SPADES, Rank.TEN)
        ]
        
        # Test bidding
        bid_result = agent.make_bid(test_hand, {}, False, True)
        print(f"✅ Bidding test: {bid_result}")
        
        # Test estimation
        estimation = agent.make_estimation(test_hand, Suit.SPADES, 7, [6, 2], False, True)
        print(f"✅ Estimation test: {estimation}")
        
        # Test card play
        valid_plays = test_hand[:3]
        chosen_card = agent.choose_card(test_hand, valid_plays, Suit.SPADES, None, [])
        print(f"✅ Card play test: {chosen_card}")
        
        # Get performance stats
        stats = agent.get_performance_stats()
        print(f"✅ Performance tracking: {len(stats)} metrics")
        
        print("✅ All basic tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all ISMCTS files are in place")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def run_mini_benchmark():
    """Run a mini benchmark against simple opponents."""
    
    print("\n🎯 Running mini benchmark (10 games)...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        from run_ismcts_agent import ISMCTSChampion
        
        # Create champion with fast config for testing
        config = {
            'simulations_per_move': 100,  # Faster for testing
            'nil_threshold': 0.4,
            'endgame_threshold': 5
        }
        
        champion = ISMCTSChampion(config)
        results = champion.run_competitive_benchmark(12)  # Quick test
        
        print(f"\n📊 Mini Benchmark Results:")
        print(f"Win Rate: {results['overall_win_rate']:.1%}")
        print(f"Speed: {results['overall_avg_time']:.2f}s per game")
        print(f"Rating: {results['rating']}")
        
        if results['overall_win_rate'] >= 0.5:
            print("✅ ISMCTS agent is competitive!")
        else:
            print("⚠️  Performance below 50% - may need tuning")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def main():
    """Main setup and test routine."""
    
    print("🚀 ISMCTS Competitive Agent Setup")
    print("Advanced AI for Estimation Card Game")
    print("=" * 50)
    
    # Step 1: Setup
    if not setup_ismcts():
        print("\n❌ Setup failed. Please fix errors and try again.")
        return
    
    # Step 2: Quick test
    if not run_quick_test():
        print("\n❌ Basic tests failed. Check your installation.")
        return
    
    # Step 3: Mini benchmark
    print("\n" + "="*50)
    response = input("Run mini benchmark? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = run_mini_benchmark()
        if success:
            print("\n🎉 ISMCTS Agent is ready for competitive play!")
        else:
            print("\n⚠️  Agent works but needs performance tuning.")
    else:
        print("\n✅ ISMCTS Agent basic setup complete!")
    
    print("\n📖 Next steps:")
    print("  - Run full benchmark: python estimation_bot/run_ismcts_agent.py --mode benchmark")
    print("  - Integrate with training: python estimation_bot/run_ismcts_agent.py --mode integrate")
    print("  - Train variations: python estimation_bot/train_ismcts.py --mode train")
    
if __name__ == "__main__":
    main()