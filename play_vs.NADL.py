"""
Play against 3 NADL bots as a human player.
Location: play_vs_nadl.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from estimation_bot.game import EstimationGame
from estimation_bot.player import Player, HumanPlayer
from estimation_bot.training.champion import ChampionManager


def main():
    print("🤖 Welcome to Estimation vs NADL!")
    print("You'll play against 3 versions of NADL")
    print("=" * 50)
    
    # Load NADL champion
    champion_mgr = ChampionManager()
    champion = champion_mgr.get_champion()
    
    if not champion:
        print("❌ No NADL champion found. Run training first!")
        return
    
    print(f"🏆 Playing against NADL v{champion_mgr.generation_count}.0")
    print(f"   NADL's win rate: {champion_mgr.champion_stats['best_win_rate']:.1%}")
    print(f"   Total games played: {champion_mgr.champion_stats['total_games']}")
    
    # Get game mode
    while True:
        mode = input("\nSelect game mode (MICRO/MINI/FULL) [MICRO]: ").upper() or "MICRO"
        if mode in ["MICRO", "MINI", "FULL"]:
            break
        print("Invalid mode. Please choose MICRO, MINI, or FULL")
    
    # Create players
    players = [
        HumanPlayer(0, "You"),
        Player(1, "NADL-Alpha"),
        Player(2, "NADL-Beta"), 
        Player(3, "NADL-Gamma")
    ]
    
    # Assign NADL bots to players 1, 2, 3
    for i in range(1, 4):
        nadl_bot = champion.create_bot(players[i].name)
        players[i].strategy = nadl_bot
    
    print(f"\n🎮 Starting {mode} game...")
    print("Players:")
    for p in players:
        print(f"  {p.id}: {p.name}")
    
    # Play game
    try:
        game = EstimationGame(players, mode)
        final_scores = game.play_game()
        
        # Show results
        print("\n" + "="*50)
        print("🏁 FINAL RESULTS")
        print("="*50)
        
        # Sort by score
        results = [(p.id, p.name, p.score) for p in players]
        results.sort(key=lambda x: x[2], reverse=True)
        
        for rank, (pid, name, score) in enumerate(results, 1):
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "4️⃣"
            print(f"{emoji} {rank}. {name}: {score} points")
        
        # Special message based on human performance
        human_rank = next(i for i, (pid, _, _) in enumerate(results, 1) if pid == 0)
        
        if human_rank == 1:
            print("\n🌟 Congratulations! You beat all 3 NADLs!")
        elif human_rank == 2:
            print("\n👏 Great job! You finished 2nd against the NADLs!")
        elif human_rank == 3:
            print("\n👍 Not bad! You beat one NADL!")
        else:
            print("\n🤖 The NADLs dominated this time. Keep practicing!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted. Thanks for playing!")
    except Exception as e:
        print(f"\n❌ Game error: {e}")


if __name__ == "__main__":
    main()