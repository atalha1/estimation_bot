#!/usr/bin/env python3
"""
Quick setup and commands for NADL system.
Location: setup_nadl.py
"""

import sys
import subprocess
from pathlib import Path

def train_nadl(generations=20):
    """Train NADL for specified generations."""
    cmd = [sys.executable, "train_selfplay.py", "--generations", str(generations)]
    subprocess.run(cmd)

def play_vs_nadl():
    """Start human vs NADL game."""
    cmd = [sys.executable, "play_vs_nadl.py"]
    subprocess.run(cmd)

def main():
    if len(sys.argv) < 2:
        print("🤖 NADL Command Center")
        print("Commands:")
        print("  train [generations] - Train NADL (default: 20)")
        print("  play               - Play against NADL")
        print("  status             - Show NADL status")
        return
    
    command = sys.argv[1].lower()
    
    if command == "train":
        gens = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        print(f"🎯 Starting NADL training for {gens} generations...")
        train_nadl(gens)
        
    elif command == "play":
        print("🎮 Starting human vs NADL game...")
        play_vs_nadl()
        
    elif command == "status":
        from estimation_bot.training.champion import ChampionManager
        mgr = ChampionManager()
        print(f"🏆 NADL v{mgr.generation_count}.0")
        print(f"   Win Rate: {mgr.champion_stats['best_win_rate']:.2%}")
        print(f"   Total Games: {mgr.champion_stats['total_games']}")
        print(f"   Training Sessions: {mgr.champion_stats['training_sessions']}")
        
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()