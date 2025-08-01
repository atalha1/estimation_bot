"""
Manages the persistent NADL champion model across training sessions.
Location: training/champion.py
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

from .models import HeuristicModelWrapper, LearnableModelWrapper
from .trainer import ModelInterface, ModelStats


class ChampionManager:
    """Manages persistent champion model and opponent selection."""
    
    def __init__(self, champions_dir: str = "champions"):
        self.champions_dir = Path(champions_dir)
        self.champions_dir.mkdir(exist_ok=True)
        
        self.current_champion: Optional[ModelInterface] = None
        self.champion_history: list = []
        self.generation_count = 0
        
        # Champion metadata
        self.champion_stats = {
            'total_games': 0,
            'wins': 0,
            'training_sessions': 0,
            'best_win_rate': 0.0,
            'created': None,
            'last_updated': None
        }
        
        self._load_champion()
    
    def _load_champion(self):
        """Load existing NADL champion or create new one."""
        champion_path = self.champions_dir / "NADL_champion.pkl"
        metadata_path = self.champions_dir / "NADL_metadata.json"
        
        if champion_path.exists() and metadata_path.exists():
            try:
                # Load champion model
                with open(champion_path, 'rb') as f:
                    self.current_champion = pickle.load(f)
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.champion_stats = data['stats']
                    self.champion_history = data['history']
                    self.generation_count = data['generation_count']
                
                print(f"🏆 Loaded NADL champion (Generation {self.generation_count})")
                print(f"   Total games: {self.champion_stats['total_games']}")
                print(f"   Win rate: {self.champion_stats['best_win_rate']:.2%}")
                
            except Exception as e:
                print(f"⚠️ Error loading champion: {e}")
                self._create_new_champion()
        else:
            self._create_new_champion()
    
    def _create_new_champion(self):
        """Create new NADL champion."""
        self.current_champion = HeuristicModelWrapper(model_id="NADL_v1.0")
        self.generation_count = 1
        self.champion_stats['created'] = datetime.now().isoformat()
        print("🌟 Created new NADL champion (v1.0)")
    
    def save_champion(self):
        """Save current champion and metadata."""
        champion_path = self.champions_dir / "NADL_champion.pkl"
        metadata_path = self.champions_dir / "NADL_metadata.json"
        
        # Save model
        with open(champion_path, 'wb') as f:
            pickle.dump(self.current_champion, f)
        
        # Save metadata
        metadata = {
            'stats': self.champion_stats,
            'history': self.champion_history,
            'generation_count': self.generation_count
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def update_champion(self, new_champion: ModelInterface, stats: ModelStats):
        """Update NADL with better performing model."""
        old_win_rate = self.champion_stats['best_win_rate']
        new_win_rate = stats.win_rate
        
        if new_win_rate > old_win_rate:
            # Archive old champion
            self._archive_champion()
            
            # Update champion
            self.current_champion = new_champion
            self.generation_count += 1
            
            # Update stats
            self.champion_stats.update({
                'total_games': self.champion_stats['total_games'] + stats.games_played,
                'wins': self.champion_stats['wins'] + stats.wins,
                'best_win_rate': new_win_rate,
                'last_updated': datetime.now().isoformat()
            })
            
            # Set new model ID
            self.current_champion.model_id = f"NADL_v{self.generation_count}.0"
            
            print(f"🚀 NADL evolved to v{self.generation_count}.0")
            print(f"   Win rate improved: {old_win_rate:.2%} → {new_win_rate:.2%}")
            
            self.save_champion()
            return True
        
        return False
    
    def _archive_champion(self):
        """Archive current champion before updating."""
        if self.current_champion:
            archive_path = self.champions_dir / f"NADL_v{self.generation_count}_archive.pkl"
            with open(archive_path, 'wb') as f:
                pickle.dump(self.current_champion, f)
            
            self.champion_history.append({
                'version': f"v{self.generation_count}.0",
                'archived_at': datetime.now().isoformat(),
                'win_rate': self.champion_stats['best_win_rate']
            })
    
    def get_training_opponents(self) -> list[ModelInterface]:
        """Get progressively stronger opponents for training (self-play focused)."""
        opponents = []
        
        # Primary opponent: Current champion (50% of games)
        if self.current_champion:
            opponents.append(self.current_champion.clone())
        
        # Secondary: Previous versions (30% of games) 
        archive_files = sorted(self.champions_dir.glob("NADL_v*_archive.pkl"))
        if archive_files:
            # Load 1-2 most recent archived versions
            recent_archives = archive_files[-2:] if len(archive_files) >= 2 else archive_files[-1:]
            for archive_path in recent_archives:
                try:
                    with open(archive_path, 'rb') as f:
                        archived_champion = pickle.load(f)
                        opponents.append(archived_champion)
                except Exception as e:
                    print(f"⚠️ Could not load archived champion: {e}")
        
        # Tertiary: Heuristic bot only as baseline (20% of games)
        # Only add heuristic if we don't have enough self-play opponents
        while len(opponents) < 3:
            if self.current_champion:
                # Create slightly weakened version of current champion
                weaker_clone = self.current_champion.clone()
                if hasattr(weaker_clone, 'add_noise'):
                    weaker_clone.add_noise(0.2)  # Add more noise to make it weaker
                opponents.append(weaker_clone)
            else:
                # Fallback to heuristic only if no champion exists
                from training.models import HeuristicModelWrapper
                opponents.append(HeuristicModelWrapper(f"Fallback_Heuristic_{len(opponents)}"))
        
        return opponents[:3]  # Return exactly 3 opponents

    def get_champion(self) -> ModelInterface:
        """Get current NADL champion."""
        return self.current_champion
    
    def increment_training_session(self):
        """Record new training session."""
        self.champion_stats['training_sessions'] += 1
        self.save_champion()
    def load_specific_version(self, version: str) -> Optional[ModelInterface]:
        """Load a specific version of NADL."""
        version_path = self.champions_dir / f"NADL_{version}.pkl"
        if version_path.exists():
            try:
                with open(version_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Error loading version {version}: {e}")
        return None
    
    def get_training_population(self, population_size: int = 4) -> List[ModelInterface]:
        """Create diverse training population."""
        population = []
        
        # Always include current champion
        if self.current_champion:
            population.append(self.current_champion.clone())
        
        # Add some previous versions
        archive_files = sorted(self.champions_dir.glob("NADL_v*_archive.pkl"))
        recent_archives = archive_files[-2:] if len(archive_files) > 2 else archive_files
        
        for archive_path in recent_archives[:population_size-1]:
            try:
                with open(archive_path, 'rb') as f:
                    archived_model = pickle.load(f)
                    population.append(archived_model.clone())
            except Exception as e:
                print(f"⚠️ Could not load archived model: {e}")
        
        # Fill remaining slots with champion variants
        while len(population) < population_size:
            if self.current_champion:
                variant = self.current_champion.clone()
                # Add small random variations for diversity
                if hasattr(variant, 'add_noise'):
                    variant.add_noise(0.1)
                population.append(variant)
            else:
                break
        
        return population[:population_size]