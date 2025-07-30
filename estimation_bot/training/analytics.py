"""
Analytics and visualization for training progress.
Location: training/analytics.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict


class TrainingAnalyzer:
    """Analyzes training data and generates insights."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.game_data = []
        self.generation_stats = {}
        
    def load_training_data(self):
        """Load all training data from directory."""
        # Load all games
        all_games_path = self.data_dir / "all_games.json"
        if all_games_path.exists():
            with open(all_games_path, 'r') as f:
                self.game_data = json.load(f)
        
        # Load generation stats
        for gen_dir in sorted(self.data_dir.glob("generation_*")):
            gen_num = int(gen_dir.name.split("_")[1])
            stats_path = gen_dir / "stats.json"
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.generation_stats[gen_num] = json.load(f)
    
    def analyze_learning_curves(self) -> pd.DataFrame:
        """Generate learning curves showing model improvement."""
        curves_data = []
        
        for gen, stats in sorted(self.generation_stats.items()):
            for model_id, model_stats in stats.items():
                curves_data.append({
                    'generation': gen,
                    'model_id': model_id,
                    'win_rate': model_stats.get('win_rate', 0),
                    'avg_score': model_stats.get('avg_score_per_game', 0),
                    'estimation_accuracy': model_stats.get('estimation_accuracy', 0),
                    'games_played': model_stats.get('games_played', 0)
                })
        
        return pd.DataFrame(curves_data)
    
    def analyze_strategy_evolution(self) -> Dict[str, pd.DataFrame]:
        """Analyze how strategies evolve over generations."""
        strategy_data = defaultdict(list)
        
        for game in self.game_data:
            gen = int(game['game_id'].split('_')[0].replace('gen', ''))
            
            for round_data in game['rounds']:
                # Analyze bidding patterns
                if 'declarer_id' in round_data and round_data['declarer_id'] is not None:
                    strategy_data['bidding'].append({
                        'generation': gen,
                        'bid_amount': round_data.get('declarer_bid', 0),
                        'trump_suit': round_data.get('trump_suit', 'Unknown'),
                        'round_type': round_data.get('round_type', 'Unknown')
                    })
                
                # Analyze estimation patterns
                total_est = sum(round_data['estimations'].values())
                strategy_data['estimation'].append({
                    'generation': gen,
                    'total_estimation': total_est,
                    'over_round': total_est > 13,
                    'under_round': total_est < 13,
                    'exact_round': total_est == 13,
                    'dash_count': len(round_data.get('dash_players', [])),
                    'with_count': len(round_data.get('with_players', []))
                })
        
        return {k: pd.DataFrame(v) for k, v in strategy_data.items()}
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot learning curves for all metrics."""
        curves_df = self.analyze_learning_curves()
        
        if curves_df.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Learning Curves Over Generations', fontsize=16)
        
        # Win rate
        ax = axes[0, 0]
        for model_id in curves_df['model_id'].unique():
            model_data = curves_df[curves_df['model_id'] == model_id]
            ax.plot(model_data['generation'], model_data['win_rate'], 
                   marker='o', label=model_id[:15])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Average score
        ax = axes[0, 1]
        for model_id in curves_df['model_id'].unique():
            model_data = curves_df[curves_df['model_id'] == model_id]
            ax.plot(model_data['generation'], model_data['avg_score'], 
                   marker='s', label=model_id[:15])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Score')
        ax.set_title('Score Evolution')
        
        # Estimation accuracy
        ax = axes[1, 0]
        for model_id in curves_df['model_id'].unique():
            model_data = curves_df[curves_df['model_id'] == model_id]
            ax.plot(model_data['generation'], model_data['estimation_accuracy'], 
                   marker='^', label=model_id[:15])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Estimation Accuracy')
        ax.set_title('Prediction Accuracy Evolution')
        
        # Games played (bar chart)
        ax = axes[1, 1]
        gen_games = curves_df.groupby('generation')['games_played'].sum()
        ax.bar(gen_games.index, gen_games.values)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Total Games Played')
        ax.set_title('Games Per Generation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    
    def plot_strategy_evolution(self, save_path: Optional[str] = None):
        """Plot how strategies evolve over time."""
        strategy_dfs = self.analyze_strategy_evolution()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Strategy Evolution Over Generations', fontsize=16)
        
        # Bidding patterns
        if 'bidding' in strategy_dfs and not strategy_dfs['bidding'].empty:
            ax = axes[0, 0]
            bid_df = strategy_dfs['bidding']
            bid_avg = bid_df.groupby('generation')['bid_amount'].mean()
            ax.plot(bid_avg.index, bid_avg.values, marker='o', linewidth=2)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Average Bid Amount')
            ax.set_title('Bidding Aggressiveness')
        
        # Round type distribution
        if 'estimation' in strategy_dfs and not strategy_dfs['estimation'].empty:
            ax = axes[0, 1]
            est_df = strategy_dfs['estimation']
            round_types = est_df.groupby('generation')[['over_round', 'under_round']].mean()
            round_types.plot(kind='bar', ax=ax, stacked=True)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Proportion')
            ax.set_title('Over vs Under Round Distribution')
            ax.legend(['Over', 'Under'])
        
        # Special calls usage
        if 'estimation' in strategy_dfs and not strategy_dfs['estimation'].empty:
            ax = axes[1, 0]
            special_calls = est_df.groupby('generation')[['dash_count', 'with_count']].mean()
            special_calls.plot(ax=ax, marker='o')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Average Count per Round')
            ax.set_title('Special Calls Usage')
            ax.legend(['DASH', 'WITH'])
        
        # Estimation variance
        if 'estimation' in strategy_dfs and not strategy_dfs['estimation'].empty:
            ax = axes[1, 1]
            est_variance = est_df.groupby('generation')['total_estimation'].std()
            ax.plot(est_variance.index, est_variance.values, marker='s', color='red')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Standard Deviation')
            ax.set_title('Estimation Variance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
    
    def generate_report(self, output_path: str):
        """Generate comprehensive training report."""
        report = []
        report.append("# Estimation Bot Training Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Training Summary")
        
        # Basic stats
        report.append(f"- Total games played: {len(self.game_data)}")
        report.append(f"- Generations completed: {len(self.generation_stats)}")
        
        # Performance trends
        curves_df = self.analyze_learning_curves()
        if not curves_df.empty:
            latest_gen = curves_df['generation'].max()
            latest_stats = curves_df[curves_df['generation'] == latest_gen]
            
            report.append("\n## Latest Generation Performance")
            for _, row in latest_stats.iterrows():
                report.append(f"\n### {row['model_id']}")
                report.append(f"- Win Rate: {row['win_rate']:.2%}")
                report.append(f"- Average Score: {row['avg_score']:.1f}")
                report.append(f"- Estimation Accuracy: {row['estimation_accuracy']:.2%}")
        
        # Strategy insights
        strategy_dfs = self.analyze_strategy_evolution()
        if 'estimation' in strategy_dfs and not strategy_dfs['estimation'].empty:
            est_df = strategy_dfs['estimation']
            latest_est = est_df[est_df['generation'] == latest_gen]
            
            report.append("\n## Strategy Analysis")
            report.append(f"- Over rounds: {latest_est['over_round'].mean():.2%}")
            report.append(f"- Under rounds: {latest_est['under_round'].mean():.2%}")
            report.append(f"- Avg DASH calls per round: {latest_est['dash_count'].mean():.2f}")
            report.append(f"- Avg WITH calls per round: {latest_est['with_count'].mean():.2f}")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to: {output_path}")
    
    def export_training_dataset(self, output_path: str):
        """Export training data in format suitable for supervised learning."""
        training_data = []
        
        for game in self.game_data:
            for round_idx, round_data in enumerate(game['rounds']):
                # Extract features for each player's decision
                for player_id in range(4):
                    # Features
                    features = {
                        'game_id': game['game_id'],
                        'round_number': round_data['round_number'],
                        'player_id': player_id,
                        'is_speed_round': round_data.get('is_speed_round', False),
                        'trump_suit': round_data.get('trump_suit', 'No Trump'),
                        'is_declarer': player_id == round_data.get('declarer_id'),
                        'declarer_bid': round_data.get('declarer_bid', 0),
                        
                        # Previous round context
                        'current_score': game['final_scores'].get(str(player_id), 0),
                        'score_rank': 0,  # To be calculated
                        
                        # Labels (what we want to predict)
                        'estimation': round_data['estimations'].get(str(player_id), 0),
                        'actual_tricks': round_data['actual_tricks'].get(str(player_id), 0),
                        'made_estimation': round_data['estimations'].get(str(player_id), 0) == 
                                         round_data['actual_tricks'].get(str(player_id), 0),
                        'round_score': game['rounds'][round_idx].get('scores', {}).get(str(player_id), 0)
                    }
                    
                    training_data.append(features)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(training_data)
        df.to_csv(output_path, index=False)
        print(f"Training dataset exported to: {output_path}")
        print(f"Shape: {df.shape}")
        
        return df