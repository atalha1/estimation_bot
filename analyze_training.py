"""
Analyze training results and generate reports.
Location: analyze_training.py
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from estimation_bot.training.analytics import TrainingAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze Estimation bot training results")
    parser.add_argument('data_dir', type=str,
                       help='Training data directory to analyze')
    parser.add_argument('--plots', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--report', action='store_true',
                       help='Generate text report')
    parser.add_argument('--export', action='store_true',
                       help='Export training dataset for supervised learning')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Directory for output files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"📊 Analyzing training data from: {args.data_dir}")
    
    # Initialize analyzer
    analyzer = TrainingAnalyzer(args.data_dir)
    analyzer.load_training_data()
    
    print(f"Loaded {len(analyzer.game_data)} games")
    print(f"Found {len(analyzer.generation_stats)} generations")
    
    # Generate plots
    if args.plots:
        print("\n📈 Generating plots...")
        
        # Learning curves
        analyzer.plot_learning_curves(
            save_path=str(output_dir / "learning_curves.png")
        )
        
        # Strategy evolution
        analyzer.plot_strategy_evolution(
            save_path=str(output_dir / "strategy_evolution.png")
        )
        
        print(f"Plots saved to {output_dir}")
    
    # Generate report
    if args.report:
        print("\n📝 Generating report...")
        report_path = output_dir / "training_report.md"
        analyzer.generate_report(str(report_path))
    
    # Export dataset
    if args.export:
        print("\n💾 Exporting training dataset...")
        dataset_path = output_dir / "training_dataset.csv"
        df = analyzer.export_training_dataset(str(dataset_path))
        
        # Basic stats
        print(f"\nDataset Statistics:")
        print(f"- Total samples: {len(df)}")
        print(f"- Features: {list(df.columns)}")
        print(f"- Estimation accuracy: {df['made_estimation'].mean():.2%}")
        print(f"- Average score per round: {df['round_score'].mean():.1f}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()