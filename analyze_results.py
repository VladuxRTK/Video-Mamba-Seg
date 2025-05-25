import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def analyze_training_results(results_dir):
    """Analyze and visualize training results."""
    results_dir = Path(results_dir)
    
    # Load JSON results
    json_file = results_dir / 'training_results.json'
    if json_file.exists():
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        print("="*80)
        print("TRAINING ANALYSIS")
        print("="*80)
        print(f"Total epochs: {len(results['training_history'])}")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        
        # Load CSV data for plotting
        train_csv = results_dir / 'training_metrics.csv'
        val_csv = results_dir / 'validation_metrics.csv'
        
        if train_csv.exists():
            train_df = pd.read_csv(train_csv)
            print(f"\nTraining metrics available: {list(train_df.columns)}")
            
            # Plot training curves
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Training loss
            axes[0, 0].plot(train_df['epoch'], train_df['loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
            # Dice loss
            axes[0, 1].plot(train_df['epoch'], train_df['dice_loss'])
            axes[0, 1].set_title('Dice Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Dice Loss')
            axes[0, 1].grid(True)
            
            # Learning rate
            axes[1, 0].plot(train_df['epoch'], train_df['learning_rate'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
            
            # Validation metrics if available
            if val_csv.exists():
                val_df = pd.read_csv(val_csv)
                axes[1, 1].plot(val_df['epoch'], val_df['J&F'], label='J&F', marker='o')
                axes[1, 1].plot(val_df['epoch'], val_df['iou'], label='IoU', marker='s')
                axes[1, 1].set_title('Validation Metrics')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
                
                print(f"\nBest validation J&F: {val_df['J&F'].max():.4f}")
                print(f"Best validation IoU: {val_df['iou'].max():.4f}")
            
            plt.tight_layout()
            plt.savefig(results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nTraining curves saved to: {results_dir / 'training_curves.png'}")
        
        # Print recent validation results
        if results['validation_history']:
            print("\nRecent Validation Results:")
            print("-" * 60)
            for result in results['validation_history'][-5:]:
                print(f"Epoch {result['epoch']:3d}: J&F={result.get('J&F', 0):.4f}, IoU={result.get('iou', 0):.4f}, Loss={result.get('val_loss', 0):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='checkpoints/results', 
                       help='Directory containing training results')
    args = parser.parse_args()
    
    analyze_training_results(args.results_dir)

if __name__ == '__main__':
    main()