import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import pandas as pd
from datetime import datetime

class DataQualityReport:
    """Generates visual and textual quality reports"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate_visual_report(self, stats: Dict[str, int], 
                             before_counts: Dict[str, int], 
                             after_counts: Dict[str, int]):
        """Create visual report with matplotlib"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Cleaning actions
        plt.subplot(2, 2, 1)
        pd.Series(stats).plot.bar(color='skyblue')
        plt.title("Data Cleaning Actions")
        plt.xticks(rotation=45)
        
        # Plot 2: Before vs After
        plt.subplot(2, 2, 2)
        df_counts = pd.DataFrame({'Before': before_counts, 'After': after_counts})
        df_counts.plot.bar(rot=45)
        plt.title("Dataset Size Before/After Cleaning")
        
        # Save figure
        report_path = self.output_dir / f"data_quality_{datetime.now().date()}.png"
        plt.tight_layout()
        plt.savefig(report_path)
        plt.close()
        return report_path

    def generate_text_report(self, stats: Dict[str, int], 
                           before_counts: Dict[str, int], 
                           after_counts: Dict[str, int]) -> Path:
        """Generate textual summary report"""
        report = f"""
        DATA QUALITY REPORT
        {'='*40}
        Generated: {datetime.now()}
        
        CLEANING STATISTICS:
        - Duplicates removed: {stats['duplicates_removed']}
        - Low quality removed: {stats['low_quality_removed']}
        - Corrupted removed: {stats['corrupted_removed']}
        
        DATASET COUNTS:
        Before Cleaning:
        {pd.Series(before_counts).to_string()}
        
        After Cleaning:
        {pd.Series(after_counts).to_string()}
        """
        
        report_path = self.output_dir / f"quality_report_{datetime.now().date()}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        return report_path

class ModelPerformanceReport:
    """Handles model evaluation reporting"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
    def generate_classification_report(self, y_true, y_pred, labels):
        """Generate classification report with visualizations"""
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=labels)
        
        report_path = self.output_dir / "model_performance.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path
    
    def generate_model_performance_report(self, results: list):
        """Extended reporting for model metrics"""
        
        
        # Create comparative visualization
        df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        df.plot(x='model', y=['accuracy', 'f1_score'], kind='bar')
        plt.title("Model Performance Comparison")
        plt.savefig(self.output_dir / "model_comparison.png")
        plt.close()
        
        # Save detailed metrics
        df.to_csv(self.output_dir / "model_metrics.csv", index=False)