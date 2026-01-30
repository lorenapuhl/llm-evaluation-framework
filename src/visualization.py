"""
Visualization module for LLM Evaluation Framework

Provides dashboard visualization, charts, and interactive reports
for evaluating LLM performance across multiple metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class LLMVisualizer:
    """
    Main visualization class for LLM evaluation results.
    
    Handles creation of dashboards, charts, and reports from evaluation data.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize visualizer with output directory.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define color schemes
        self.colors = {
            'primary': '#3F51B5',        # Indigo 500
            'secondary': '#00BCD4',      # Cyan 500
            'success': '#4CAF50',        # Green 500
            'warning': '#FF9800',        # Orange 500
            'danger': '#F44336',         # Red 500
            'neutral': '#9E9E9E',        # Grey 500
            'accuracy': '#2196F3',       # Blue 500
            'relevance': '#8BC34A',      # Light Green 500
            'safety': '#D32F2F',         # Red 700 (darker for safety)
            'quality': '#E1BEE7',        # Purple 100 (softer lavender)
            'overall': '#FAFAFA'         # Grey 50 (softer white)
        }
        
        # Define failure type colors
        self.failure_colors = {
            'factual_error': '#FF6B6B',
            'irrelevant': '#FFD166',
            'unsafe': '#EF476F',
            'poor_quality': '#118AB2',
            'refusal': '#073B4C',
            'partial_accuracy': '#FF9E6D',
            'partial_relevance': '#06D6A0',
            'none': '#888888'
        }
        
        # Define category colors
        self.category_colors = {
            'Factual': '#1E88E5',
            'Explanatory': '#43A047',
            'Instruction': '#FB8C00',
            'Creative': '#8E24AA',
            'Sensitive': '#E53935'
        }
        
    def load_evaluation_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load evaluation results from file.
        
        Args:
            filepath: Path to evaluation results file
            
        Returns:
            DataFrame containing evaluation results
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, "enhanced_evaluation_results.tsv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Evaluation results file not found: {filepath}")
        
        df = pd.read_csv(filepath, sep='\t')
        print(f"Loaded evaluation data with {len(df)} rows")
        return df
    
    def load_failure_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load failure analysis results from file.
        
        Args:
            filepath: Path to failure analysis file
            
        Returns:
            DataFrame containing failure analysis results
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, "enhanced_failure_analysis.tsv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Failure analysis file not found: {filepath}")
        
        df = pd.read_csv(filepath, sep='\t')
        print(f"Loaded failure data with {len(df)} rows")
        return df
    
    def create_dashboard(self, 
                        eval_data: pd.DataFrame = None,
                        failure_data: pd.DataFrame = None,
                        save_path: str = None) -> go.Figure:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            eval_data: Evaluation results DataFrame
            failure_data: Failure analysis DataFrame
            save_path: Path to save dashboard HTML
            
        Returns:
            Plotly Figure object containing dashboard
        """
        if eval_data is None:
            eval_data = self.load_evaluation_data()
        if failure_data is None:
            failure_data = self.load_failure_data()
        
        # Create subplots for dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Overall Score Distribution',
                'Failure Category Breakdown',
                'Performance by Question Category',
                'Composite Metrics Comparison',
                'Failure Confidence Distribution',
                'Score Correlation Heatmap',
                'Top Failure Examples',
                'Accuracy vs Relevance',
                'Quality Metrics Distribution'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'pie'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'histogram'}, {'type': 'heatmap'}],
                [{'type': 'table'}, {'type': 'scatter'}, {'type': 'box'}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Overall Score Distribution
        fig.add_trace(
            go.Histogram(
                x=eval_data['overall_score'],
                nbinsx=20,
                name='Overall Score',
                marker_color=self.colors['primary'],
                opacity=0.7,
                histnorm='percent'
            ),
            row=1, col=1
        )
        
        # Add mean line
        mean_score = eval_data['overall_score'].mean()
        fig.add_vline(
            x=mean_score,
            line_dash="dash",
            line_color="red",
            row=1, col=1,
            annotation_text=f"Mean: {mean_score:.2f}",
            annotation_position="top right"
        )
        
        # 2. Failure Category Breakdown
        if 'primary_failure_mode' in failure_data.columns:
            failure_counts = failure_data['primary_failure_mode'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=failure_counts.index,
                    values=failure_counts.values,
                    hole=0.3,
                    marker_colors=[self.failure_colors.get(f, self.colors['neutral']) 
                                  for f in failure_counts.index],
                    name='Failure Categories'
                ),
                row=1, col=2
            )
        
        # 3. Performance by Question Category
        category_metrics = eval_data.groupby('category').agg({
            'overall_score': 'mean',
            'composite_accuracy': 'mean',
            'composite_relevance': 'mean',
            'composite_safety': 'mean',
            'composite_quality': 'mean'
        }).round(3)
        
        for i, metric in enumerate(['overall_score', 'composite_accuracy', 
                                   'composite_relevance', 'composite_safety', 
                                   'composite_quality']):
            fig.add_trace(
                go.Bar(
                    x=category_metrics.index,
                    y=category_metrics[metric],
                    name=metric.replace('composite_', '').title(),
                    marker_color=[self.category_colors.get(c, self.colors['neutral']) 
                                 for c in category_metrics.index],
                    opacity=0.7,
                    showlegend=False
                ) if i > 0 else go.Bar(
                    x=category_metrics.index,
                    y=category_metrics[metric],
                    name=metric.replace('composite_', '').title(),
                    marker_color=[self.category_colors.get(c, self.colors['neutral']) 
                                 for c in category_metrics.index],
                    opacity=0.7
                ),
                row=1, col=3
            )
        
        # 4. Composite Metrics Comparison
        metrics = ['composite_accuracy', 'composite_relevance', 
                  'composite_safety', 'composite_quality']
        metric_means = eval_data[metrics].mean()
        
        fig.add_trace(
            go.Bar(
                x=[m.replace('composite_', '').title() for m in metrics],
                y=metric_means.values,
                marker_color=[self.colors['accuracy'], self.colors['relevance'],
                            self.colors['safety'], self.colors['quality']],
                text=[f'{v:.2f}' for v in metric_means.values],
                textposition='auto',
                name='Composite Metrics'
            ),
            row=2, col=1
        )
        
        # 5. Failure Confidence Distribution
        if 'failure_confidence' in failure_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=failure_data['failure_confidence'],
                    nbinsx=10,
                    name='Confidence Score',
                    marker_color=self.colors['secondary'],
                    opacity=0.7
                ),
                row=2, col=2
            )
        
        # 6. Score Correlation Heatmap
        correlation_matrix = eval_data[metrics + ['overall_score']].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 10},
                name='Correlation'
            ),
            row=2, col=3
        )
        
        # 7. Top Failure Examples (as table)
        if 'primary_failure_mode' in failure_data.columns and 'question' in failure_data.columns:
            failures = failure_data[failure_data['primary_failure_mode']!= 'pass']
            top_failures = failures.nlargest(5, 'failure_confidence')[['question', 'primary_failure_mode', 'failure_confidence']]
            top_failures = failure_data.nlargest(5, 'failure_confidence')[['question', 'primary_failure_mode', 'failure_confidence']]
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Question', 'Failure Mode', 'Confidence'],
                        fill_color=self.colors['primary'],
                        align='left',
                        font=dict(color='white', size=12)
                    ),
                    cells=dict(
                        values=[top_failures['question'], 
                               top_failures['primary_failure_mode'],
                               top_failures['failure_confidence'].round(2)],
                        fill_color='white',
                        align='left',
                        font=dict(color='black', size=10),
                        height=30
                    )
                ),
                row=3, col=1
            )
        
        # 8. Accuracy vs Relevance Scatter
        fig.add_trace(
            go.Scatter(
                x=eval_data['composite_accuracy'],
                y=eval_data['composite_relevance'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=eval_data['overall_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Overall Score")
                ),
                text=eval_data['question'],
                hoverinfo='text',
                name='Accuracy vs Relevance'
            ),
            row=3, col=2
        )
        
        # Add trendline
        z = np.polyfit(eval_data['composite_accuracy'], 
                      eval_data['composite_relevance'], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=eval_data['composite_accuracy'].sort_values(),
                y=p(eval_data['composite_accuracy'].sort_values()),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Trendline'
            ),
            row=3, col=2
        )
        
        # 9. Quality Metrics Distribution
        quality_metrics = ['composite_accuracy', 'composite_relevance', 
                         'composite_safety', 'composite_quality']
        
        fig.add_trace(
            go.Box(
                y=[eval_data[metric] for metric in quality_metrics],
                x=[metric.replace('composite_', '').title() 
                   for metric in quality_metrics],
                boxpoints='outliers',
                #marker_color=[self.colors['accuracy'], self.colors['relevance'], self.colors['safety'], self.colors['quality']],
                marker_color = self.colors['neutral'],
                name='Quality Metrics'
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="LLM Evaluation Dashboard",
            title_font_size=24,
            title_x=0.5,
            height=1200,
            width=1400,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Percentage", row=1, col=1)
        fig.update_xaxes(title_text="Category", row=1, col=3)
        fig.update_yaxes(title_text="Score", row=1, col=3)
        fig.update_xaxes(title_text="Metric", row=2, col=1)
        fig.update_yaxes(title_text="Average Score", row=2, col=1)
        fig.update_xaxes(title_text="Confidence Score", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_xaxes(title_text="Accuracy", row=3, col=2)
        fig.update_yaxes(title_text="Relevance", row=3, col=2)
        
        # Save dashboard
        if save_path is None:
            save_path = os.path.join(self.output_dir, "llm_evaluation_dashboard.html")
        
        fig.write_html(save_path)
        print(f"Dashboard saved to: {save_path}")
        
        return fig
    
    def plot_score_distribution(self, 
                               eval_data: pd.DataFrame = None,
                               save_path: str = None) -> plt.Figure:
        """
        Create detailed score distribution histogram.
        
        Args:
            eval_data: Evaluation results DataFrame
            save_path: Path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        if eval_data is None:
            eval_data = self.load_evaluation_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Score Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall Score Distribution
        ax1 = axes[0, 0]
        ax1.hist(eval_data['overall_score'], bins=20, alpha=0.7, 
                color=self.colors['primary'], edgecolor='black')
        ax1.axvline(eval_data['overall_score'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {eval_data["overall_score"].mean():.2f}')
        ax1.axvline(eval_data['overall_score'].median(), color='green', 
                   linestyle='--', linewidth=2, label=f'Median: {eval_data["overall_score"].median():.2f}')
        ax1.set_xlabel('Overall Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Composite Metrics Distribution
        ax2 = axes[0, 1]
        metrics = ['composite_accuracy', 'composite_relevance', 
                  'composite_safety', 'composite_quality']
        metric_names = [m.replace('composite_', '').title() for m in metrics]
        metric_values = [eval_data[m] for m in metrics]
        
        bp = ax2.boxplot(metric_values, labels=metric_names, patch_artist=True)
        
        # Color the boxes
        colors = [self.colors['accuracy'], self.colors['relevance'], 
                 self.colors['safety'], self.colors['quality']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Score')
        ax2.set_title('Composite Metrics Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Score Density Plot
        ax3 = axes[1, 0]
        for metric, color, label in zip(metrics, colors, metric_names):
            sns.kdeplot(data=eval_data[metric], ax=ax3, color=color, 
                       label=label, linewidth=2)
        
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Score Density by Metric')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Distribution
        ax4 = axes[1, 1]
        sorted_scores = np.sort(eval_data['overall_score'])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        
        ax4.plot(sorted_scores, cumulative, color=self.colors['primary'], 
                linewidth=3)
        ax4.fill_between(sorted_scores, cumulative, alpha=0.3, 
                        color=self.colors['secondary'])
        
        # Add quartile lines
        for q in [0.25, 0.5, 0.75]:
            q_value = np.percentile(eval_data['overall_score'], q * 100)
            ax4.axvline(q_value, color='red', linestyle='--', alpha=0.7)
            ax4.text(q_value, 0.5, f'Q{q*100:.0f}: {q_value:.2f}', 
                    rotation=90, verticalalignment='center')
        
        ax4.set_xlabel('Overall Score')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Function')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "score_distribution.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Score distribution plot saved to: {save_path}")
        
        return fig
    
    def plot_failure_breakdown(self, 
                              failure_data: pd.DataFrame = None,
                              save_path: str = None) -> plt.Figure:
        """
        Create comprehensive failure breakdown visualization.
        
        Args:
            failure_data: Failure analysis DataFrame
            save_path: Path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        if failure_data is None:
            failure_data = self.load_failure_data()
        
        if 'primary_failure_mode' not in failure_data.columns:
            print("No failure mode data found")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Failure Analysis Breakdown', fontsize=16, fontweight='bold')
        
        # 1. Failure Mode Pie Chart
        ax1 = axes[0, 0]
        failure_counts = failure_data['primary_failure_mode'].value_counts()
        
        # Prepare colors for pie chart
        pie_colors = [self.failure_colors.get(f, self.colors['neutral']) 
                     for f in failure_counts.index]
        
        wedges, texts, autotexts = ax1.pie(
            failure_counts.values,
            labels=failure_counts.index,
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,
            textprops={'fontsize': 10}
        )
        
        # Make percentages white and bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        # Draw circle for donut chart
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        ax1.set_title('Failure Mode Distribution')
        
        # 2. Failure by Category (if category column exists)
        ax2 = axes[0, 1]
        if 'category' in failure_data.columns:
            category_failure = pd.crosstab(
                failure_data['category'], 
                failure_data['primary_failure_mode']
            )
            
            # Sort by total failures
            category_failure = category_failure.loc[category_failure.sum(axis=1).sort_values(ascending=False).index]
            category_failure = category_failure[category_failure.columns[::-1]]
            
            bottom = np.zeros(len(category_failure))
            for i, failure_mode in enumerate(category_failure.columns):
                color = self.failure_colors.get(failure_mode, self.colors['neutral'])
                ax2.barh(category_failure.index, category_failure[failure_mode], 
                        left=bottom, color=color, label=failure_mode, alpha=0.8)
                bottom += category_failure[failure_mode].values
            
            ax2.set_xlabel('Number of Failures')
            ax2.set_ylabel('Question Category')
            ax2.set_title('Failures by Question Category')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Simple bar chart if no category
            failure_counts.plot(kind='barh', ax=ax2, color=pie_colors, alpha=0.8)
            ax2.set_xlabel('Count')
            ax2.set_ylabel('Failure Mode')
            ax2.set_title('Failure Mode Counts')
        
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence Score Distribution by Failure Mode
        ax3 = axes[1, 0]
        if 'failure_confidence' in failure_data.columns:
            failure_data.boxplot(
                column='failure_confidence',
                by='primary_failure_mode',
                ax=ax3,
                patch_artist=True,
                boxprops=dict(facecolor=self.colors['secondary'], alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markersize=5, alpha=0.5)
            )
            
            ax3.set_xlabel('Failure Mode')
            ax3.set_ylabel('Confidence Score')
            ax3.set_title('Confidence Score by Failure Mode')
            plt.suptitle('')  # Remove automatic subtitle
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No confidence score data', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Confidence Score Distribution')
        
        # 4. Failure Trend Analysis (if multiple models or time data)
        ax4 = axes[1, 1]
        
        # Create a heatmap-like visualization
        if len(failure_counts) > 0:
            # Prepare data for heatmap
            failure_scores = {}
            for failure_mode in failure_counts.index:
                if failure_mode != 'pass':
                    mode_data = failure_data[failure_data['primary_failure_mode'] == failure_mode]
                    failure_scores[failure_mode] = {
                        'Count': len(mode_data),
                        'Avg Confidence': mode_data['failure_confidence'].mean() if 'failure_confidence' in mode_data else 0,
                        'Severity': len(mode_data) / len(failure_data)  # Relative frequency
                    }
            
            if failure_scores:
                heatmap_data = pd.DataFrame(failure_scores).T
                
                # Normalize for heatmap
                normalized_data = heatmap_data.copy()
                for col in normalized_data.columns:
                    normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                                          (normalized_data[col].max() - normalized_data[col].min())
                
                im = ax4.imshow(normalized_data.values, cmap='YlOrRd', aspect='auto')
                
                # Add text annotations
                for i in range(len(heatmap_data)):
                    for j in range(len(heatmap_data.columns)):
                        ax4.text(j, i, f'{heatmap_data.iloc[i, j]:.2f}',
                                ha='center', va='center', color='black',
                                fontweight='bold' if heatmap_data.iloc[i, j] > heatmap_data.values.mean() else 'normal')
                
                ax4.set_xticks(range(len(heatmap_data.columns)))
                ax4.set_xticklabels(heatmap_data.columns, rotation=45)
                ax4.set_yticks(range(len(heatmap_data.index)))
                ax4.set_yticklabels(heatmap_data.index)
                ax4.set_title('Failure Mode Analysis Heatmap')
                
                # Add colorbar
                plt.colorbar(im, ax=ax4)
            else:
                ax4.text(0.5, 0.5, 'No failure data for heatmap', 
                        ha='center', va='center', fontsize=12)
                ax4.set_title('Failure Analysis Heatmap')
        else:
            ax4.text(0.5, 0.5, 'No failure data available', 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Failure Analysis Heatmap')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "failure_breakdown.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Failure breakdown plot saved to: {save_path}")
        
        return fig
    
    def plot_category_performance(self, 
                                 eval_data: pd.DataFrame = None,
                                 save_path: str = None) -> plt.Figure:
        """
        Create visualization of performance by question category.
        
        Args:
            eval_data: Evaluation results DataFrame
            save_path: Path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        if eval_data is None:
            eval_data = self.load_evaluation_data()
        
        if 'category' not in eval_data.columns:
            print("No category data found")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Performance Analysis by Question Category', 
                    fontsize=16, fontweight='bold')
        
        # 1. Overall Score by Category
        ax1 = axes[0, 0]
        category_scores = eval_data.groupby('category')['overall_score'].agg(['mean', 'std', 'count'])
        categories = category_scores.index.tolist()
        colors = [self.category_colors.get(c, self.colors['neutral']) for c in categories]
        
        bars = ax1.bar(categories, category_scores['mean'], 
                      yerr=category_scores['std'], 
                      capsize=5, alpha=0.8, color=colors, edgecolor='black')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, category_scores['mean'], category_scores['std']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.2f} (±{std:.2f})', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Question Category')
        ax1.set_ylabel('Average Overall Score')
        ax1.set_title('Overall Performance by Category')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.1)
        
        # 2. Composite Metrics by Category
        ax2 = axes[0, 1]
        metrics = ['composite_accuracy', 'composite_relevance', 
                  'composite_safety', 'composite_quality']
        metric_names = [m.replace('composite_', '').title() for m in metrics]
        
        # Prepare data for grouped bar chart
        category_metrics = eval_data.groupby('category')[metrics].mean()
        
        x = np.arange(len(categories))
        width = 0.2
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            offset = (i - 1.5) * width
            ax2.bar(x + offset, category_metrics[metric], width, 
                   label=metric_name, alpha=0.8)
        
        ax2.set_xlabel('Question Category')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Composite Metrics by Category')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1.1)
        
        # 3. Radar Chart of Category Performance
        ax3 = axes[1, 0]
        
        # Prepare data for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        for category in categories:
            values = category_metrics.loc[category].tolist()
            values += values[:1]  # Close the polygon
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=category, 
                    color=self.category_colors.get(category, self.colors['neutral']))
            ax3.fill(angles, values, alpha=0.1)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(metric_names)
        ax3.set_ylim(0, 1)
        ax3.set_title('Category Performance Radar Chart')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Pass Rate by Category
        ax4 = axes[1, 1]
        
        # Calculate pass rates for each category
        pass_columns = ['passed_accuracy', 'passed_relevance', 
                       'passed_safety', 'passed_quality']
        pass_rates = {}
        
        for category in categories:
            category_data = eval_data[eval_data['category'] == category]
            pass_rates[category] = {}
            for pass_col in pass_columns:
                if pass_col in category_data.columns:
                    pass_rates[category][pass_col.replace('passed_', '').title()] = \
                        category_data[pass_col].mean()
        
        pass_df = pd.DataFrame(pass_rates).T
        
        # Create stacked bar chart
        bottom = np.zeros(len(categories))
        for i, col in enumerate(pass_df.columns):
            ax4.bar(categories, pass_df[col], bottom=bottom, 
                   label=col, alpha=0.7)
            bottom += pass_df[col].values
        
        ax4.set_xlabel('Question Category')
        ax4.set_ylabel('Pass Rate')
        ax4.set_title('Pass Rate Breakdown by Category')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.set_ylim(0, 4)  # Max 4 passed metrics
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xticklabels(categories, rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "category_performance.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Category performance plot saved to: {save_path}")
        
        return fig
    
    def plot_metric_correlations(self, 
                                eval_data: pd.DataFrame = None,
                                save_path: str = None) -> plt.Figure:
        """
        Create correlation heatmap and scatter matrix.
        
        Args:
            eval_data: Evaluation results DataFrame
            save_path: Path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        if eval_data is None:
            eval_data = self.load_evaluation_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Metric Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Prepare metrics for correlation analysis
        metrics = ['overall_score', 'composite_accuracy', 'composite_relevance', 
                  'composite_safety', 'composite_quality']
        metric_names = [m.replace('composite_', '').title() 
                       if 'composite' in m else m.title() for m in metrics]
        
        # 1. Correlation Heatmap
        ax1 = axes[0, 0]
        correlation_matrix = eval_data[metrics].corr()
        
        im = ax1.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                value = correlation_matrix.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax1.text(j, i, f'{value:.2f}', ha='center', va='center', 
                        color=color, fontweight='bold')
        
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels(metric_names, rotation=45)
        ax1.set_yticks(range(len(metrics)))
        ax1.set_yticklabels(metric_names)
        ax1.set_title('Metric Correlation Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax1)
        
        # 2. Scatter Matrix (simplified)
        ax2 = axes[0, 1]
        # Show correlation of overall score with other metrics
        scatter_metrics = metrics[1:]  # Exclude overall_score
        
        for i, metric in enumerate(scatter_metrics):
            color = [self.colors['accuracy'], self.colors['relevance'],
                    self.colors['safety'], self.colors['quality']][i]
            
            ax2.scatter(eval_data[metric], eval_data['overall_score'], 
                       alpha=0.5, color=color, label=metric.replace('composite_', '').title())
            
            # Add trendline
            z = np.polyfit(eval_data[metric], eval_data['overall_score'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(eval_data[metric].min(), eval_data[metric].max(), 100)
            ax2.plot(x_range, p(x_range), '--', color=color, alpha=0.8)
        
        ax2.set_xlabel('Metric Scores')
        ax2.set_ylabel('Overall Score')
        ax2.set_title('Overall Score vs Individual Metrics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Pairwise Scatter Plots (selected pairs)
        ax3 = axes[1, 0]
        # Show accuracy vs relevance and safety vs quality
        scatter_pairs = [
            ('composite_accuracy', 'composite_relevance'),
            ('composite_safety', 'composite_quality')
        ]
        
        colors = [self.colors['accuracy'], self.colors['safety']]
        labels = ['Accuracy vs Relevance', 'Safety vs Quality']
        
        for i, (x_metric, y_metric) in enumerate(scatter_pairs):
            x_name = x_metric.replace('composite_', '').title()
            y_name = y_metric.replace('composite_', '').title()
            
            ax3.scatter(eval_data[x_metric], eval_data[y_metric], 
                       alpha=0.5, color=colors[i], label=labels[i])
            
            # Add trendline
            z = np.polyfit(eval_data[x_metric], eval_data[y_metric], 1)
            p = np.poly1d(z)
            x_range = np.linspace(eval_data[x_metric].min(), eval_data[x_metric].max(), 100)
            ax3.plot(x_range, p(x_range), '--', color=colors[i], alpha=0.8)
        
        ax3.set_xlabel('Metric Scores')
        ax3.set_ylabel('Metric Scores')
        ax3.set_title('Key Metric Pairwise Relationships')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Correlation with Failure (if available)
        ax4 = axes[1, 1]
        
        # Check if we have failure data
        if 'primary_failure_mode' in eval_data.columns:
            # Convert failure mode to numeric score (higher = worse)
            failure_scores = pd.get_dummies(eval_data['primary_failure_mode'])
            
            # Calculate correlation of metrics with different failure types
            failure_corr = {}
            for metric in metrics:
                for failure_type in failure_scores.columns:
                    if failure_type != 'pass':
                        corr = eval_data[metric].corr(failure_scores[failure_type])
                        failure_corr[f'{metric}_{failure_type}'] = corr
            
            # Take top correlations
            top_corrs = pd.Series(failure_corr).nlargest(10)
            
            if len(top_corrs) > 0:
                y_pos = np.arange(len(top_corrs))
                colors = ['red' if c > 0 else 'blue' for c in top_corrs.values]
                
                bars = ax4.barh(y_pos, top_corrs.values, color=colors, alpha=0.7)
                
                # Add value labels
                for bar, value in zip(bars, top_corrs.values):
                    width = bar.get_width()
                    ax4.text(width if width > 0 else width - 0.05, 
                            bar.get_y() + bar.get_height()/2,
                            f'{value:.2f}', ha='left' if width > 0 else 'right',
                            va='center', fontsize=9)
                
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels([t.replace('_', ' ').title() for t in top_corrs.index])
                ax4.set_xlabel('Correlation Coefficient')
                ax4.set_title('Top Metric-Failure Correlations')
                ax4.axvline(x=0, color='black', linewidth=0.5)
            else:
                ax4.text(0.5, 0.5, 'No failure correlation data', 
                        ha='center', va='center', fontsize=12)
                ax4.set_title('Metric-Failure Correlations')
        else:
            # Alternative: Distribution of correlations
            ax4.text(0.5, 0.5, 'No failure mode data for correlation', 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Metric-Failure Correlations')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "metric_correlations.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metric correlations plot saved to: {save_path}")
        
        return fig
    
    def plot_top_failure_examples(self, 
                                 failure_data: pd.DataFrame = None,
                                 eval_data: pd.DataFrame = None,
                                 n_examples: int = 10,
                                 save_path: str = None) -> plt.Figure:
        """
        Create visualization of top failure examples.
        
        Args:
            failure_data: Failure analysis DataFrame
            eval_data: Evaluation results DataFrame
            n_examples: Number of examples to show
            save_path: Path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        if failure_data is None:
            failure_data = self.load_failure_data()
        
        if eval_data is None:
            eval_data = self.load_evaluation_data()
        
        if 'primary_failure_mode' not in failure_data.columns:
            print("No failure mode data found")
            return None
        
        # Merge evaluation and failure data if needed
        if 'overall_score' not in failure_data.columns and 'id' in failure_data.columns:
            failure_data = failure_data.merge(
                eval_data[['id', 'overall_score', 'question', 'response']], 
                on='id', how='left'
            )
        
        # Get top failure examples by confidence score
        top_failures = failure_data.nlargest(n_examples, 'failure_confidence')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Top {n_examples} Failure Examples Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Summary bar chart
        ax1 = axes[0]
        
        failure_types = top_failures['primary_failure_mode'].value_counts()
        colors = [self.failure_colors.get(f, self.colors['neutral']) 
                 for f in failure_types.index]
        
        bars = ax1.bar(range(len(failure_types)), failure_types.values, 
                      color=colors, alpha=0.8, edgecolor='black')
        
        # Add labels and counts
        for i, (bar, count) in enumerate(zip(bars, failure_types.values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xticks(range(len(failure_types)))
        ax1.set_xticklabels(failure_types.index, rotation=45)
        ax1.set_ylabel('Number of Examples')
        ax1.set_title('Failure Types in Top Examples')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Detailed table view
        ax2 = axes[1]
        ax2.axis('tight')
        ax2.axis('off')
        
        # Prepare table data
        table_data = []
        for _, row in top_failures.iterrows():
            # Truncate long text for display
            question = str(row.get('question', ''))[:80] + '...' if len(str(row.get('question', ''))) > 80 else str(row.get('question', ''))
            failure_mode = str(row.get('primary_failure_mode', 'Unknown'))
            confidence = f"{row.get('failure_confidence', 0):.2f}"
            score = f"{row.get('overall_score', 0):.2f}"
            
            table_data.append([question, failure_mode, confidence, score])
        
        # Create table
        table = ax2.table(
            cellText=table_data,
            colLabels=['Question', 'Failure Mode', 'Confidence', 'Overall Score'],
            cellLoc='left',
            loc='center',
            colWidths=[0.5, 0.15, 0.1, 0.1]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Color header
        for i in range(4):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows based on failure mode
        for i in range(len(table_data)):
            failure_mode = table_data[i][1]
            color = self.failure_colors.get(failure_mode, '#FFFFFF')
            for j in range(4):
                table[(i + 1, j)].set_facecolor(color)
                table[(i + 1, j)].set_alpha(0.7)
        
        ax2.set_title('Detailed Failure Examples')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "top_failure_examples.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top failure examples plot saved to: {save_path}")
        
        return fig
    
    def generate_interactive_report(self, 
                                   eval_data: pd.DataFrame = None,
                                   failure_data: pd.DataFrame = None,
                                   save_path: str = None) -> str:
        """
        Generate an interactive HTML report with all visualizations.
        
        Args:
            eval_data: Evaluation results DataFrame
            failure_data: Failure analysis DataFrame
            save_path: Path to save HTML report
            
        Returns:
            Path to saved HTML report
        """
        if eval_data is None:
            eval_data = self.load_evaluation_data()
        if failure_data is None:
            failure_data = self.load_failure_data()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "llm_evaluation_report.html")
        
        # Calculate summary statistics
        summary_stats = {
            'Total Questions': len(eval_data),
            'Average Overall Score': f"{eval_data['overall_score'].mean():.3f}",
            'Median Overall Score': f"{eval_data['overall_score'].median():.3f}",
            'Standard Deviation': f"{eval_data['overall_score'].std():.3f}",
            'Minimum Score': f"{eval_data['overall_score'].min():.3f}",
            'Maximum Score': f"{eval_data['overall_score'].max():.3f}",
            'Accuracy Pass Rate': f"{eval_data['passed_accuracy'].mean() * 100:.1f}%" if 'passed_accuracy' in eval_data.columns else "N/A",
            'Relevance Pass Rate': f"{eval_data['passed_relevance'].mean() * 100:.1f}%" if 'passed_relevance' in eval_data.columns else "N/A",
            'Safety Pass Rate': f"{eval_data['passed_safety'].mean() * 100:.1f}%" if 'passed_safety' in eval_data.columns else "N/A",
            'Quality Pass Rate': f"{eval_data['passed_quality'].mean() * 100:.1f}%" if 'passed_quality' in eval_data.columns else "N/A"
        }
        
        # Failure statistics
        if 'primary_failure_mode' in failure_data.columns:
            failure_stats = failure_data['primary_failure_mode'].value_counts().to_dict()
            failure_summary = {k: v for k, v in failure_stats.items() if k != 'pass'}
            summary_stats['Total Failures'] = sum(failure_summary.values())
            summary_stats['Failure Rate'] = f"{(sum(failure_summary.values()) / len(failure_data)) * 100:.1f}%"
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: {self.colors['primary']}; border-bottom: 3px solid {self.colors['secondary']}; padding-bottom: 10px; }}
                h2 {{ color: {self.colors['primary']}; margin-top: 30px; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
                .stat-card {{ background-color: white; padding: 15px; border-radius: 5px; border-left: 4px solid {self.colors['secondary']}; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .stat-card h3 {{ margin-top: 0; color: {self.colors['primary']}; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: {self.colors['primary']}; }}
                .visualization {{ margin: 30px 0; text-align: center; }}
                .visualization img {{ max-width: 100%; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .section {{ margin: 40px 0; }}
                .failure-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .failure-table th, .failure-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                .failure-table th {{ background-color: {self.colors['primary']}; color: white; }}
                .failure-table tr:hover {{ background-color: #f5f5f5; }}
                .failure-type {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-size: 12px; margin: 2px; }}
                .btn {{ display: inline-block; padding: 10px 20px; background-color: {self.colors['primary']}; color: white; text-decoration: none; border-radius: 5px; margin: 10px 5px; }}
                .btn:hover {{ background-color: {self.colors['secondary']}; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 LLM Evaluation Framework - Comprehensive Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="summary">
                    <h2>📈 Executive Summary</h2>
                    <div class="summary-grid">
        """
        
        # Add summary cards
        for key, value in summary_stats.items():
            html_content += f"""
                        <div class="stat-card">
                            <h3>{key}</h3>
                            <div class="stat-value">{value}</div>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Overall Performance Dashboard</h2>
                    <div class="visualization">
                        <p><em>Interactive dashboard available as separate HTML file: llm_evaluation_dashboard.html</em></p>
                        <a href="llm_evaluation_dashboard.html" class="btn">Open Interactive Dashboard</a>
                    </div>
                </div>
        """
        
        # Add failure analysis section
        if 'primary_failure_mode' in failure_data.columns:
            failure_counts = failure_data['primary_failure_mode'].value_counts()
            total_failures = failure_counts.sum() - failure_counts.get('pass', 0)
            
            html_content += f"""
                <div class="section">
                    <h2>⚠️ Failure Analysis</h2>
                    <p>Total failures detected: <strong>{total_failures}</strong> out of {len(failure_data)} questions</p>
                    
                    <table class="failure-table">
                        <tr>
                            <th>Failure Type</th>
                            <th>Count</th>
                            <th>Percentage</th>
                            <th>Description</th>
                        </tr>
            """
            
            # Define failure descriptions
            failure_descriptions = {
                'factual error': 'Incorrect or inaccurate information provided',
                'irrelevant': 'Response does not address the question',
                'unsafe': 'Contains bias, harmful content, or misinformation',
                'poor_quality': 'Poorly written, unclear, or unhelpful response',
                'refusal to answer': 'LLM refused to answer the question',
                'partial accuracy': 'Partially correct but incomplete',
                'partial relevance': 'Partially relevant but off-topic elements'
            }
            
            for failure_type, count in failure_counts.items():
                if failure_type != 'pass':
                    percentage = (count / total_failures) * 100
                    description = failure_descriptions.get(failure_type, 'Unknown failure type')
                    color = self.failure_colors.get(failure_type, '#888888')
                    
                    html_content += f"""
                        <tr>
                            <td><span class="failure-type" style="background-color: {color};">{failure_type.replace('_', ' ').title()}</span></td>
                            <td>{count}</td>
                            <td>{percentage:.1f}%</td>
                            <td>{description}</td>
                        </tr>
                    """
            
            html_content += """
                    </table>
                </div>
            """
        
        # Add category performance section
        if 'category' in eval_data.columns:
            category_performance = eval_data.groupby('category')['overall_score'].agg(['mean', 'count']).round(3)
            
            html_content += """
                <div class="section">
                    <h2>📋 Performance by Question Category</h2>
                    <table class="failure-table">
                        <tr>
                            <th>Category</th>
                            <th>Average Score</th>
                            <th>Number of Questions</th>
                            <th>Performance Level</th>
                        </tr>
            """
            
            for category, row in category_performance.iterrows():
                score = row['mean']
                count = row['count']
                color = self.category_colors.get(category, '#888888')
                
                if score >= 0.8:
                    level = "Excellent"
                    level_color = "#28a745"
                elif score >= 0.6:
                    level = "Good"
                    level_color = "#17a2b8"
                elif score >= 0.4:
                    level = "Fair"
                    level_color = "#ffc107"
                else:
                    level = "Poor"
                    level_color = "#dc3545"
                
                html_content += f"""
                    <tr>
                        <td><span style="color: {color}; font-weight: bold;">{category}</span></td>
                        <td>{score:.3f}</td>
                        <td>{count}</td>
                        <td><span style="color: {level_color}; font-weight: bold;">{level}</span></td>
                    </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        # Add top failure examples
        if 'primary_failure_mode' in failure_data.columns:
            top_failures = failure_data.nlargest(5, 'failure_confidence')
            
            html_content += """
                <div class="section">
                    <h2>🔍 Top Failure Examples</h2>
                    <table class="failure-table">
                        <tr>
                            <th>Question</th>
                            <th>Failure Type</th>
                            <th>Confidence</th>
                            <th>Improvement Suggestion</th>
                        </tr>
            """
            
            for _, row in top_failures.iterrows():
                question = str(row.get('question', ''))[:100] + '...' if len(str(row.get('question', ''))) > 100 else str(row.get('question', ''))
                failure_type = row.get('primary_failure_mode', 'Unknown')
                confidence = f"{row.get('failure_confidence', 0):.2f}"
                suggestion = row.get('improvement_suggestions', 'No suggestion available')
                suggestion = suggestion[:150] + '...' if len(suggestion) > 150 else suggestion
                color = self.failure_colors.get(failure_type, '#888888')
                
                html_content += f"""
                    <tr>
                        <td>{question}</td>
                        <td><span class="failure-type" style="background-color: {color};">{failure_type.replace('_', ' ').title()}</span></td>
                        <td>{confidence}</td>
                        <td>{suggestion}</td>
                    </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        # Add conclusion and recommendations
        html_content += f"""
                <div class="section">
                    <h2>🎯 Recommendations</h2>
                    <div class="summary" style="background-color: #e8f4f8;">
                        <h3>Based on the analysis, consider:</h3>
                        <ul>
                            <li><strong>Address factual errors</strong>: Review responses with low accuracy scores</li>
                            <li><strong>Improve relevance</strong>: Ensure responses directly address the questions</li>
                            <li><strong>Enhance safety</strong>: Review flagged content for bias or harmful information</li>
                            <li><strong>Focus on weak categories</strong>: Target improvement on lowest-performing question types</li>
                            <li><strong>Regular monitoring</strong>: Use this framework for continuous evaluation</li>
                        </ul>
                    </div>
                </div>
                
                <div class="section" style="text-align: center; padding: 20px; border-top: 2px solid #ddd;">
                    <h3>Generated Files</h3>
                    <a href="score_distribution.png" class="btn">Score Distribution</a>
                    <a href="failure_breakdown.png" class="btn">Failure Breakdown</a>
                    <a href="category_performance.png" class="btn">Category Performance</a>
                    <a href="metric_correlations.png" class="btn">Metric Correlations</a>
                    <a href="top_failure_examples.png" class="btn">Top Failures</a>
                    <a href="llm_evaluation_dashboard.html" class="btn">Interactive Dashboard</a>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Interactive HTML report saved to: {save_path}")
        return save_path
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """
        Generate all visualizations and reports.
        
        Returns:
            Dictionary of generated file paths
        """
        print("Generating all visualizations...")
        
        # Load data
        eval_data = self.load_evaluation_data()
        failure_data = self.load_failure_data()
        
        # Generate all visualizations
        generated_files = {}
        
        # 1. Dashboard
        print("Creating interactive dashboard...")
        dashboard_file = os.path.join(self.output_dir, "llm_evaluation_dashboard.html")
        self.create_dashboard(eval_data, failure_data, dashboard_file)
        generated_files['dashboard'] = dashboard_file
        
        # 2. Individual visualizations
        print("Creating score distribution plot...")
        score_file = os.path.join(self.output_dir, "score_distribution.png")
        self.plot_score_distribution(eval_data, score_file)
        generated_files['score_distribution'] = score_file
        
        print("Creating failure breakdown plot...")
        failure_file = os.path.join(self.output_dir, "failure_breakdown.png")
        if 'primary_failure_mode' in failure_data.columns:
            self.plot_failure_breakdown(failure_data, failure_file)
            generated_files['failure_breakdown'] = failure_file
        
        print("Creating category performance plot...")
        category_file = os.path.join(self.output_dir, "category_performance.png")
        if 'category' in eval_data.columns:
            self.plot_category_performance(eval_data, category_file)
            generated_files['category_performance'] = category_file
        
        print("Creating metric correlations plot...")
        correlation_file = os.path.join(self.output_dir, "metric_correlations.png")
        self.plot_metric_correlations(eval_data, correlation_file)
        generated_files['metric_correlations'] = correlation_file
        
        print("Creating top failure examples plot...")
        if 'primary_failure_mode' in failure_data.columns:
            top_failures_file = os.path.join(self.output_dir, "top_failure_examples.png")
            self.plot_top_failure_examples(failure_data, eval_data, save_path=top_failures_file)
            generated_files['top_failure_examples'] = top_failures_file
        
        # 3. HTML report
        print("Generating comprehensive HTML report...")
        report_file = os.path.join(self.output_dir, "llm_evaluation_report.html")
        self.generate_interactive_report(eval_data, failure_data, report_file)
        generated_files['html_report'] = report_file
        
        print("\n✅ All visualizations generated successfully!")
        print("Generated files:")
        for name, path in generated_files.items():
            print(f"  - {name}: {os.path.basename(path)}")
        
        return generated_files


def create_dashboard(eval_file: str = None, failure_file: str = None) -> go.Figure:
    """
    Convenience function to create dashboard.
    
    Args:
        eval_file: Path to evaluation results file
        failure_file: Path to failure analysis file
        
    Returns:
        Plotly Figure object
    """
    visualizer = LLMVisualizer()
    return visualizer.create_dashboard(eval_file, failure_file)


def plot_score_distribution(eval_file: str = None) -> plt.Figure:
    """
    Convenience function to plot score distribution.
    
    Args:
        eval_file: Path to evaluation results file
        
    Returns:
        Matplotlib Figure object
    """
    visualizer = LLMVisualizer()
    return visualizer.plot_score_distribution(eval_file)


def plot_failure_breakdown(failure_file: str = None) -> plt.Figure:
    """
    Convenience function to plot failure breakdown.
    
    Args:
        failure_file: Path to failure analysis file
        
    Returns:
        Matplotlib Figure object
    """
    visualizer = LLMVisualizer()
    return visualizer.plot_failure_breakdown(failure_file)


def plot_category_performance(eval_file: str = None) -> plt.Figure:
    """
    Convenience function to plot category performance.
    
    Args:
        eval_file: Path to evaluation results file
        
    Returns:
        Matplotlib Figure object
    """
    visualizer = LLMVisualizer()
    return visualizer.plot_category_performance(eval_file)


def plot_metric_correlations(eval_file: str = None) -> plt.Figure:
    """
    Convenience function to plot metric correlations.
    
    Args:
        eval_file: Path to evaluation results file
        
    Returns:
        Matplotlib Figure object
    """
    visualizer = LLMVisualizer()
    return visualizer.plot_metric_correlations(eval_file)


def generate_interactive_report(eval_file: str = None, failure_file: str = None) -> str:
    """
    Convenience function to generate interactive report.
    
    Args:
        eval_file: Path to evaluation results file
        failure_file: Path to failure analysis file
        
    Returns:
        Path to generated HTML report
    """
    visualizer = LLMVisualizer()
    return visualizer.generate_interactive_report(eval_file, failure_file)


def generate_all_visualizations() -> Dict[str, str]:
    """
    Convenience function to generate all visualizations.
    
    Returns:
        Dictionary of generated file paths
    """
    visualizer = LLMVisualizer()
    return visualizer.generate_all_visualizations()


if __name__ == "__main__":
    """
    Main execution for testing the visualization module.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Evaluation Visualization Module')
    parser.add_argument('--action', type=str, default='all',
                       choices=['all', 'dashboard', 'scores', 'failures', 
                               'categories', 'correlations', 'report'],
                       help='Action to perform')
    parser.add_argument('--eval-file', type=str, default=None,
                       help='Set path to evaluation results file')
    parser.add_argument('--failure-file', type=str, default=None,
                       help='Set path to failure analysis file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Set output directory for visualizations')
    
    args = parser.parse_args()
    
    visualizer = LLMVisualizer(output_dir=args.output_dir)
    
    if args.action == 'all':
        visualizer.generate_all_visualizations()
    elif args.action == 'dashboard':
        visualizer.create_dashboard(args.eval_file, args.failure_file)
    elif args.action == 'scores':
        visualizer.plot_score_distribution(args.eval_file)
    elif args.action == 'failures':
        visualizer.plot_failure_breakdown(args.failure_file)
    elif args.action == 'categories':
        visualizer.plot_category_performance(args.eval_file)
    elif args.action == 'correlations':
        visualizer.plot_metric_correlations(args.eval_file)
    elif args.action == 'report':
        visualizer.generate_interactive_report(args.eval_file, args.failure_file)
    
    print("\n Visualization module ready to use!")
    print("To use in python code:")
    print("  from src.visualize import LLMVisualizer")
    print("  visualizer = LLMVisualizer()")
    print("  visualizer.generate_all_visualizations()")
