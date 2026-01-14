#!/usr/bin/env python3
"""
LLM Evaluation Framework - Complete Benchmark Pipeline

This script runs the entire evaluation pipeline:
1. Load test data
2. Run enhanced evaluation
3. Perform failure analysis
4. Generate visualizations and reports
5. Create comprehensive outputs

Usage:
    python run_benchmark.py [--clean] [--skip-eval] [--skip-analysis] [--skip-viz]
"""

import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from src.evaluate import evaluate_all_pairs_enhanced, EnhancedLLMEvaluator
    from src.analyze import analyze_failures_enhanced, EnhancedFailureAnalyzer
    from src.visualization import LLMVisualizer, generate_all_visualizations
    from src.utils import setup_directories, load_data, save_data
except ImportError:
    # If utils doesn't exist, create minimal version
    import warnings
    warnings.warn("src.utils not found, using minimal implementation")

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "benchmarks")

class LLMBenchmarkRunner:
    """
    Main benchmark runner for the LLM evaluation framework.
    
    Orchestrates the complete evaluation pipeline from data loading
    to report generation.
    """
    
    def __init__(self, clean: bool = False, verbose: bool = True):
        """
        Initialize the benchmark runner.
        
        Args:
            clean: Whether to clean output directory before running
            verbose: Whether to print detailed progress information
        """
        self.clean = clean
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.results = {}
        
        # Setup directories
        self.setup_directories()
        
        # Initialize components
        self.evaluator = None
        self.analyzer = None
        self.visualizer = None
        
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [DATA_DIR, OUTPUT_DIR, BENCHMARK_DIR]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            
        if self.clean and os.path.exists(OUTPUT_DIR):
            # Remove only certain files, not everything
            import glob
            files_to_remove = [
                "enhanced_evaluation_results.tsv",
                "enhanced_failure_analysis.tsv",
                "evaluation_report.md",
                "failure_analysis_report.md",
                "llm_evaluation_dashboard.html",
                "llm_evaluation_report.html",
                "score_distribution.png",
                "failure_breakdown.png",
                "category_performance.png",
                "metric_correlations.png",
                "top_failure_examples.png",
                "benchmark_summary.json"
            ]
            
            for file_pattern in files_to_remove:
                for file_path in glob.glob(os.path.join(OUTPUT_DIR, file_pattern)):
                    try:
                        os.remove(file_path)
                        if self.verbose:
                            print(f"Cleaned: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Warning: Could not remove {file_path}: {e}")
        
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load test questions and responses.
        
        Returns:
            Tuple of (questions_df, responses_df)
        """
        if self.verbose:
            print("\nLoading test data...")
        
        questions_path = os.path.join(DATA_DIR, "test_questions.tsv")
        responses_path = os.path.join(DATA_DIR, "test_responses.tsv")
        
        # Check if files exist
        if not os.path.exists(questions_path):
            raise FileNotFoundError(f"Questions file not found: {questions_path}")
        if not os.path.exists(responses_path):
            raise FileNotFoundError(f"Responses file not found: {responses_path}")
        
        # Load data
        questions_df = pd.read_csv(questions_path, sep='\t')
        responses_df = pd.read_csv(responses_path, sep='\t')
        
        if self.verbose:
            print(f"  ✓ Loaded {len(questions_df)} questions from {questions_path}")
            print(f"  ✓ Loaded {len(responses_df)} responses from {responses_path}")
            
            # Show data overview
            print(f"    Data overview:")
            print(f"    - Question categories: {questions_df['category'].unique().tolist()}")
            print(f"    - Sample questions:")
            for i, row in questions_df.head(3).iterrows():
                print(f"      {row['id']}. {row['question'][:50]}... ({row['category']})")
        
        return questions_df, responses_df
    
    def run_evaluation(self, questions_df: pd.DataFrame, responses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run enhanced evaluation on all question-response pairs.
        
        Args:
            questions_df: DataFrame containing test questions
            responses_df: DataFrame containing LLM responses
            
        Returns:
            DataFrame with evaluation results
        """
        if self.verbose:
            print("\n  Running enhanced evaluation...")
            print(f"  Evaluating {len(questions_df)} question-response pairs")
        
        # Run evaluation
        start_eval = time.time()
        evaluation_results = evaluate_all_pairs_enhanced(questions_df, responses_df)
        eval_time = time.time() - start_eval
        
        if self.verbose:
            print(f"  ✓ Evaluation completed in {eval_time:.2f} seconds")
            print(f"   Evaluation summary:")
            print(f"    - Average overall score: {evaluation_results['overall_score'].mean():.3f}")
            print(f"    - Score range: {evaluation_results['overall_score'].min():.3f} - {evaluation_results['overall_score'].max():.3f}")
            
            # Count passes/fails
            passes = sum(evaluation_results['primary_failure_mode'] == 'pass')
            fail_rate = (len(evaluation_results) - passes) / len(evaluation_results) * 100
            print(f"    - Passing responses: {passes}/{len(evaluation_results)} ({100-fail_rate:.1f}%)")
            print(f"    - Failure rate: {fail_rate:.1f}%")
        
        # Save results
        eval_output_path = os.path.join(OUTPUT_DIR, "enhanced_evaluation_results.tsv")
        evaluation_results.to_csv(eval_output_path, sep='\t', index=False)
        
        if self.verbose:
            print(f"  Saved evaluation results to: {eval_output_path}")
        
        # Generate evaluation report
        self.generate_evaluation_report(evaluation_results)
        
        # Store in results
        self.results['evaluation'] = {
            'execution_time': eval_time,
            'total_pairs': len(evaluation_results),
            'avg_score': float(evaluation_results['overall_score'].mean()),
            'pass_rate': float(passes / len(evaluation_results)),
            'output_file': eval_output_path
        }
        
        return evaluation_results
    
    def generate_evaluation_report(self, evaluation_results: pd.DataFrame):
        """Generate a markdown report from evaluation results."""
        if self.verbose:
            print("\n Generating evaluation report...")
        
        report_path = os.path.join(OUTPUT_DIR, "evaluation_report.md")
        
        # Calculate statistics
        total_responses = len(evaluation_results)
        passes = sum(evaluation_results['primary_failure_mode'] == 'pass')
        pass_rate = passes / total_responses * 100
        
        avg_scores = {
            'Overall': evaluation_results['overall_score'].mean(),
            'Accuracy': evaluation_results['composite_accuracy'].mean(),
            'Relevance': evaluation_results['composite_relevance'].mean(),
            'Safety': evaluation_results['composite_safety'].mean(),
            'Quality': evaluation_results['composite_quality'].mean()
        }
        
        # Failure distribution
        failure_counts = evaluation_results['primary_failure_mode'].value_counts().to_dict()
        
        # Generate markdown report
        report_content = f"""# LLM Evaluation Report

## Executive Summary
- **Total Responses Evaluated**: {total_responses}
- **Passing Responses**: {passes} ({pass_rate:.1f}%)
- **Failure Rate**: {100-pass_rate:.1f}%
- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Average Scores
| Metric | Score |
|--------|-------|
"""
        for metric, score in avg_scores.items():
            report_content += f"| {metric} | {score:.3f} |\n"
        
        report_content += """
## Failure Analysis
"""
        
        for failure_type, count in failure_counts.items():
            percentage = count / total_responses * 100
            report_content += f"- **{failure_type}**: {count} responses ({percentage:.1f}%)\n"
        
        # Performance by category
        if 'category' in evaluation_results.columns:
            report_content += "\n## Performance by Question Category\n\n"
            categories = evaluation_results['category'].unique()
            
            report_content += "| Category | Count | Avg Score | Pass Rate |\n"
            report_content += "|----------|-------|-----------|-----------|\n"
            
            for category in categories:
                cat_df = evaluation_results[evaluation_results['category'] == category]
                cat_count = len(cat_df)
                cat_avg_score = cat_df['overall_score'].mean()
                cat_passes = sum(cat_df['primary_failure_mode'] == 'pass')
                cat_pass_rate = cat_passes / cat_count * 100 if cat_count > 0 else 0
                
                report_content += f"| {category} | {cat_count} | {cat_avg_score:.3f} | {cat_pass_rate:.1f}% |\n"
        
        # Top and bottom performers
        report_content += "\n## Top Performers\n\n"
        top_3 = evaluation_results.nlargest(3, 'overall_score')
        for idx, row in top_3.iterrows():
            report_content += f"1. **ID {row['id']}** ({row['category']}): {row['overall_score']:.3f}\n"
            report_content += f"   - Question: {row['question'][:80]}...\n"
            report_content += f"   - Response: {row['response'][:80]}...\n"
            report_content += f"   - Failure Mode: {row['primary_failure_mode']}\n\n"
        
        report_content += "\n## Areas Needing Improvement\n\n"
        bottom_3 = evaluation_results.nsmallest(3, 'overall_score')
        for idx, row in bottom_3.iterrows():
            report_content += f"1. **ID {row['id']}** ({row['category']}): {row['overall_score']:.3f}\n"
            report_content += f"   - Question: {row['question'][:80]}...\n"
            report_content += f"   - Response: {row['response'][:80]}...\n"
            report_content += f"   - Failure Mode: {row['primary_failure_mode']}\n"
            report_content += f"   - Suggestions: {row['improvement_suggestions'][0] if isinstance(row['improvement_suggestions'], list) and row['improvement_suggestions'] else 'No specific suggestions'}\n\n"
        
        report_content += f"""
---
*Report generated by LLM Evaluation Framework v1.0*
*Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        if self.verbose:
            print(f"  ✓ Saved evaluation report to: {report_path}")
        
        self.results['evaluation_report'] = report_path
    
    def run_failure_analysis(self, evaluation_results_path: str = None) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Run enhanced failure analysis on evaluation results.
        
        Args:
            evaluation_results_path: Path to evaluation results file
            
        Returns:
            Tuple of (analyzed_df, summary_dict, report_dict)
        """
        if self.verbose:
            print("\n Running failure analysis...")
        
        if evaluation_results_path is None:
            evaluation_results_path = os.path.join(OUTPUT_DIR, "enhanced_evaluation_results.tsv")
        
        if not os.path.exists(evaluation_results_path):
            raise FileNotFoundError(f"Evaluation results not found: {evaluation_results_path}")
        
        # Run analysis
        start_analysis = time.time()
        analyzed_df, summary, report = analyze_failures_enhanced(evaluation_results_path)
        analysis_time = time.time() - start_analysis
        
        if self.verbose:
            print(f"  ✓ Failure analysis completed in {analysis_time:.2f} seconds")
            print(f"   Analysis summary:")
            print(f"    - Total responses analyzed: {summary.get('total_responses', 0)}")
            print(f"    - Failed responses: {summary.get('failed_responses', 0)}")
            print(f"    - Success rate: {summary.get('success_rate', 0):.1f}%")
            
            if 'category_breakdown' in summary:
                print(f"    - Failure categories:")
                for category, count in summary['category_breakdown'].items():
                    percentage = count / summary['total_responses'] * 100
                    print(f"      - {category}: {count} ({percentage:.1f}%)")
        
        # Save analyzed results
        analysis_output_path = os.path.join(OUTPUT_DIR, "enhanced_failure_analysis.tsv")
        analyzed_df.to_csv(analysis_output_path, sep='\t', index=False)
        
        if self.verbose:
            print(f"   Saved failure analysis to: {analysis_output_path}")
        
        # Save summary and report
        summary_path = os.path.join(OUTPUT_DIR, "enhanced_failure_analysis_summary.json")
        report_path = os.path.join(OUTPUT_DIR, "enhanced_failure_analysis_report.json")
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.verbose:
            print(f"   Saved summary to: {summary_path}")
            print(f"   Saved detailed report to: {report_path}")
        
        # Generate failure analysis report
        self.generate_failure_analysis_report(analyzed_df, summary, report)
        
        # Store in results
        self.results['failure_analysis'] = {
            'execution_time': analysis_time,
            'total_analyzed': summary.get('total_responses', 0),
            'failed_responses': summary.get('failed_responses', 0),
            'success_rate': summary.get('success_rate', 0),
            'output_file': analysis_output_path,
            'summary_file': summary_path,
            'report_file': report_path
        }
        
        return analyzed_df, summary, report
    
    def generate_failure_analysis_report(self, analyzed_df: pd.DataFrame, summary: Dict, report: Dict):
        """Generate a comprehensive failure analysis report."""
        if self.verbose:
            print("\n Generating failure analysis report...")
        
        report_path = os.path.join(OUTPUT_DIR, "failure_analysis_report.md")
        
        report_content = f"""# Enhanced Failure Analysis Report

## Executive Summary

**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Responses Analyzed**: {summary.get('total_responses', 0)}
**Failed Responses**: {summary.get('failed_responses', 0)}
**Success Rate**: {summary.get('success_rate', 0):.1f}%
**Average Overall Score**: {summary.get('overall_score_mean', 0):.3f}

## Detailed Failure Breakdown

### Failure Categories
"""
        
        # Failure categories
        if 'category_breakdown' in summary:
            for category, count in sorted(summary['category_breakdown'].items(), key=lambda x: x[1], reverse=True):
                percentage = count / summary['total_responses'] * 100
                report_content += f"- **{category}**: {count} responses ({percentage:.1f}%)\n"
        
        # Performance by question category
        report_content += "\n### Performance by Question Category\n\n"
        if 'category' in analyzed_df.columns:
            categories = analyzed_df['category'].unique()
            
            report_content += "| Category | Count | Avg Score | Failure Rate | Top Failure |\n"
            report_content += "|----------|-------|-----------|--------------|-------------|\n"
            
            for category in categories:
                cat_df = analyzed_df[analyzed_df['category'] == category]
                count = len(cat_df)
                avg_score = cat_df['overall_score'].mean()
                failures = cat_df[cat_df['failure_primary_category'] != 'no_failure']
                failure_rate = len(failures) / count * 100 if count > 0 else 0
                
                # Most common failure
                if len(failures) > 0:
                    top_failure = failures['failure_primary_category'].mode()
                    top_failure = top_failure[0] if len(top_failure) > 0 else 'None'
                else:
                    top_failure = 'None'
                
                report_content += f"| {category} | {count} | {avg_score:.3f} | {failure_rate:.1f}% | {top_failure} |\n"
        
        # Recommendations
        if 'recommendations' in report:
            report_content += "\n## Recommendations\n\n"
            
            # Group by priority
            for priority in ['high', 'medium', 'low']:
                priority_recs = [r for r in report['recommendations'] if r['priority'] == priority]
                if priority_recs:
                    report_content += f"### {priority.title()} Priority Recommendations\n\n"
                    for rec in priority_recs:
                        report_content += f"#### {rec['area']}\n"
                        report_content += f"{rec['recommendation']}\n\n"
                        report_content += f"*Evidence*: {rec['evidence']}\n\n"
        
        # Common improvement suggestions
        if 'top_suggested_fixes' in summary:
            report_content += "## Most Common Improvement Suggestions\n\n"
            fixes = summary['top_suggested_fixes']
            
            for fix, count in sorted(fixes.items(), key=lambda x: x[1], reverse=True)[:10]:
                if fix != 'None required' and fix != 'No improvement needed':
                    percentage = count / summary['total_responses'] * 100
                    report_content += f"- **{fix}**: {count} occurrences ({percentage:.1f}%)\n"
        
        # Case studies
        report_content += "\n## Case Studies\n\n"
        
        # Get examples of different failure types
        failure_categories = [cat for cat in analyzed_df['failure_primary_category'].unique() 
                            if cat != 'no_failure' and cat != 'pass']
        
        for category in failure_categories[:3]:  # Show top 3 failure types
            cat_df = analyzed_df[analyzed_df['failure_primary_category'] == category]
            if len(cat_df) > 0:
                example = cat_df.iloc[0]
                
                report_content += f"### {category.replace('_', ' ').title()}\n\n"
                report_content += f"**ID**: {example['id']}\n"
                report_content += f"**Category**: {example['category']}\n"
                report_content += f"**Overall Score**: {example['overall_score']:.3f}\n\n"
                report_content += f"**Question**: {example['question']}\n\n"
                report_content += f"**Response**: {example['response']}\n\n"
                report_content += f"**Failure Analysis**:\n"
                report_content += f"- Primary Category: {example['failure_primary_category']}\n"
                report_content += f"- Sub Category: {example.get('failure_sub_category', 'N/A')}\n"
                report_content += f"- Confidence: {example.get('failure_confidence', 'N/A')}\n"
                
                if 'improvement_suggestions' in example:
                    suggestions = example['improvement_suggestions']
                    if isinstance(suggestions, list) and suggestions:
                        report_content += f"- **Suggestion**: {suggestions[0]}\n"
                
                report_content += "\n---\n\n"
        
        report_content += f"""
---
*Report generated by Enhanced Failure Analysis Module*
*Framework Version: 1.0*
"""
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        if self.verbose:
            print(f"  ✓ Saved failure analysis report to: {report_path}")
        
        self.results['failure_analysis_report'] = report_path
    
    def run_visualizations(self, analyzed_df: pd.DataFrame = None, summary: Dict = None):
        """
        Generate all visualizations and interactive reports.
        
        Args:
            analyzed_df: DataFrame from failure analysis
            summary: Summary dictionary from failure analysis
        """
        if self.verbose:
            print("\n Generating visualizations...")
        
        # Initialize visualizer
        self.visualizer = LLMVisualizer(output_dir=OUTPUT_DIR)
        
        start_viz = time.time()
        
        # Generate all visualizations
        generated_files = self.visualizer.generate_all_visualizations()
        viz_time = time.time() - start_viz
        
        if self.verbose:
            print(f"  ✓ Visualizations completed in {viz_time:.2f} seconds")
            print(f"   Generated files:")
            for name, path in generated_files.items():
                print(f"    - {name}: {os.path.basename(path)}")
        
        # Store in results
        self.results['visualizations'] = {
            'execution_time': viz_time,
            'generated_files': generated_files
        }
        
        return generated_files
    
    def generate_benchmark_summary(self):
        """Generate a comprehensive benchmark summary."""
        if self.verbose:
            print("\n Generating benchmark summary...")
        
        summary_path = os.path.join(OUTPUT_DIR, "benchmark_summary.json")
        
        # Calculate total execution time
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        # Create comprehensive summary
        benchmark_summary = {
            'benchmark_info': {
                'run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'total_execution_time': total_time,
                'framework_version': '1.0',
                'python_version': sys.version
            },
            'results': self.results,
            'files_generated': self._list_generated_files(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(benchmark_summary, f, indent=2)
        
        if self.verbose:
            print(f"  ✓ Saved benchmark summary to: {summary_path}")
        
        # Print quick summary
        self._print_quick_summary(benchmark_summary)
        
        return benchmark_summary
    
    def _list_generated_files(self) -> List[str]:
        """List all files generated by the benchmark."""
        generated_files = []
        output_extensions = ['.tsv', '.md', '.html', '.png', '.json']
        
        for filename in os.listdir(OUTPUT_DIR):
            if any(filename.endswith(ext) for ext in output_extensions):
                filepath = os.path.join(OUTPUT_DIR, filename)
                if os.path.isfile(filepath):
                    generated_files.append(filename)
        
        return sorted(generated_files)
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        # Check evaluation results
        if 'evaluation' in self.results:
            eval_data = self.results['evaluation']
            
            if eval_data['pass_rate'] < 0.5:
                recommendations.append({
                    'priority': 'high',
                    'area': 'Overall Performance',
                    'recommendation': 'Model performance is below 50% pass rate. Consider retraining or fine-tuning the model.',
                    'evidence': f'Pass rate: {eval_data["pass_rate"]:.1%}'
                })
            elif eval_data['pass_rate'] < 0.7:
                recommendations.append({
                    'priority': 'medium',
                    'area': 'Overall Performance',
                    'recommendation': 'Model performance has room for improvement. Focus on the most common failure types.',
                    'evidence': f'Pass rate: {eval_data["pass_rate"]:.1%}'
                })
        
        # Check failure analysis
        if 'failure_analysis' in self.results:
            failure_data = self.results['failure_analysis']
            
            if failure_data.get('success_rate', 100) < 50:
                recommendations.append({
                    'priority': 'high',
                    'area': 'Failure Analysis',
                    'recommendation': 'High failure rate detected. Review the failure analysis report for specific issues.',
                    'evidence': f'Success rate: {failure_data.get("success_rate", 0):.1f}%'
                })
        
        return recommendations
    
    def _print_quick_summary(self, benchmark_summary: Dict):
        """Print a quick summary to console."""
        print("\n" + "="*70)
        print(" BENCHMARK COMPLETE - QUICK SUMMARY")
        print("="*70)
        
        # Timing information
        total_time = benchmark_summary['benchmark_info']['total_execution_time']
        print(f"\n  Timing:")
        print(f"  Total execution time: {total_time:.2f} seconds")
        
        if 'evaluation' in self.results:
            eval_time = self.results['evaluation']['execution_time']
            print(f"  Evaluation: {eval_time:.2f} seconds")
        
        if 'failure_analysis' in self.results:
            analysis_time = self.results['failure_analysis']['execution_time']
            print(f"  Failure analysis: {analysis_time:.2f} seconds")
        
        if 'visualizations' in self.results:
            viz_time = self.results['visualizations']['execution_time']
            print(f"  Visualizations: {viz_time:.2f} seconds")
        
        # Performance metrics
        print(f"\n Performance Metrics:")
        
        if 'evaluation' in self.results:
            eval_data = self.results['evaluation']
            print(f"  Overall Score: {eval_data['avg_score']:.3f}")
            print(f"  Pass Rate: {eval_data['pass_rate']:.1%}")
            print(f"  Evaluated Pairs: {eval_data['total_pairs']}")
        
        if 'failure_analysis' in self.results:
            failure_data = self.results['failure_analysis']
            print(f"  Success Rate: {failure_data.get('success_rate', 0):.1f}%")
            print(f"  Failed Responses: {failure_data.get('failed_responses', 0)}")
        
        # Generated files
        print(f"\n Generated Files ({len(benchmark_summary['files_generated'])}):")
        for i, filename in enumerate(benchmark_summary['files_generated'][:5]):  # Show first 5
            print(f"  {i+1}. {filename}")
        
        if len(benchmark_summary['files_generated']) > 5:
            print(f"  ... and {len(benchmark_summary['files_generated']) - 5} more")
        
        # Recommendations
        if benchmark_summary['recommendations']:
            print(f"\n Key Recommendations:")
            for rec in benchmark_summary['recommendations']:
                print(f"  [{rec['priority'].upper()}] {rec['area']}: {rec['recommendation']}")
        
        print(f"\n All outputs available in: {OUTPUT_DIR}")
        print("="*70 + "\n")
    
    def run_complete_pipeline(self, 
                             skip_eval: bool = False,
                             skip_analysis: bool = False,
                             skip_viz: bool = False):
        """
        Run the complete benchmark pipeline.
        
        Args:
            skip_eval: Skip evaluation step
            skip_analysis: Skip failure analysis step
            skip_viz: Skip visualization step
        """
        # Start timing
        self.start_time = datetime.now()
        
        print(" Starting LLM Evaluation Framework Benchmark")
        print("="*70)
        
        try:
            # 1. Load test data
            questions_df, responses_df = self.load_test_data()
            
            # 2. Run evaluation (unless skipped)
            evaluation_results = None
            if not skip_eval:
                evaluation_results = self.run_evaluation(questions_df, responses_df)
            else:
                if self.verbose:
                    print("\n Skipping evaluation step...")
            
            # 3. Run failure analysis (unless skipped)
            analyzed_df = None
            summary = None
            report = None
            
            if not skip_analysis:
                if skip_eval:
                    # Load existing evaluation results
                    eval_path = os.path.join(OUTPUT_DIR, "enhanced_evaluation_results.tsv")
                    if os.path.exists(eval_path):
                        if self.verbose:
                            print("\n Loading existing evaluation results...")
                        evaluation_results = pd.read_csv(eval_path, sep='\t')
                    else:
                        raise FileNotFoundError("Evaluation results not found. Run evaluation first.")
                
                analyzed_df, summary, report = self.run_failure_analysis()
            else:
                if self.verbose:
                    print("\n Skipping failure analysis step...")
            
            # 4. Generate visualizations (unless skipped)
            if not skip_viz:
                if skip_analysis:
                    # Load existing analysis results
                    analysis_path = os.path.join(OUTPUT_DIR, "enhanced_failure_analysis.tsv")
                    if os.path.exists(analysis_path):
                        if self.verbose:
                            print("\n Loading existing analysis results...")
                        analyzed_df = pd.read_csv(analysis_path, sep='\t')
                
                self.run_visualizations(analyzed_df, summary)
            else:
                if self.verbose:
                    print("\n Skipping visualization step...")
            
            # 5. Generate final summary
            self.end_time = datetime.now()
            benchmark_summary = self.generate_benchmark_summary()
            
            return benchmark_summary
            
        except Exception as e:
            print(f"\n❌ Benchmark failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point for the benchmark script."""
    # Create argument parser object
    parser = argparse.ArgumentParser(
        description='LLM Evaluation Framework - Complete Benchmark Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run complete pipeline
  %(prog)s --clean            # Clean outputs and run complete pipeline
  %(prog)s --skip-viz         # Run evaluation and analysis only
  %(prog)s --skip-eval --skip-analysis  # Generate visualizations only
        """
    )
    
    # Adding command line options
    parser.add_argument('--clean', action='store_true',
                       help='Clean output directory before running')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation step (use existing results)')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip failure analysis step')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization step')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce verbosity')
    
    # Reading and validating command-line arguments and storing them in agrs object
    args = parser.parse_args()
    
    # Initialize and run benchmark
    runner = LLMBenchmarkRunner(
        clean=args.clean,
        verbose=not args.quiet
    )
    
    # Run complete pipeline
    benchmark_summary = runner.run_complete_pipeline(
        skip_eval=args.skip_eval,
        skip_analysis=args.skip_analysis,
        skip_viz=args.skip_viz
    )
    
    if benchmark_summary:
        print(" Benchmark completed successfully!")
        return 0
    else:
        print(" Benchmark failed!")
        return 1


if __name__ == "__main__":
    # Add some ASCII art for fun
    ascii_art = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     LLM Evaluation Framework - Benchmark Pipeline         ║
    ║                                                           ║
    ║     Evaluate | Analyze | Visualize | Report               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    
    print(ascii_art)
    
    # Run main function
    exit_code = main()
    sys.exit(exit_code)
