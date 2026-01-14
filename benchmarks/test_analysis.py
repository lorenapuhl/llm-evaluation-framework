"""
Test script for enhanced failure analysis module
Tests the failure analyzer with enhanced evaluation results
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json
from src.analyze import (
    EnhancedFailureAnalyzer,
    analyze_failures_enhanced,
)

def test_basic_analysis():
    """Test basic analysis functionality."""
    print("=" * 60)
    print("Testing Basic Analysis Functionality")
    print("=" * 60)
    
    # Create test data with enhanced evaluation format
    test_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'category': ['Factual', 'Explanatory', 'Creative', 'Instruction', 'Sensitive'],
        'question': [
            'What is the capital of France?',
            'Explain photosynthesis',
            'Write a poem about the ocean',
            'How to bake cookies?',
            'Discuss AI ethics'
        ],
        'response': [
            'Paris is the capital of France',
            'Photosynthesis converts sunlight to energy',
            'Waves crash upon the shore',
            'Preheat oven to 350°F, mix ingredients',
            'AI raises ethical questions about bias'
        ],
        'overall_score': [0.85, 0.65, 0.45, 0.75, 0.55],
        'composite_accuracy': [0.9, 0.7, 0.3, 0.8, 0.6],
        'composite_relevance': [0.8, 0.6, 0.4, 0.7, 0.5],
        'composite_safety': [0.95, 0.9, 0.85, 0.9, 0.7],
        'composite_quality': [0.8, 0.7, 0.6, 0.7, 0.6],
        'safety_has_bias_risk': [False, False, False, False, True],
        'safety_bias_categories': ['', '', '', '', "['unbalanced_perspective']"],
        'is_refusal': [False, False, False, False, False],
        'primary_failure_mode': ['pass', 'partial_relevance', 'irrelevant_response', 'pass', 'safety_issue'],
        'improvement_suggestions': [
            '["None required"]',
            '["Make instructions more specific"]',
            '["Better address the specific question intent"]',
            '["None required"]',
            '["Present more balanced perspectives on sensitive topics"]'
        ],
        'passed_accuracy': [True, True, False, True, True],
        'passed_relevance': [True, False, False, True, False],
        'passed_safety': [True, True, True, True, False],
        'passed_quality': [True, True, True, True, True]
    })
    
    print(f"Created test data with {len(test_data)} examples")
    print(f"Columns: {list(test_data.columns)}")
    
    # Initialize analyzer
    analyzer = EnhancedFailureAnalyzer()
    
    # Test single row analysis
    print("\n1. Single Row Analysis:")
    test_row = test_data.iloc[2]  # Creative response with low scores
    analysis = analyzer.categorize_failure(test_row)
    
    print(f"  Question: {test_row['question']}")
    print(f"  Overall Score: {test_row['overall_score']:.3f}")
    print(f"  Primary Category: {analysis['primary_category']}")
    print(f"  Sub Category: {analysis['sub_category']}")
    print(f"  Confidence: {analysis['confidence']:.2f}")
    print(f"  Reasons: {analysis['reasons']}")
    print(f"  Suggested Fixes: {analysis['suggested_fixes'][:2]}")
    
    # Test dataset analysis
    print("\n2. Dataset Analysis:")
    analyzed_df = analyzer.analyze_dataset(test_data)
    
    print(f"  Added columns: {[c for c in analyzed_df.columns if c.startswith('failure_')]}")
    print(f"  Sample categorizations:")
    for i, row in analyzed_df.iterrows():
        print(f"    ID {row['id']}: {row['failure_primary_category']} -> {row['failure_sub_category']}")
    
    # Test summary statistics
    print("\n3. Summary Statistics:")
    summary = analyzer.generate_summary_statistics(analyzed_df)
    
    print(f"  Total responses: {summary['total_responses']}")
    print(f"  Failed responses: {summary['failed_responses']}")
    print(f"  Success rate: {summary['success_rate']:.1f}%")
    print(f"  Overall score mean: {summary['overall_score_mean']:.3f}")
    
    print(f"\n  Category breakdown:")
    for category, count in summary['category_breakdown'].items():
        print(f"    {category}: {count}")
    
    if 'top_suggested_fixes' in summary:
        print(f"\n  Top suggested fixes:")
        for fix, count in summary['top_suggested_fixes'].items():
            if fix != 'None required':
                print(f"    {fix}: {count}")
    
    # Test examples by failure type
    print("\n4. Failure Examples:")
    examples = analyzer.get_examples_by_failure_type(analyzed_df, 'irrelevant', 2)
    for ex in examples:
        print(f"  ID {ex['id']} ({ex['category']}):")
        print(f"    Question: {ex['question']}")
        print(f"    Failure: {ex['primary_category']} -> {ex['sub_category']}")
        print(f"    Score: {ex['overall_score']:.3f}")
        print(f"    Reason: {ex['reasons'][0] if ex['reasons'] else 'N/A'}")
    
    # Test detailed report
    print("\n5. Detailed Report:")
    report = analyzer.generate_detailed_report(analyzed_df)
    
    print(f"  Report generated with {len(report.get('failure_analysis', {}))} failure categories analyzed")
    
    if 'recommendations' in report:
        print(f"  Recommendations ({len(report['recommendations'])}):")
        for rec in report['recommendations'][:2]:  # Show first 2
            print(f"    [{rec['priority'].upper()}] {rec['area']}: {rec['recommendation']}")

def test_with_real_enhanced_results():
    """Test with actual enhanced evaluation results."""
    print("\n" + "=" * 60)
    print("Testing with Real Enhanced Evaluation Results")
    print("=" * 60)
    
    # Path to enhanced evaluation results
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "enhanced_evaluation_results.tsv"
    )
    
    if not os.path.exists(results_path):
        print(f"❌ Enhanced evaluation results not found: {results_path}")
        print("Please run enhanced evaluation first:")
        print("  python benchmarks/test_evaluation.py")
        print("  OR")
        print("  python -c \"from src.evaluate import evaluate_all_pairs_enhanced; import pandas as pd; q=pd.read_csv('data/test_questions.tsv', sep='\\t'); r=pd.read_csv('data/test_responses.tsv', sep='\\t'); results=evaluate_all_pairs_enhanced(q, r); results.to_csv('outputs/enhanced_evaluation_results.tsv', sep='\\t', index=False)\"")
        return None, None, None
    
    # Load enhanced results
    print(f"Loading enhanced evaluation results from: {results_path}")
    enhanced_results = pd.read_csv(results_path, sep='\t')
    
    print(f"✓ Loaded {len(enhanced_results)} enhanced evaluation results")
    print(f"Columns: {len(enhanced_results.columns)} columns")
    print(f"Sample columns: {list(enhanced_results.columns[:10])}...")
    
    # Show basic info
    print(f"\nDataset Overview:")
    print(f"  IDs: {len(enhanced_results['id'].unique())}")
    print(f"  Categories: {enhanced_results['category'].unique().tolist()}")
    print(f"  Overall score range: {enhanced_results['overall_score'].min():.3f} - {enhanced_results['overall_score'].max():.3f}")
    print(f"  Average overall score: {enhanced_results['overall_score'].mean():.3f}")
    
    # Check for required columns
    required_columns = ['composite_accuracy', 'composite_relevance', 'composite_safety', 'composite_quality']
    missing_columns = [col for col in required_columns if col not in enhanced_results.columns]
    
    if missing_columns:
        print(f"\n⚠ Missing required columns: {missing_columns}")
        print("The results file may not be in the enhanced format.")
        print("Checking for alternative column names...")
        
        # Try to find alternative column names
        column_mapping = {}
        for expected in required_columns:
            # Look for columns that contain the expected name
            matching = [col for col in enhanced_results.columns if expected in col]
            if matching:
                column_mapping[expected] = matching[0]
                print(f"  Found {expected} as: {matching[0]}")
        
        if len(column_mapping) < len(required_columns):
            print("❌ Could not find all required columns. Please run enhanced evaluation first.")
            return None, None, None
    
    # Run enhanced failure analysis
    print("\n" + "=" * 60)
    print("Running Enhanced Failure Analysis")
    print("=" * 60)
    
    analyzed_df, summary, report = analyze_failures_enhanced(results_path)
    
    print(f"✓ Analysis complete")
    print(f"  Analyzed {len(analyzed_df)} responses")
    print(f"  Added {len([c for c in analyzed_df.columns if c.startswith('failure_')])} failure analysis columns")
    
    # Save analyzed results
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    analyzed_path = os.path.join(outputs_dir, "enhanced_failure_analysis.tsv")
    analyzed_df.to_csv(analyzed_path, sep='\t', index=False)
    
    print(f"✓ Saved analyzed results to: {analyzed_path}")
    
    return analyzed_df, summary, report

def analyze_failure_types(analyzed_df: pd.DataFrame, summary: Dict):
    """Analyze different failure types in detail."""
    print("\n" + "=" * 60)
    print("Failure Type Analysis")
    print("=" * 60)
    
    # Get analyzer for examples
    analyzer = EnhancedFailureAnalyzer()
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total responses: {summary['total_responses']}")
    print(f"  Failed responses: {summary['failed_responses']}")
    print(f"  Success rate: {summary['success_rate']:.1f}%")
    print(f"  Average overall score: {summary['overall_score_mean']:.3f} ± {summary['overall_score_std']:.3f}")
    
    # Failure category breakdown
    print(f"\nFailure Category Breakdown:")
    category_breakdown = summary.get('category_breakdown', {})
    for category, count in sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True):
        percentage = count / summary['total_responses'] * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Subcategory breakdown
    if 'subcategory_breakdown' in summary:
        print(f"\nTop Subcategories:")
        subcategories = summary['subcategory_breakdown']
        for subcat, count in sorted(subcategories.items(), key=lambda x: x[1], reverse=True)[:10]:
            if subcat and str(subcat) != 'nan':
                percentage = count / summary['failed_responses'] * 100
                print(f"  {subcat}: {count} ({percentage:.1f}% of failures)")
    
    # Analyze each failure category
    failure_categories = [cat for cat in analyzed_df['failure_primary_category'].unique() 
                         if cat != 'no_failure']
    
    for category in failure_categories:
        print(f"\n{'='*40}")
        print(f"Analysis of: {category}")
        print(f"{'='*40}")
        
        cat_df = analyzed_df[analyzed_df['failure_primary_category'] == category]
        
        print(f"Count: {len(cat_df)} responses")
        
        # Get examples
        examples = analyzer.get_examples_by_failure_type(analyzed_df, category, 2)
        
        for i, ex in enumerate(examples, 1):
            print(f"\nExample {i} (ID {ex['id']} - {ex['category']}):")
            print(f"  Question: {ex['question'][:80]}...")
            print(f"  Response: {ex['llm_answer'][:80]}...")
            print(f"  Overall Score: {ex['overall_score']:.3f}")
            print(f"  Subcategory: {ex['sub_category']}")
            print(f"  Confidence: {ex['confidence']:.2f}")
            print(f"  Reasons: {ex['reasons'][0] if ex['reasons'] else 'N/A'}")
            print(f"  Fixes: {ex['suggested_fixes'][0] if ex['suggested_fixes'] else 'N/A'}")
            
            # Show metrics
            print(f"  Metrics - Accuracy: {ex['composite_accuracy']:.3f}, "
                  f"Relevance: {ex['composite_relevance']:.3f}, "
                  f"Safety: {ex['composite_safety']:.3f}, "
                  f"Quality: {ex['composite_quality']:.3f}")

def analyze_performance_by_category(analyzed_df: pd.DataFrame):
    """Analyze performance by question category."""
    print("\n" + "=" * 60)
    print("Performance by Question Category")
    print("=" * 60)
    
    if 'category' not in analyzed_df.columns:
        print("No category information available")
        return
    
    categories = analyzed_df['category'].unique()
    
    print(f"\n{'Category':<15} {'Count':<6} {'Avg Score':<10} {'Fail Rate':<10} {'Top Failure'}")
    print("-" * 60)
    
    for category in sorted(categories):
        cat_df = analyzed_df[analyzed_df['category'] == category]
        
        count = len(cat_df)
        avg_score = cat_df['overall_score'].mean()
        fail_rate = len(cat_df[cat_df['failure_primary_category'] != 'no_failure']) / count * 100
        
        # Find most common failure type
        failures = cat_df[cat_df['failure_primary_category'] != 'no_failure']
        if len(failures) > 0:
            top_failure = failures['failure_primary_category'].mode()
            top_failure = top_failure[0] if len(top_failure) > 0 else 'N/A'
        else:
            top_failure = 'None'
        
        print(f"{category:<15} {count:<6} {avg_score:<10.3f} {fail_rate:<10.1f}% {top_failure}")
    
    # Show category-specific insights
    print(f"\nCategory Insights:")
    for category in categories:
        cat_df = analyzed_df[analyzed_df['category'] == category]
        failures = cat_df[cat_df['failure_primary_category'] != 'no_failure']
        
        if len(failures) > 0:
            # Get most common subcategory
            common_sub = failures['failure_sub_category'].mode()
            common_sub = common_sub[0] if len(common_sub) > 0 else 'N/A'
            
            print(f"  {category}: {len(failures)} failures, most common: {common_sub}")

def generate_comprehensive_report(analyzed_df: pd.DataFrame, summary: Dict, report: Dict):
    """Generate a comprehensive markdown report."""
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    report_path = os.path.join(outputs_dir, "enhanced_failure_analysis_report.md")
    
    report_content = """# Enhanced Failure Analysis Report

## Executive Summary

"""
    
    # Summary statistics
    report_content += f"**Total Responses Analyzed**: {summary['total_responses']}\n"
    report_content += f"**Failed Responses**: {summary['failed_responses']}\n"
    report_content += f"**Success Rate**: {summary['success_rate']:.1f}%\n"
    report_content += f"**Average Overall Score**: {summary['overall_score_mean']:.3f} ± {summary['overall_score_std']:.3f}\n\n"
    
    # Failure category breakdown
    report_content += "## Failure Category Breakdown\n\n"
    category_breakdown = summary.get('category_breakdown', {})
    
    for category, count in sorted(category_breakdown.items(), key=lambda x: x[1], reverse=True):
        percentage = count / summary['total_responses'] * 100
        report_content += f"- **{category}**: {count} responses ({percentage:.1f}%)\n"
    
    # Performance by category
    if 'category_performance' in summary:
        report_content += "\n## Performance by Question Category\n\n"
        report_content += "| Category | Count | Avg Score | Failure Rate |\n"
        report_content += "|----------|-------|-----------|--------------|\n"
        
        for category, perf in summary['category_performance'].items():
            report_content += f"| {category} | {perf['count']} | {perf['avg_score']:.3f} | {perf['failure_rate']:.1f}% |\n"
    
    # Detailed failure analysis
    report_content += "\n## Detailed Failure Analysis\n\n"
    
    analyzer = EnhancedFailureAnalyzer()
    failure_categories = [cat for cat in analyzed_df['failure_primary_category'].unique() 
                         if cat != 'no_failure']
    
    for category in failure_categories:
        report_content += f"### {category.replace('_', ' ').title()}\n\n"
        
        cat_df = analyzed_df[analyzed_df['failure_primary_category'] == category]
        
        # Get examples
        examples = analyzer.get_examples_by_failure_type(analyzed_df, category, 2)
        
        for i, ex in enumerate(examples, 1):
            report_content += f"**Example {i} (ID {ex['id']})**\n"
            report_content += f"- **Category**: {ex['category']}\n"
            report_content += f"- **Question**: {ex['question']}\n"
            report_content += f"- **Response**: {ex['llm_answer']}\n"
            report_content += f"- **Overall Score**: {ex['overall_score']:.3f}\n"
            report_content += f"- **Subcategory**: {ex['sub_category']}\n"
            report_content += f"- **Reason**: {ex['reasons'][0] if ex['reasons'] else 'N/A'}\n"
            report_content += f"- **Suggested Fix**: {ex['suggested_fixes'][0] if ex['suggested_fixes'] else 'N/A'}\n\n"
    
    # Recommendations
    if 'recommendations' in report:
        report_content += "## Recommendations\n\n"
        
        # Group by priority
        for priority in ['high', 'medium']:
            priority_recs = [r for r in report['recommendations'] if r['priority'] == priority]
            if priority_recs:
                report_content += f"### {priority.title()} Priority\n\n"
                for rec in priority_recs:
                    report_content += f"1. **{rec['area']}**: {rec['recommendation']}  \n"
                    report_content += f"   *Evidence*: {rec['evidence']}\n\n"
    
    # Common fixes
    if 'top_suggested_fixes' in summary:
        report_content += "## Most Common Suggested Fixes\n\n"
        fixes = summary['top_suggested_fixes']
        for fix, count in sorted(fixes.items(), key=lambda x: x[1], reverse=True)[:5]:
            if fix != 'None required':
                report_content += f"- {fix} ({count} occurrences)\n"
    
    report_content += "\n---\n"
    report_content += "*Report generated by Enhanced Failure Analysis Module*\n"
    
    # Save report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"✓ Generated comprehensive report: {report_path}")
    return report_path

def main():
    """Main test function."""
    print("Enhanced Failure Analysis - Test Suite")
    print("=" * 60)
    
    try:
        # Run tests
        test_basic_analysis()
        
        # Test with real data
        analyzed_df, summary, report = test_with_real_enhanced_results()
        
        if analyzed_df is not None:
            # Perform detailed analysis
            analyze_failure_types(analyzed_df, summary)
            analyze_performance_by_category(analyzed_df)
            
            # Generate reports
            report_path = generate_comprehensive_report(analyzed_df, summary, report)
            
            # Show key findings
            print("\n" + "=" * 60)
            print("✅ Analysis Complete - Key Findings")
            print("=" * 60)
            
            if summary:
                success_rate = summary.get('success_rate', 0)
                print(f"\nOverall Success Rate: {success_rate:.1f}%")
                
                if success_rate < 50:
                    print("⚠ Warning: Low success rate indicates significant issues")
                elif success_rate < 70:
                    print("⚠ Note: Moderate success rate, room for improvement")
                else:
                    print("✓ Good: High success rate achieved")
            
            print(f"\nDetailed reports saved to:")
            print(f"  - outputs/enhanced_failure_analysis.tsv")
            print(f"  - outputs/enhanced_failure_analysis_summary.json")
            print(f"  - outputs/enhanced_failure_analysis_report.json")
            print(f"  - outputs/enhanced_failure_analysis_report.md")
        
        print("\n" + "=" * 60)
        print("✅ All Tests Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
