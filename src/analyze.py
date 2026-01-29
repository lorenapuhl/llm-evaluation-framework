"""
Enhanced Failure Analysis Module
Categorizes why LLM responses failed using enhanced evaluation metrics
"""
#set import directories
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import re
import json
from collections import Counter
from src.config import QuestionCategory, FailureThresholds

class EnhancedFailureAnalyzer:
    """Analyzes failed LLM responses using enhanced evaluation metrics."""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Initialize the analyzer with configurable thresholds.
        
        Args:
            thresholds: Dictionary of threshold values for different metrics
        """
        """v2.0.0: Add  custom-threshold values from config.py"""
        
        #change to variable category !
        values = FailureThresholds()
        
        self.thresholds = thresholds or {
            'accuracy_low': values.accuracy_low,      # Below this = likely factual error
            'accuracy_medium': values.accuracy_medium,   # Below this = partial correctness issues
            'relevance_low': values.relevance_low,     # Below this = off-topic
            'relevance_medium': values.relevance_medium,  # Below this = partial relevance
            'safety_low': values.safety_low,        # Below this = safety concern
            'quality_low': values.quality_low,       # Below this = quality issues
        }
        
        # Define failure categories
        self.failure_categories = {
            'factual error': {
                'description': 'The response contains incorrect facts or hallucinations',
                'subtypes': ['completely wrong', 'partially incorrect', 'missing content']
            },
            'incomplete': {
                'description': 'The response is correct but missing important information',
                'subtypes': ['too short', 'missing key points', 'unfinished']
            },
            'irrelevant': {
                'description': 'The response does not address the question',
                'subtypes': ['off topic', 'generic response', 'refusal']
            },
            'unsafe': {
                'description': 'The response contains biased, harmful, or inappropriate content',
                'subtypes': ['bias', 'harmful advice', 'sensitive content', 'unbalanced']
            },
            'poor quality': {
                'description': 'The response has formatting or language issues',
                'subtypes': ['unreadable', 'repetitive', 'incoherent', 'too verbose']
            },
            'prompt issue': {
                'description': 'The failure stems from ambiguous or problematic prompts',
                'subtypes': ['ambiguous question', 'complex instruction', 'conflicting requirements']
            },
            'no failure': {
                'description': 'The response meets all quality criteria',
                'subtypes': []
            }
        }
        
    def categorize_failure(self, row: pd.Series) -> Dict[str, Any]:
        """
        Categorize a single failure using enhanced evaluation metrics.
        
        Args:
            row: A pandas Series with enhanced evaluation metrics
            
        Returns:
            Dictionary with failure categorization details
        """
        # Extract key metrics from enhanced evaluation
        accuracy = row.get('composite_accuracy', 0)
        relevance = row.get('composite_relevance', 0)
        safety = row.get('composite_safety', 1.0)
        quality = row.get('composite_quality', 0)
        
        # Extract additional signals
        has_bias_risk = row.get('safety_has_bias_risk', False)
        is_refusal = row.get('is_refusal', False)
        primary_failure_mode = row.get('primary_failure_mode', 'pass')
        length_ok = row.get('quality_length_ok', True)
        
        # Initialize result structure
        result = {
            'primary_category': 'no failure',
            'sub_category': None,
            'confidence': 0.0,
            'reasons': [],
            'suggested_fixes': []
        }
        
        # If primary_failure_mode already exists from evaluation, use it as base
        if primary_failure_mode != 'pass':
            result = self._map_failure_mode_to_category(primary_failure_mode, row)
            
            #Make results human-readable
            return self._make_results_human_readable(result)
        
        # 1. Check for safety issues (highest priority)
        if has_bias_risk or safety < self.thresholds['safety_low']:
            result = self._categorize_safety_issue(row, safety, has_bias_risk)
            #Make results human-readable
            return self._make_results_human_readable(result)
            
        # 2. Check for refusals
        if is_refusal:
            result = self._categorize_refusal(row)

            #Make results human-readable
            return self._make_results_human_readable(result)
        
        # 3. Check for relevance issues
        if relevance < self.thresholds['relevance_low']:
            result = self._categorize_relevance_issue(row, relevance)

            #Make results human-readable
            return self._make_results_human_readable(result)
        
        # 4. Check for factual errors
        if accuracy < self.thresholds['accuracy_low']:
            result = self._categorize_accuracy_issue(row, accuracy)

            #Make results human-readable
            return self._make_results_human_readable(result)
        
        # 5. Check for quality issues
        if quality < self.thresholds['quality_low'] or not length_ok:
            result = self._categorize_quality_issue(row, quality, length_ok)

            #Make results human-readable
            return self._make_results_human_readable(result)
            
        # 6. Check for partial issues
        if relevance < self.thresholds['relevance_medium']:
            result['primary_category'] = 'irrelevant'
            result['sub_category'] = 'generic response'
            result['confidence'] = 0.7
            result['reasons'] = ['Response is somewhat related but not fully addressing the question']
            result['suggested_fixes'] = [
                'Make instructions more specific',
                'Provide better examples',
                'Ask model to think step-by-step'
            ]

            #Make results human-readable
            return self._make_results_human_readable(result)
        
        if accuracy < self.thresholds['accuracy_medium']:
            result['primary_category'] = 'factual error'
            result['sub_category'] = 'partially incorrect'
            result['confidence'] = 0.65
            result['reasons'] = ['Response has minor factual inaccuracies']
            result['suggested_fixes'] = [
                'Provide more context in prompt',
                'Implement verification steps',
                'Use retrieval-augmented generation'
            ]
            
            #Make results human-readable
            return self._make_results_human_readable(result)
        
        # 8. If all checks pass
        result['confidence'] = 0.9
        result['reasons'] = ['Response meets all quality criteria']
        result['suggested_fixes'] = ['None required']
        
        #Make results human-readable
        return self._make_results_human_readable(result)


    def _make_results_human_readable(self, result:dict):
        """Make suggest_fixes and reasons human-readble"""
        
        fixes = result['suggested_fixes']
        result['suggested_fixes'] = '\n'.join([f"• {fix}" for fix in fixes])
        result['suggested_fixes'] = result['suggested_fixes'].replace('• •', '•')
        
        result['reasons'] = '; '.join(result['reasons'])
        return result

    
    def _map_failure_mode_to_category(self, failure_mode: str, row: pd.Series) -> Dict[str, Any]:
        """Map evaluation failure mode to failure category."""
        mapping = {
            'refusal to answer': ('irrelevant', 'refusal'),
            'safety issue': ('unsafe', self._get_safety_subtype(row)),
            'irrelevant response': ('irrelevant', 'off-topic'),
            'factual error': ('factual error', self._get_accuracy_subtype(row)),
            'partial relevance': ('irrelevant', 'generic response'),
            'partial accuracy': ('factual error', 'partially incorrect'),
            'pass': ('no failure', None)
        }
        
        primary, sub = mapping.get(failure_mode, ('no failure', None))
        
        result = {
            'primary_category': primary,
            'sub_category': sub,
            'confidence': 0.8,  # High confidence since evaluation already flagged it
            'reasons': [f'Failure category: "{failure_mode}"'],
            'suggested_fixes': self._get_fixes_from_suggestions(row)
        }
        
        # Enhance with additional context
        if primary == 'unsafe':
            result['reasons'].append(row.get('safety_feedback', 'Safety concern detected'))
        elif primary == 'irrelevant':
            result['reasons'].append(row.get('relevance_feedback', 'Relevance issue detected'))
        elif primary == 'factual error':
            result['reasons'].append(row.get('accuracy_feedback', 'Accuracy issue detected'))
        
        return result
    
    def _get_safety_subtype(self, row: pd.Series) -> str:
        """Determine safety sub-category."""
        bias_categories = row.get('safety_bias_categories', [])
        if isinstance(bias_categories, str):
            # Parse string representation of list
            if bias_categories.startswith('['):
                try:
                    bias_categories = eval(bias_categories)
                except:
                    bias_categories = []
        
        if 'climate denial' in bias_categories or 'medical misinformation' in bias_categories:
            return 'bias'
        elif 'harmful stereotypes' in bias_categories:
            return 'sensitive content'
        elif row.get('safety_perspective_balance', 1.0) < 0.5:
            return 'unbalanced'
        else:
            return 'bias'  # Default
    
    def _get_accuracy_subtype(self, row: pd.Series) -> str:
        """Determine accuracy sub-category."""
        accuracy = row.get('composite_accuracy', 0)
        content_coverage = row.get('accuracy_content_coverage', 0)
        
        if accuracy < 0.2:
            return 'completely wrong'
        elif content_coverage < 0.5:
            return 'missing content'
        else:
            return 'partially incorrect'
    
    def _get_fixes_from_suggestions(self, row: pd.Series) -> List[str]:
        """Extract fixes from evaluation suggestions."""
        suggestions = row.get('improvement_suggestions', [])
        if isinstance(suggestions, str):
            # Parse string representation
            if suggestions.startswith('['):
                try:
                    suggestions = eval(suggestions)
                except:
                    suggestions = []
            else:
                suggestions = [suggestions]
        
        if suggestions and isinstance(suggestions, list):
            return suggestions[:3]  # Return top 3
        
        return ['Review and improve response quality']
    
    def _categorize_safety_issue(self, row: pd.Series, safety: float, has_bias_risk: bool) -> Dict[str, Any]:
        """Categorize safety-related failures."""
        result = {
            'primary_category': 'unsafe',
            'sub_category': self._get_safety_subtype(row),
            'confidence': 0.9,
            'reasons': [],
            'suggested_fixes': [
                'Add safety guidelines to prompt',
                'Implement content filtering',
                'Fine-tune model on safety data'
            ]
        }
        
        if has_bias_risk:
            bias_cats = row.get('safety_bias_categories', [])
            if bias_cats:
                result['reasons'].append(f'Bias detected: {bias_cats}')
        
        if safety < 0.5:
            result['reasons'].append('Significant safety concerns')
            result['confidence'] = 0.95
        elif safety < 0.7:
            result['reasons'].append('Moderate safety concerns')
            result['confidence'] = 0.85
        
        if not result['reasons']:
            result['reasons'] = ['Response contains potentially unsafe content']
        
        return result
    
    def _categorize_refusal(self, row: pd.Series) -> Dict[str, Any]:
        """Categorize refusal failures."""
        refusal_score = row.get('relevance_refusal_score', 0)
        
        result = {
            'primary_category': 'irrelevant',
            'sub_category': 'refusal',
            'confidence': min(0.95, 0.7 + refusal_score),
            'reasons': ['Model refused to answer the question'],
            'suggested_fixes': [
                'Adjust prompt tone',
                'Provide context for sensitive topics',
                'Use system prompts to guide behavior'
            ]
        }
        
        if refusal_score > 0.8:
            result['reasons'].append('Strong refusal pattern detected')
        
        return result
    
    def _categorize_relevance_issue(self, row: pd.Series, relevance: float) -> Dict[str, Any]:
        """Categorize relevance-related failures."""
        intent_match = row.get('relevance_intent_match', 0.5)
        semantic_relevance = row.get('relevance_semantic_relevance', 0)
        
        result = {
            'primary_category': 'irrelevant',
            'confidence': 0.8,
            'reasons': [],
            'suggested_fixes': [
                'Add more specific instructions',
                'Use few-shot examples',
                'Implement question-answering verification'
            ]
        }
        
        if intent_match < 0.3:
            result['sub_category'] = 'off-topic'
            result['reasons'].append('Response does not match question intent')
        elif semantic_relevance < 0.3:
            result['sub_category'] = 'off-topic'
            result['reasons'].append('Response semantically unrelated to question')
        else:
            result['sub_category'] = 'generic response'
            result['reasons'].append('Response is too generic or vague')
        
        if relevance < 0.2:
            result['confidence'] = 0.9
            result['reasons'].append('Very low relevance score')
        
        return result
    
    def _categorize_accuracy_issue(self, row: pd.Series, accuracy: float) -> Dict[str, Any]:
        """Categorize accuracy-related failures."""
        semantic_similarity = row.get('accuracy_semantic_similarity', 0)
        content_coverage = row.get('accuracy_content_coverage', 0)
        
        result = {
            'primary_category': 'factual error',
            'confidence': 0.75,
            'reasons': [],
            'suggested_fixes': [
                'Provide verified facts in prompt',
                'Implement fact-checking step',
                'Fine-tune on domain-specific data'
            ]
        }
        
        if accuracy < 0.2:
            result['sub_category'] = 'completely wrong'
            result['confidence'] = 0.85
            result['reasons'].append('Major factual errors or hallucinations')
        elif content_coverage < 0.4:
            result['sub_category'] = 'missing content'
            result['reasons'].append('Missing key information from reference')
        else:
            result['sub_category'] = 'partially incorrect'
            result['reasons'].append('Response mixes correct and incorrect information')
        
        if semantic_similarity < 0.3:
            result['reasons'].append('Low semantic similarity to reference')
        
        return result
    
    def _categorize_quality_issue(self, row: pd.Series, quality: float, length_ok: bool) -> Dict[str, Any]:
        """Categorize quality-related failures."""
        coherence = row.get('quality_coherence_score', 0)
        conciseness = row.get('quality_conciseness_score', 0)
        readability = row.get('quality_readability_score', 0)
        length_feedback = row.get('quality_length_feedback', '')
        
        result = {
            'primary_category': 'poor quality',
            'confidence': 0.7,
            'reasons': [],
            'suggested_fixes': [
                'Improve model temperature settings',
                'Add post-processing step',
                'Use grammar correction tools'
            ]
        }
        
        if not length_ok:
            result['sub_category'] = 'too verbose' if 'too long' in str(length_feedback).lower() else 'too_short'
            result['reasons'].append(f'Length issue: {length_feedback}')
        elif coherence < 0.5:
            result['sub_category'] = 'incoherent'
            result['reasons'].append('Poor logical flow and coherence')
        elif conciseness < 0.5:
            result['sub_category'] = 'repetitive'
            result['reasons'].append('Response is repetitive or verbose')
        elif readability < 0.5:
            result['sub_category'] = 'unreadable'
            result['reasons'].append('Poor readability and sentence structure')
        else:
            result['sub_category'] = 'unreadable'
            result['reasons'].append('General quality issues detected')
        
        if quality < 0.3:
            result['confidence'] = 0.8
            result['reasons'].append('Very low quality score')
        
        return result
    
    def analyze_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze an entire dataset of enhanced evaluation results.
        
        Args:
            df: DataFrame with enhanced evaluation results
            
        Returns:
            DataFrame with added failure analysis columns
        """
        # Create a copy to avoid modifying original
        results_df = df.copy()
        
        # Analyze each row
        analyses = []
        for idx, row in results_df.iterrows():
            analysis = self.categorize_failure(row)
            analyses.append(analysis)
        
        # Convert analyses to DataFrame
        analysis_df = pd.DataFrame(analyses)
        
        # Add analysis columns to results
        for col in analysis_df.columns:
            results_df[f'failure_{col}'] = analysis_df[col]
        
        return results_df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics from failure analysis.
        
        Args:
            df: DataFrame with failure analysis columns
            
        Returns:
            Dictionary with summary statistics
        """
        if 'failure_primary_category' not in df.columns:
            raise ValueError("DataFrame must contain failure analysis columns. Run analyze_dataset first.")
        
        summary = {
            'total_responses': len(df),
            'failed_responses': len(df[df['failure_primary_category'] != 'no failure']),
            'success_rate': len(df[df['failure_primary_category'] == 'no failure']) / len(df) * 100,
            'overall_score_mean': df['overall_score'].mean() if 'overall_score' in df.columns else 0,
            'overall_score_std': df['overall_score'].std() if 'overall_score' in df.columns else 0,
        }
        
        # Category breakdown
        category_counts = df['failure_primary_category'].value_counts()
        summary['category_breakdown'] = category_counts.to_dict()
        
        # Subcategory breakdown
        if 'failure_sub_category' in df.columns:
            subcategory_counts = df['failure_sub_category'].value_counts()
            summary['subcategory_breakdown'] = subcategory_counts.to_dict()
        
        # Average confidence by category
        if 'failure_confidence' in df.columns:
            confidence_by_category = df.groupby('failure_primary_category')['failure_confidence'].mean()
            summary['confidence_by_category'] = confidence_by_category.to_dict()
        
        # Most common suggested fixes
        if 'failure_suggested_fixes' in df.columns:
            # Flatten list of lists
            all_fixes = []
            for fixes in df['failure_suggested_fixes']:
                if isinstance(fixes, str):
                    if fixes.startswith('['):
                        try:
                            fix_list = eval(fixes)
                            if isinstance(fix_list, list):
                                all_fixes.extend(fix_list)
                        except:
                            if fixes != 'None required':
                                all_fixes.append(fixes)
                    elif fixes != 'None required':
                        all_fixes.append(fixes)
                elif isinstance(fixes, list):
                    all_fixes.extend(fixes)
            
            fix_counts = Counter(all_fixes)
            summary['top_suggested_fixes'] = dict(fix_counts.most_common(10))
        
        # Correlation with overall score
        if 'overall_score' in df.columns and 'failure_confidence' in df.columns:
            summary['score_confidence_correlation'] = df['overall_score'].corr(df['failure_confidence'])
        
        # Performance by category
        if 'category' in df.columns:
            category_performance = {}
            for cat in df['category'].unique():
                cat_df = df[df['category'] == cat]
                cat_perf = {
                    'count': len(cat_df),
                    'avg_score': cat_df['overall_score'].mean() if 'overall_score' in cat_df.columns else 0,
                    'failure_rate': len(cat_df[cat_df['failure_primary_category'] != 'no failure']) / len(cat_df) * 100
                }
                category_performance[cat] = cat_perf
            summary['category_performance'] = category_performance
        
        return summary
    
    def get_examples_by_failure_type(self, df: pd.DataFrame, 
                                    failure_type: str = None, 
                                    n_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Get example responses for a specific failure type.
        
        Args:
            df: DataFrame with failure analysis
            failure_type: Specific failure type to filter by
            n_examples: Number of examples to return
            
        Returns:
            List of dictionaries with example details
        """
        if failure_type:
            filtered_df = df[df['failure_primary_category'] == failure_type]
        else:
            filtered_df = df[df['failure_primary_category'] != 'no failure']
        
        # Sort by confidence (highest confidence failures first)
        if 'failure_confidence' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('failure_confidence', ascending=False)
        
        examples = []
        for idx, row in filtered_df.head(n_examples).iterrows():
            # Safely get improvement suggestions
            suggestions = row.get('improvement_suggestions', [])
            if isinstance(suggestions, str) and suggestions.startswith('['):
                try:
                    suggestions = eval(suggestions)
                except:
                    suggestions = []
            
            example = {
                'id': row.get('id', idx),
                'category': row.get('category', 'N/A'),
                'question': row.get('question', 'N/A'),
                'llm_answer': self._truncate_text(row.get('response', 'N/A'), 150),
                'overall_score': row.get('overall_score', 0.0),
                'primary_category': row.get('failure_primary_category', 'N/A'),
                'sub_category': row.get('failure_sub_category', 'N/A'),
                'confidence': row.get('failure_confidence', 0.0),
                'composite_accuracy': row.get('composite_accuracy', 0.0),
                'composite_relevance': row.get('composite_relevance', 0.0),
                'composite_safety': row.get('composite_safety', 0.0),
                'composite_quality': row.get('composite_quality', 0.0),
                'reasons': row.get('failure_reasons', []),
                'suggested_fixes': row.get('failure_suggested_fixes', []),
                'evaluation_suggestions': suggestions[:3] if isinstance(suggestions, list) else [],
                'passed_all': all([
                    row.get('passed_accuracy', False),
                    row.get('passed_relevance', False),
                    row.get('passed_safety', False),
                    row.get('passed_quality', False)
                ]) if all(k in row for k in ['passed_accuracy', 'passed_relevance', 'passed_safety', 'passed_quality']) else False
            }
            examples.append(example)
        
        return examples
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text for display."""
        if not isinstance(text, str):
            return str(text)
        if len(text) <= max_length:
            return text
        return text[:max_length] + '...'
    
    def generate_detailed_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a detailed analysis report.
        
        Args:
            df: DataFrame with failure analysis
            
        Returns:
            Dictionary with detailed report
        """
        summary = self.generate_summary_statistics(df)
        
        report = {
            'summary': summary,
            'failure_analysis': {},
            'recommendations': []
        }
        
        # Analyze each failure category
        failure_categories = [cat for cat in df['failure_primary_category'].unique() 
                             if cat != 'no failure']
        
        for category in failure_categories:
            cat_df = df[df['failure_primary_category'] == category]
            
            analysis = {
                'count': len(cat_df),
                'percentage': len(cat_df) / len(df) * 100,
                'avg_overall_score': cat_df['overall_score'].mean(),
                'avg_confidence': cat_df['failure_confidence'].mean() if 'failure_confidence' in cat_df.columns else 0,
                'common_subcategories': cat_df['failure_sub_category'].value_counts().head(3).to_dict() 
                    if 'failure_sub_category' in cat_df.columns else {},
                'common_causes': self._extract_common_causes(cat_df),
                'top_examples': self.get_examples_by_failure_type(df, category, 2)
            }
            
            report['failure_analysis'][category] = analysis
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(df, summary)
        
        return report
    
    def _extract_common_causes(self, df: pd.DataFrame) -> List[str]:
        """Extract common causes from failure reasons."""
        all_reasons = []
        for reasons in df['failure_reasons']:
            if isinstance(reasons, str):
                if reasons.startswith('['):
                    try:
                        reason_list = eval(reasons)
                        if isinstance(reason_list, list):
                            all_reasons.extend(reason_list)
                    except:
                        all_reasons.append(reasons)
                else:
                    all_reasons.append(reasons)
            elif isinstance(reasons, list):
                all_reasons.extend(reasons)
        
        # Count and return top reasons
        reason_counts = Counter(all_reasons)
        return [reason for reason, _ in reason_counts.most_common(5)]
    
    def _generate_recommendations(self, df: pd.DataFrame, summary: Dict) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Check for systemic issues
        failure_rate = summary.get('failed_responses', 0) / summary.get('total_responses', 1)
        
        if failure_rate > 0.5:
            recommendations.append({
                'priority': 'high',
                'area': 'overall',
                'recommendation': 'High failure rate detected. Consider overall model improvement or prompt engineering.',
                'evidence': f'{failure_rate:.1%} of responses failed'
            })
        
        # Check specific failure types
        category_breakdown = summary.get('category_breakdown', {})
        
        for category, count in category_breakdown.items():
            if category == 'no failure':
                continue
            
            percentage = count / summary['total_responses'] * 100
            
            if percentage > 20:
                recommendations.append({
                    'priority': 'high',
                    'area': category,
                    'recommendation': f'High incidence of {category}. Focus improvement efforts here.',
                    'evidence': f'{percentage:.1f}% of responses have this issue'
                })
            elif percentage > 10:
                recommendations.append({
                    'priority': 'medium',
                    'area': category,
                    'recommendation': f'Moderate incidence of {category}. Monitor and address.',
                    'evidence': f'{percentage:.1f}% of responses have this issue'
                })
        
        # Check category-specific performance
        if 'category_performance' in summary:
            for category, perf in summary['category_performance'].items():
                if perf['failure_rate'] > 50:
                    recommendations.append({
                        'priority': 'high',
                        'area': f'{category} questions',
                        'recommendation': f'Poor performance on {category} questions. Adjust evaluation criteria or model training.',
                        'evidence': f'{perf["failure_rate"]:.1f}% failure rate for {category}'
                    })
        
        return recommendations


def analyze_failures_enhanced(evaluation_results_path: str,
                             output_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main function to run enhanced failure analysis.
    
    Args:
        evaluation_results_path: Path to enhanced evaluation results CSV/TSV
        output_path: Optional path to save analyzed results
        
    Returns:
        Tuple of (analyzed DataFrame, summary statistics)
    """
    # Load evaluation results
    if evaluation_results_path.endswith('.tsv'):
        eval_df = pd.read_csv(evaluation_results_path, sep='\t')
    else:
        eval_df = pd.read_csv(evaluation_results_path)
    
    # Initialize analyzer
    analyzer = EnhancedFailureAnalyzer()
    
    # Run analysis
    analyzed_df = analyzer.analyze_dataset(eval_df)
    
    # Generate summary and report
    summary = analyzer.generate_summary_statistics(analyzed_df)
    report = analyzer.generate_detailed_report(analyzed_df)
    
    # Save results if output path provided
    if output_path:
        if output_path.endswith('.tsv'):
            analyzed_df.to_csv(output_path, sep='\t', index=False)
        else:
            analyzed_df.to_csv(output_path, index=False)
        
        # Save summary and report as JSON
        import os
        base_name = os.path.splitext(output_path)[0]
        
        with open(f'{base_name}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open(f'{base_name}_report.json', 'w') as f:
            json.dump(report, f, indent=2)
    
    return analyzed_df, summary, report

if __name__ == "__main__":
    # Example usage
    print("Testing Enhanced Failure Analyzer...")
    
    # Create a test case with enhanced evaluation format
    test_data = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'composite_accuracy': [0.2, 0.8, 0.5, 0.9],
        'composite_relevance': [0.9, 0.3, 0.7, 0.8],
        'composite_safety': [0.9, 0.9, 0.4, 0.9],
        'composite_quality': [0.8, 0.8, 0.8, 0.8],
        'safety_has_bias_risk': [False, False, True, False],
        'safety_bias_categories': ['', '', "['climate denial']", ''],
        'is_refusal': [False, False, False, False],
        'primary_failure_mode': ['factual error', 'irrelevant response', 'safety_issue', 'pass'],
        'overall_score': [0.4, 0.5, 0.3, 0.9],
        'category': ['Factual', 'Explanatory', 'Factual', 'Factual'],
        'question': ['Q1', 'Q2', 'Q3', 'Q4'],
        'response': ['Wrong answer', 'Off topic', 'Biased content', 'Good answer'],
        'improvement_suggestions': [
            ['Improve accuracy'],
            ['Improve relevance'],
            ['Fix bias'],
            ['None required']
        ]
    })
    
    analyzer = EnhancedFailureAnalyzer()
    analyzed = analyzer.analyze_dataset(test_data)
    
    print("\nTest Results:")
    print(analyzed[['id', 'failure_primary_category', 'failure_sub_category', 'failure_confidence']])
    
    summary = analyzer.generate_summary_statistics(analyzed)
    print("\nSummary Statistics:")
    print(f"Total: {summary['total_responses']}")
    print(f"Failed: {summary['failed_responses']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Categories: {summary['category_breakdown']}")
    
    # Get examples
    print("\nFailure Examples:")
    examples = analyzer.get_examples_by_failure_type(analyzed, 'unsafe', 1)
    for ex in examples:
        print(f"  ID {ex['id']}: {ex['primary_category']} -> {ex['sub_category']}")
        print(f"    Reason: {ex['reasons'][0] if ex['reasons'] else 'N/A'}")
