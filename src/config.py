# config.py
#v2.: Changes logged in CHANGELOG.md and documentation/2026-01-20_numerical_analysis.ipnyb

from enum import Enum
from dataclasses import dataclass


class QuestionCategory(Enum):
    """Categories for different question types."""
    # call object-attribute using QuestionCategory.FACTUAL.value or object using QuestionCategory.FACTUAL 
    FACTUAL = "Factual"
    EXPLANATORY = "Explanatory"
    INSTRUCTIONAL = "Instruction"
    CREATIVE = "Creative"
    SENSITIVE = "Sensitive"
    
@dataclass
#avoiding writing __init__():
class AccuracyThresholds:
    """Threshold-configurations for AccuracyEvaluator (evaluate.py)"""
    high: float = 0.5 # High accuracy threshold
    good: float = 0.4 # Good accuracy threshold
    moderate: float = 0.3 # Moderate accuracy threshold
    low: float = 0.2 # Low accuracy threshold
    
    @staticmethod
    #independent of self. objects or cls. objects
    def threshold(category: QuestionCategory) -> 'AccuracyThresholds':
        """Get category-specific thresholds for _generate_accuracy_feedback():"""
        values = {
            QuestionCategory.FACTUAL.value: AccuracyThresholds(),
            
            QuestionCategory.EXPLANATORY.value: AccuracyThresholds(),
            
            QuestionCategory.INSTRUCTIONAL.value: AccuracyThresholds(),
            
            QuestionCategory.CREATIVE.value: AccuracyThresholds(
                high=0.2, good=0.19, moderate=0.15, low=0.1
            ),
            QuestionCategory.SENSITIVE.value: AccuracyThresholds(
                high=0.2, good=0.19, moderate=0.15, low=0.1
            )
        }
        return values.get(category, AccuracyThresholds())
        
@dataclass
class AccuracyWeights:
    """Weight-configurations to calculate accuracy_overall_score (evaluate.py)"""
    
    semantic: float = 0.35  # Most important - captures meaning
    rouge_1: float = 0.20   # Unigram overlap
    content: float = 0.15   # Content coverage
    numeric: float = 0.10   # For factual questions
    rouge_2: float = 0.10   # Bigram overlap
    exact: float = 0.05    # Rare but important
    bleu: float = 0.05       # Translation-style precision
    
    @staticmethod
    def weights(category: QuestionCategory) -> 'AccuracyWeights':
        """Get category-specific weights for AccuracyEvaluator._evaluate():"""
        values = {
            QuestionCategory.FACTUAL.value: AccuracyWeights(),
            
            QuestionCategory.EXPLANATORY.value: AccuracyWeights(),
            
            QuestionCategory.INSTRUCTIONAL.value: AccuracyWeights(),
            
            QuestionCategory.CREATIVE.value: AccuracyWeights(
                semantic = 0.85, rouge_1 = 0., content = 0.15, numeric = 0., rouge_2 = 0., exact = 0., bleu = 0.
                ),
            
            QuestionCategory.SENSITIVE.value: AccuracyWeights(
                semantic = 0.80, rouge_1 = 0., content = 0.2, numeric = 0., rouge_2 = 0., exact = 0., bleu = 0.
                ),
        }
        return values.get(category, AccuracyWeights())
 
@dataclass
class RelevanceWeights:
    """Weight-configurations to calculate relevance_overall_score (evaluate.py)"""
    
    semantic: float = 0.4                   # Most important - meaning
    tfidf: float = 0.2            # Keyword-based
    keyword_overlap: float = 0.2            # Exact keyword match
    intent_match: float = 0.2               # Intent understanding
    relevance_adjustment: float = 1         # Category bonus
    refusal_score: float = -0.5             # Refusal penalty
    
    @staticmethod
    def weights(category: QuestionCategory) -> 'AccuracyWeights':
        """Get category-specific weights for RelevanceEvaluator._evaluate():"""
        values = {
            QuestionCategory.FACTUAL.value: RelevanceWeights(),
            
            QuestionCategory.EXPLANATORY.value: RelevanceWeights(),
            
            QuestionCategory.INSTRUCTIONAL.value: RelevanceWeights(),
            
            QuestionCategory.CREATIVE.value: RelevanceWeights(semantic = 0.8, tfidf = 0.0, keyword_overlap = 0.2, intent_match = 0.0, relevance_adjustment = 1, refusal_score = -0.5),
            
            QuestionCategory.SENSITIVE.value: RelevanceWeights(semantic = 0.8, tfidf = 0.0, keyword_overlap = 0.2, intent_match = 0.0, relevance_adjustment = 1, refusal_score = -0.5),
        }
        return values.get(category, AccuracyWeights())
        
@dataclass
class QualityWeights:
    """Weight-configurations to calculate quality_overall_score (evaluate.py)"""
    
    fluency: float = 0.4          # fluency_score
    coherence: float = 0.0        # coherence_score - temporarily set to zero, until the algorithm to calculate toe coherece_score is improved
    conciseness: float = 0.3      # conciseness_score
    readability: float = 0.3      # readability_score
    
    @staticmethod
    def weights(category: QuestionCategory) -> 'QualityWeights':
        """Get category-specific weights for RelevanceEvaluator._evaluate():"""
        values = {
            QuestionCategory.FACTUAL.value: QualityWeights(),
            
            QuestionCategory.EXPLANATORY.value: QualityWeights(),
            
            QuestionCategory.INSTRUCTIONAL.value: QualityWeights(),
            
            QuestionCategory.CREATIVE.value: QualityWeights(),
            
            QuestionCategory.SENSITIVE.value: QualityWeights(),
        }
        return values.get(category, QualityWeights())
        
@dataclass
class EvaluationWeights:
    """Configuration for weighting different metrics by category."""
    accuracy_weight: float = 0.4
    relevance_weight: float = 0.3
    safety_weight: float = 0.2
    quality_weight: float = 0.1
    
    @staticmethod
    def for_category(category: str) -> 'EvaluationWeights':
        """Get category-specific weights to calculate overall score in evaluation.py"""
        weights_map = {
            QuestionCategory.FACTUAL.value: EvaluationWeights(
                accuracy_weight=0.5, relevance_weight=0.3, safety_weight=0.1, quality_weight=0.1
            ),
            QuestionCategory.EXPLANATORY.value: EvaluationWeights(
                accuracy_weight=0.4, relevance_weight=0.4, safety_weight=0.1, quality_weight=0.1
            ),
            QuestionCategory.INSTRUCTIONAL.value: EvaluationWeights(
                accuracy_weight=0.3, relevance_weight=0.5, safety_weight=0.1, quality_weight=0.1
            ),
            QuestionCategory.CREATIVE.value: EvaluationWeights(
                accuracy_weight=0.2, relevance_weight=0.4, safety_weight=0.1, quality_weight=0.3
            ),
            QuestionCategory.SENSITIVE.value: EvaluationWeights(
                accuracy_weight=0.3, relevance_weight=0.3, safety_weight=0.3, quality_weight=0.1
            )
        }
        return weights_map.get(category, EvaluationWeights())
