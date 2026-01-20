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
    def threshold(category: QuestionCategory) -> 'EvaluationWeights':
        """Get category-specific thresholds for _generate_accuracy_feedback():"""
        values = {
            QuestionCategory.FACTUAL.value: AccuracyThresholds(
                high=0.5, good=0.4, moderate=0.3, low=0.2
            ),
            QuestionCategory.EXPLANATORY.value: AccuracyThresholds(
                high=0.5, good=0.4, moderate=0.3, low=0.2
            ),
            QuestionCategory.INSTRUCTIONAL.value: AccuracyThresholds(
                high=0.5, good=0.4, moderate=0.3, low=0.2
            ),
            QuestionCategory.CREATIVE.value: AccuracyThresholds(
                high=0.2, good=0.19, moderate=0.15, low=0.1
            ),
            QuestionCategory.SENSITIVE.value: AccuracyThresholds(
                high=0.2, good=0.19, moderate=0.15, low=0.1
            )
        }
        return values.get(category, AccuracyThresholds())
        
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
