# CHANGELOG.md

    
## [2.0.0] - 2026-01-29
### Changed
- **src.config.py: FailureThresholds**: Added `FailureWeights` in `config.py` to make weights better configurable and better adapted to the results from `evaluate.py`
- **src.analyze.py: EnhancedFailureAnalyzer.__init__()**: Function now uses custom-threshold values
- **Reason**: Threshold values were to high for the average results from `evaluate.py` and did not accurately reflect the failure/success rates of llm-answers
- **Analysis**: See notebook `2026-01-20_numerical_analysis.ipynb`, section 6
- **Impact**: Failure/success rates are now beter adapted
- **commit**: 76a1da5
    ```
## [2.0.0] - 2026-01-29
### Changed
- **src.config.py: RelevanceThresholds**: Added question-category-specific thresholds better adapted to the values of `relevance_overall_score`
- **src.evaluate.py: AccuracyEvaluator._generate_relevance_feedback()**: Function now uses custom-threshold values and shows caution message for creative and sensitive questions
- **Reason**: `relevance_overall_score`- values are generally lower, since algorithms to determine this measure need to be refined. This is a temporary solution. 
- **Analysis**: See notebook `2026-01-20_numerical_analysis.ipynb`, section 2
- **Impact**: `relevance_feedback` better reflects the `relevance_overall_score`
- **commit**: 35ebbd8

## [2.0.0] - 2026-01-28
### Changed
- **src.config.py: QualityWeights**: Added custom-weights when calulating the `quality_overall_score`
- **src.evaluate.py: QualityEvaluator._evaluate()**: Added `QualityWeights` in `config.py` as a provisional solution for the inaccurate algorithm in `QualityEvaluator._check_coherence()`. Weights for `coherence_score` are set to zero.
- **Reason**: Vocabulary in `QualityEvaluator._check_coherence()` is not diverse enough for an apropriate analysis
- **Analysis**: See notebook `2026-01-20_numerical_analysis.ipynb`, section 3
- **Impact**: Results of `quality_overall_score` are not impacted by the insufficiently complex algorithm in `QualityEvaluator._check_coherence()`
- **commit**: f2bfbaf


## [2.0.0] - 2026-01-28
### Changed
- **src.evaluate.py: QualityEvaluator._check_readability()**: - Change normalization-factor to `normalized = 1.0 - min(1.0, readability_score / 100)`
- **Reason**: Normalization factors were not adapted to the typical length and type of questions/llm-answers
- **Analysis**: See notebook `2026-01-20_numerical_analysis.ipynb`, section 3
- **Impact**: readability_scores are now more representative of the readability of llm-answers
- **commit**: 541bb14


## [2.0.0] - 2026-01-28
### Changed
- **src.config.py: RelevanceWeights**: Added special handling for Creative/Sensitive questions and question-category-specific weights when calulating the relevance_overall_score
- **src.evaluate.py: RelelvanceEvaluator._evaluate()**: Function now uses category-specifc weights from RelvanceWeights.values(category)
- **Reason**: Accuracy measures not appropriate for creative and sensitive questions
- **Analysis**: See notebook `2026-01-20_numerical_analysis.ipynb`, section 2
- **Impact**: Overall_scores for creative and questions are better fine-tuned to their category-specific nature
- **commit**: f774b36


## [2.0.0] - 2026-01-27
### Changed
- **src.config.py: AccuracyWeights**: Added special handling for Creative/Sensitive questions and question-category-specific weights when calulating the accuracy_overall_score
- **src.evaluate.py: AccuracyEvaluator._evaluate()**: Function now uses category-specifc weights from AccuracyWeights-values(category)
- **Reason**: Accuracy measures not appropriate for creative and sensitive questions
- **Analysis**: See notebook `2026-01-20_numerical_analysis.ipynb`, section 1
- **Impact**: Overall_scores for creative and questions are better fine-tuned to their category-specific nature
- **commit**: e3d9e44

## [2.0.0] - 2026-01-27
### Changed
- **src.analyze.py: EnhancesFailureAnalyzer**: Added ```_make_results_human_readable()``` funtion for readability of improvement suggestions and failure reasons 
- **Analysis**: See notebook ```2026-01-20_numerical_analysis.ipynb```, section 6
- **Impact**: Users now see strings instead of lists
- **commit**: e498cfd

## [2.0.0] - 2026-01-20
### Changed
- **src.config.py: AccuracyThresholds**: Added special handling for Creative/Sensitive questions and question-category-specific thresholds
- **src.evaluate.py: AccuracyEvaluator._generate_accuracy_feedback()**: Function now uses custom-threshold values and shows caution message for creative and sensitive questions
- **Reason**: N-gram metrics (BLEU/ROUGE-2) perform poorly on diverse responses and accuracy measures not appropriate for creative and sensitive questions
- **Analysis**: See notebook `2026-01-20_numerical_analysis.ipynb`, section 1
- **Impact**: Users now see warnings when metrics are unreliable and more appropriate feedback
- **commit**: 76a1da5

## [1.0.0] - 2026-01-14
### Added
- Initial evaluation system
