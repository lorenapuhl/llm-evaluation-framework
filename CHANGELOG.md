# CHANGELOG.md


## [2.0.0] - 2026-01-28
### Changed
- **src.config.py: RelevanceWeights**: Added special handling for Creative/Sensitive questions and question-category-specific weights when calulating the relevance_overall_score
- **src.evaluate.py: RelelvanceEvaluator._evaluate()**: Function now uses category-specifc weights from RelvanceWeights.values(category)
- **Reason**: Accuracy measures not appropriate for creative and sensitive questions
- **Analysis**: See notebook `2026-01-20_numerical_analysis.ipynb`, section 2
- **Impact**: Overall_scores for creative and questions are better fine-tuned to their category-specific nature
- **commit**: e3d9e44


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
