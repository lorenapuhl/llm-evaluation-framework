# CHANGELOG.md

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
