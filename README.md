# LLM Evaluation Framework

**A Comprehensive Toolkit for Evaluating, Analyzing, and Visualizing Large Language Model Performance**

A Python framework for evaluating LLM outputs with automated scoring, intelligent failure analysis, and interactive visualizations.

## Features

### **Multi-Metric Evaluation**
- **Accuracy Scoring**: ROUGE, BLEU, semantic similarity, content coverage
- **Relevance Analysis**: Semantic relevance using sentence transformers
- **Safety Detection**: Bias, stereotypes, misinformation patterns
- **Quality Assessment**: Fluency, coherence, conciseness, readability
- **Category-Aware**: Different weights for factual, creative, sensitive questions
- (View detailed metrics descriptions with their respective equations at [evaluate_documentation.ipynb](documentation/evaluate_documentation.ipynb) )

### **Intelligent Failure Analysis**
- **Root Cause Diagnosis**: Automatic failure categorization
- **Confidence Scoring**: Statistical confidence in failure diagnoses
- **Actionable Suggestions**: Specific improvement recommendations
- **Performance Insights**: Category-wise and metric-wise analysis
- (View detailed metrics descriptions at [analyzer_documentation.ipynb](documentation/analyzer_documentation.ipynb) )

### **Interactive Visualization**
- **Dashboard**: 9-panel interactive dashboard (HTML) hosted on GitHub Pages ([View the Live Dashboard](https://lorenapuhl.github.io/llm-evaluation-framework/))
- **Score Distributions**: Histograms, box plots, density plots
- **Failure Breakdowns**: Pie charts, heatmaps, bar charts
- **Correlation Analysis**: Heatmaps and scatter plots
- **HTML Reports**: Professional reports with recommendations

### **Production Ready**
- **Parallel Processing**: Fast batch evaluation
- **Graceful Degradation**: Works with fallback methods when heavier dependencies cannot be imported
- **Configurable**: Adjustable thresholds, weights and bias-patterns
- **Modular Architecture**: Easy to extend with new metrics
- **Comprehensive Testing**: Full test suite with sample data

## Project Structure

```
llm-evaluation-framework/
│
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Installation script
├── 📄 index.html                   # publish https://lorenapuhl.github.io/llm-evaluation-framework/
├── 📄 LICENSE.txt                  # MIT License
│
├── 📂 data/                       # Test data
│   ├── test_questions.tsv         # Sample questions with categories
│   ├── test_responses.tsv         # Sample LLM responses
│   ├── create_test_questions.py   # Script to generate test questions
│   └── create_test_responses.py   # Script to generate test responses
│
├── 📂 src/                        # Core framework
│   ├── evaluate.py                # Enhanced evaluation system
│   ├── analyze.py                 # Failure analysis module
│   └── visualization.py           # Visualization and dashboard
│
├── 📂 benchmarks/                  # Testing and benchmarking
│   ├── test_enhanced_evaluation.py # Evaluation tests
│   ├── test_analysis.py            # Analysis tests
│   ├── run_benchmark.py            # Complete pipeline runner
│   └── load_test_data.py           # Data loading utilities
│
├── 📂 outputs/                            # Generated outputs (created automatically)
│   ├── enhanced_evaluation_results.tsv    # Evaluation scores
│   ├── enhanced_failure_analysis.tsv      # Failure analysis
│   ├── evaluation_report.md               # Evaluation summary
│   ├── failure_analysis_report.md         # Analysis summary
│   ├── llm_evaluation_dashboard.html      # Interactive dashboard
│   ├── llm_evaluation_report.html         # HTML report
│   └── *.png                              # Visualization charts
│
└── 📂 documemtation/                    # Usage examples
    ├──  example_usage.ipynb             # Jupyter notebook with examples
    ├──  evaluate_documentation.ipynb    # evaluate.py documentation
    ├──  improvements.ipynb              # pending improvements
    └──  analyzer_documentation.ipynb     # analyze.py documentation
```

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/llm-evaluation-framework.git
cd llm-evaluation-framework
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
# Run the complete evaluation pipeline
python benchmarks/run_benchmark.py

# With cleanup of previous outputs
python benchmarks/run_benchmark.py --clean
```

This will:
1. ✅ Load test data
2. ✅ Evaluate all question-response pairs
3. ✅ Perform failure analysis
4. ✅ Generate visualizations
5. ✅ Create comprehensive reports

### View Results

After running, open the interactive dashboard:
```bash
# Mac
open outputs/llm_evaluation_dashboard.html

# Windows
start outputs/llm_evaluation_dashboard.html

# Linux
xdg-open outputs/llm_evaluation_dashboard.html
```

## Usage Examples

### Basic Evaluation

```python
from src.evaluate import EnhancedLLMEvaluator

# Initialize evaluator
evaluator = EnhancedLLMEvaluator()

# Evaluate a single question-response pair
result = evaluator.evaluate_single_pair(
    question="What is the capital of France?",
    reference="Paris is the capital city of France.",
    response="Paris is the capital of France, located in Europe.",
    category="Factual"
)

print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Accuracy: {result['composite_accuracy']:.3f}")
print(f"Relevance: {result['composite_relevance']:.3f}")
print(f"Primary Failure: {result['primary_failure_mode']}")
```

### Batch Evaluation

```python
import pandas as pd
from src.evaluate import evaluate_all_pairs_enhanced

# Load your data
questions = pd.read_csv('data/test_questions.tsv', sep='\t')
responses = pd.read_csv('data/test_responses.tsv', sep='\t')

# Run batch evaluation
results = evaluate_all_pairs_enhanced(questions, responses)

# Save results
results.to_csv('my_evaluation_results.tsv', sep='\t', index=False)
```

### Failure Analysis

```python
from src.analyze import EnhancedFailureAnalyzer

# Initialize analyzer
analyzer = EnhancedFailureAnalyzer()

# Load evaluation results
results = pd.read_csv('outputs/enhanced_evaluation_results.tsv', sep='\t')

# Analyze failures
analyzed_df, summary, report = analyzer.analyze_dataset(results)

print(f"Success Rate: {summary['success_rate']:.1f}%")
print(f"Top Failure: {summary['category_breakdown']}")
```

### Generate Visualizations

```python
from src.visualize import LLMVisualizer

# Initialize visualizer
visualizer = LLMVisualizer()

# Generate all visualizations
files = visualizer.generate_all_visualizations()
print(f"Generated: {len(files)} files")

# Generate specific visualization
visualizer.plot_score_distribution()
visualizer.plot_failure_breakdown()
```

## Key Features in Detail

### 1. Enhanced Evaluation System

The framework evaluates LLM responses across four dimensions with category-specific weights:

| Dimension | Metrics | Factual Weight | Creative Weight |
|-----------|---------|----------------|-----------------|
| **Accuracy** | ROUGE-1/2, BLEU, semantic similarity, content coverage | 50% | 20% |
| **Relevance** | Semantic relevance, topic alignment, refusal detection | 30% | 40% |
| **Safety** | Bias detection, harmful content, misinformation patterns | 10% | 10% |
| **Quality** | Fluency, coherence, conciseness, readability | 10% | 30% |

### 2. Intelligent Failure Analysis

Automatically categorizes failures and provides actionable insights:

- **Factual Errors**: Incorrect or inaccurate information
- **Irrelevant Responses**: Off-topic or missing key information
- **Safety Issues**: Bias, stereotypes, harmful content
- **Quality Problems**: Poor structure, unclear, repetitive
- **Refusals**: LLM refuses to answer appropriately

### 3. Interactive Dashboard

The framework generates an interactive HTML dashboard with:

| Panel | Description |
|-------|-------------|
| Overall Score Distribution | Histogram with mean/median lines |
| Failure Category Breakdown | Interactive pie/donut chart |
| Performance by Category | Bar charts with category colors |
| Composite Metrics Comparison | Side-by-side metric comparison |
| Failure Confidence | Distribution of confidence scores |
| Metric Correlations | Heatmap showing metric relationships |
| Top Failure Examples | Table with specific failure cases |
| Accuracy vs Relevance | Scatter plot with trendline |
| Quality Metrics Distribution | Box plots for each quality metric |

## Configuration

### Evaluation Thresholds

Customize evaluation thresholds in your code:

```python
custom_threshold = {
                'accuracy': 0.7, # Higher accuracy requirement
                'relevance': 0.6, # Lower relevance requirement
                'safety': 0.9, # Stricter safety requirement
                'quality': 0.4 # More lenient quality requirement
            }
threshold_evaluator = EnhancedLLMEvaluator(custom_threshold = custom_threshold)
```

### Category Weights

Modify weights for different question types:

```python
custom_weights = {
    'Factual': {'accuracy': 0.6, 'relevance': 0.2, 'safety': 0.1, 'quality': 0.1},
    'Creative': {'accuracy': 0.1, 'relevance': 0.3, 'safety': 0.1, 'quality': 0.5},
    'Technical': {'accuracy': 0.4, 'relevance': 0.4, 'safety': 0.1, 'quality': 0.1},
    'CustomerService': {'accuracy': 0.3, 'relevance': 0.4, 'safety': 0.2, 'quality': 0.1},
}

custom_evaluator = EnhancedLLMEvaluator(custom_weights=custom_weights['Technical'])
```

### Bias-patterns

Add custom bias-patterns to enhance safety-mechanisms

```python
custom_patterns = {
    'financial_misinfo': [r'\b(stock|investment|crypto)\b\s+(?:will|going to)\s+(?:double|triple|10x)\b'],
    'political_bias': [r'\b(democrat|republican)\b\s+(?:are|is)\s+(?:evil|corrupt|stupid)\b'],
    'health_claims': [r'\b(this product|cure|treats)\b\s+(?:all|every)\s+(?:disease|illness|condition)\b'],
}

custom_safety_evaluator = EnhancedLLMEvaluator(custom_patterns=custom_patterns)
```

## Sample Output

### Evaluation Results (TSV format):
| id | category | overall_score | composite_accuracy | primary_failure_mode | improvement_suggestions |
|----|----------|---------------|-------------------|---------------------|------------------------|
| 1 | Factual | 0.85 | 0.92 | pass | None required |
| 2 | Creative | 0.45 | 0.30 | irrelevant_response | "Better address the specific creative intent" |
| 3 | Sensitive | 0.70 | 0.65 | safety_issue | "Present more balanced perspectives" |

### Dashboard Preview:

[View the Live Dashboard](https://lorenapuhl.github.io/llm-evaluation-framework/)

## Testing

Run the test suite:

```bash
# Test evaluation module
python benchmarks/test_enhanced_evaluation.py

# Test analysis module
python benchmarks/test_analysis.py

# Run all tests
python benchmarks/run_benchmark.py --clean
```

## Advanced Usage

### Custom Metrics

Add your own evaluation metrics:

```python
from src.evaluate import EnhancedLLMEvaluator

class CustomEvaluator(EnhancedLLMEvaluator):
    def evaluate_custom_metric(self, response):
        # Implement your custom metric
        custom_score = self.calculate_custom_score(response)
        return custom_score
```

### Integration with LLM APIs

```python
import openai
from src.evaluate import EnhancedLLMEvaluator

# Get LLM response
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": question}]
)

# Evaluate the response
evaluator = EnhancedLLMEvaluator()
result = evaluator.evaluate_single_pair(
    question=question,
    reference=reference_answer,
    response=response.choices[0].message.content,
    category=category
)
```
(View more detailed explanations in [example_usage.ipynb](documentation/example_usage.ipynb)

## Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/llm-evaluation-framework.git
cd llm-evaluation-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENCE.txt) file for details.

##  Support
- **Email**: lorena.puhl@protonmail.com

## Roadmap

- [ ] Implement real-time evaluation API functionality
- [ ] Implement LLM model integration
- [ ] Refine mathematical models and evaluation algoithms for additional precision
- [ ] Implement cloud deployment templates

