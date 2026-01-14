# create_test_data.py
import pandas as pd
import os

# Instead of absolute path, go up one level
os.chdir("..")  # From current dir to parent

# Then load data
test_qs = pd.read_csv('data/test_questions.tsv', sep='\t')  # Read with tabs
llm_responses = pd.read_csv('data/test_responses.tsv', sep='\t')

print("Test Questions:")
print(test_qs.head())
print(f"\nTotal test questions: {len(test_qs)}")
print(f"Categories: {test_qs['category'].unique()}")

print("\n" + "="*50 + "\n")

print("LLM Responses:")
print(llm_responses.head())
print(f"\nTotal LLM responses: {len(llm_responses)}")

# Quick merge to see correspondence
merged = pd.merge(test_qs, llm_responses, on='id', how='inner')
print(f"\nSuccessfully matched: {len(merged)} pairs")
