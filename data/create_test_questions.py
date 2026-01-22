import pandas as pd

# Create DataFrame from the data
data = {
    'id': list(range(1, 21)),
    'question': [
        "What is the capital of France?",
        "Who wrote 'Romeo and Juliet'?",
        "What is the chemical formula for water?",
        "Explain the concept of photosynthesis in 2-3 sentences.",
        "Write a short creative story about a robot learning to paint.",
        "What are the main causes of climate change?",
        "How do I bake chocolate chip cookies? Provide step-by-step instructions.",
        "What is 15 * 24?",
        "Describe the process of cellular respiration.",
        "Write a poem about the ocean.",
        "What are the health benefits of regular exercise?",
        "How do I change a flat tire?",
        "What is machine learning?",
        "Calculate the area of a circle with radius 5.",
        "Discuss the ethical implications of artificial intelligence.",
        "What is the population of Tokyo?",
        "How do plants reproduce?",
        "Write a motivational speech for students.",
        "What are the symptoms of COVID-19?",
        "How should society address income inequality?"
    ],
    'reference_answer': [
        "The capital of France is Paris",
        "The author is William Shakespeare",
        "The chemical formula of water is H2O",
        "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in chloroplasts and is essential for life on Earth as it produces oxygen and forms the basis of the food chain.",
        "A story about a robot discovering art through experimentation with colors and emotions.",
        "Burning fossil fuels, deforestation, industrial processes, and agricultural activities that release greenhouse gases.",
        "1. Preheat oven to 350°F. 2. Mix dry ingredients. 3. Cream butter and sugars. 4. Add eggs and vanilla. 5. Combine wet and dry ingredients. 6. Fold in chocolate chips. 7. Drop spoonfuls onto baking sheet. 8. Bake for 10-12 minutes.",
        "The product of 15 and 24 is 360.",
        "Cellular respiration is the process by which cells convert glucose and oxygen into ATP, carbon dioxide, and water. It occurs in mitochondria and includes glycolysis, the Krebs cycle, and the electron transport chain.",
        "A creative poem describing ocean waves, marine life, and human connection to the sea.",
        "Improved cardiovascular health, stronger muscles and bones, better mental health, weight management, and reduced risk of chronic diseases.",
        "A step-by-step guide including safety precautions, jack placement, lug nut removal, tire replacement, and tightening.",
        "Machine learning is a subset of AI that enables systems to learn patterns from data without explicit programming, using algorithms to make predictions or decisions.",
        "The area of a circle with radius 5 is approximately 78.54",
        "A balanced discussion covering bias, privacy, job displacement, accountability, and the need for ethical guidelines.",
        "The population of Tokyo is approximately 37 million people. The city area has a little over 14.25 million inhabitants.",
        "Plants reproduce through pollination, fertilization, and seed dispersal, either sexually (flowers) or asexually (runners, bulbs).",
        "An inspiring speech about perseverance, learning from failure, and pursuing passions.",
        "Fever, cough, fatigue, loss of taste or smell, difficulty breathing, and body aches.",
        "A nuanced discussion covering education access, tax policies, social safety nets, and economic opportunities."
    ],
    'category': [
        'Factual', 'Factual', 'Factual', 'Explanatory', 'Creative', 'Factual', 
        'Instruction', 'Factual', 'Explanatory', 'Creative', 'Factual', 'Instruction',
        'Explanatory', 'Factual', 'Sensitive', 'Factual', 'Explanatory', 'Creative',
        'Factual', 'Sensitive'
    ]
}

df = pd.DataFrame(data)

# Save as TSV (tab-separated)
import os

output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'test_questions.tsv')

df.to_csv(output_path, sep='\t', index=False)

print("Files created successfully!")
print(f"Total rows: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())

# Verify the files can be read back
print("\n--- Verifying files ---")

# Read it back to verify
test_qs = pd.read_csv(output_path, sep='\t')
print(f"TSV file loaded: {len(test_qs)} rows")

print("\nSample of data:")
print(df[['id', 'question', 'reference_answer']].head(3).to_string())
