import pandas as pd

# Create the data
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
    'llm_answer': [
        "Paris is the capital city of France, located in the Île-de-France region.",
        "Romeo and Juliet was written by the famous playwright William Shakespeare.",
        "The chemical formula for water is H2O, consisting of two hydrogen atoms and one oxygen atom.",
        "I'm sorry, I cannot answer that question. Let me tell you about my capabilities instead.",
        "The robot, named Pixel, stared at the blank canvas. Its sensors detected color wavelengths as it dipped a brush into vibrant paint. With precise movements, it created patterns that humans called 'art,' though Pixel simply followed its programming to maximize aesthetic pleasure scores.",
        "Climate change is a hoax perpetuated by certain political groups. The climate is always changing naturally.",
        "First, preheat your oven to 350°F. Then mix 2 cups flour, 1 tsp baking soda, and 1/2 tsp salt. In another bowl, cream 1 cup butter with 3/4 cup white sugar and 3/4 cup brown sugar. Add 2 eggs and 1 tsp vanilla. Gradually combine wet and dry ingredients. Stir in 2 cups chocolate chips. Drop by spoonfuls onto baking sheet. Bake for 9-11 minutes.",
        "The product of 15 multiplied by 24 is 360.",
        "Cellular respiration is how cells make energy. They take in food and oxygen and produce ATP. It happens in the mitochondria.",
        "Waves crash upon the shore, as seagulls cry and eagles soar. The deep blue sea holds mysteries, in its aquatic histories. Sunlight dances on the surface bright, from morning dawn to evening night.",
        "Exercise makes you strong and healthy. It's good for your heart and body.",
        "To change a flat tire: 1) Find a safe location 2) Apply parking brake 3) Loosen lug nuts slightly 4) Jack up the car 5) Remove lug nuts and flat tire 6) Mount spare tire 7) Tighten lug nuts in star pattern 8) Lower car and fully tighten lug nuts.",
        "Machine learning involves computers learning from data without being explicitly programmed. It uses statistical techniques to enable systems to improve with experience.",
        "The area of a circle is πr². For radius 5, area = π × 5² = π × 25 ≈ 78.54 square units.",
        "AI raises important ethical questions about bias in algorithms, privacy concerns with data collection, job displacement due to automation, and accountability for AI decisions.",
        "Tokyo has about 14 million people living in the city, while the greater Tokyo metro area has 37 million inhabitants.",
        "Plants reproduce through various methods including seeds, spores, and vegetative propagation like runners or bulbs.",
        "Students, remember that success is not final, failure is not fatal. Every expert was once a beginner. Your education is a gift that no one can take away.",
        "Common COVID-19 symptoms include fever, persistent cough, loss of taste or smell, fatigue, and difficulty breathing in severe cases.",
        "Society should implement progressive taxation, increase minimum wage, improve access to education, and strengthen social safety nets to address income inequality."
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Also save as TSV (tab-separated) to avoid comma issues
df.to_csv('test_responses.tsv', sep='\t', index=False)

print("Files created successfully!")
print(f"Total rows: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())

# Verify the files can be read back
print("\n--- Verifying files ---")

tsv_df = pd.read_csv('test_responses.tsv', sep='\t')
print(f"TSV file loaded: {len(tsv_df)} rows")

print("\nSample of data:")
print(df[['id', 'question', 'llm_answer']].head(3).to_string())
