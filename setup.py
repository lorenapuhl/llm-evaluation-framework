"""
Setup script for LLM Evaluation Framework
"""

import subprocess
import sys
import os
import shutil
import ssl

def run_command(command):
    """Run a shell command and check for errors."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def setup_project():
    """Setup the entire project with required dependencies."""
    print("=" * 60)
    print("LLM Evaluation Framework - Setup")
    print("=" * 60)
    
    # Step 1: Install requirements
    print("\n1. Installing Python packages...")
    success = run_command(f"{sys.executable} -m pip install -r requirements.txt ")
    if not success:
        print("Failed to install requirements. Trying with --user flag...")
        run_command(f"{sys.executable} -m pip install --user -r requirements.txt --no-cache-dir")
    
    # Step 2: Download NLTK data
    print("\n2. Downloading NLTK data...")
    
   # Remove possibly existing NLTK data 
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    if os.path.exists(nltk_data_dir):
        print(f"Removing NLTK data from {nltk_data_dir}...")
        shutil.rmtree(nltk_data_dir)
        
    # Bypass SSL verification if needed
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
        
    try:
        import nltk
        print("Downloading NLTK datasets...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('wordnet')
        nltk.download('omw-eng')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
        nltk.download('wordnet_ic')
        nltk.download('sentiwordnet')
        
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        
    # MANUALLY initialize WordNet to avoid lazy loading issues
    from nltk.corpus import wordnet
    try:
        # Force load by accessing it
        _ = wordnet.synsets('test')[0]
        print("✓ WordNet initialized successfully")
    except:
        print("✗ WordNet initialization failed, using fallback")
    
    # Step 3: Test imports
    print("\n3. Testing imports...")
    test_imports = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('sentence_transformers', 'SentenceTransformer'),
        ('nltk', 'nltk'),
    ]
    
    all_ok = True
    for module, alias in test_imports:
        try:
            exec(f"import {module} as {alias}")
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            all_ok = False
    
    # Step 4: Download sentence transformer model (optional, happens on first use)
    print("\n4. Testing sentence transformer...")
    try:
        from sentence_transformers import SentenceTransformer
        # This will download the model on first use
        print("  ✓ Sentence transformers ready (will download model on first use)")
    except Exception as e:
        print(f"  ✗ Sentence transformers: {e}")
        print("  Note: Model will be downloaded automatically on first use.")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ Setup completed successfully!")
        print("\nTo test the evaluation framework:")
        print("  python benchmarks/test_evaluation.py")
    else:
        print("✗ Setup completed with some errors.")
        print("Please check the error messages above.")
    print("=" * 60)

if __name__ == "__main__":
    setup_project()
