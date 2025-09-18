#!/usr/bin/env python3
"""
NLTK Setup Script for AI Ticket Classification System
Run this script first to download all required NLTK data
"""

import nltk
import ssl
import os
from pathlib import Path

def setup_nltk():
    """Setup NLTK with all required packages"""
    
    print("🚀 Setting up NLTK for AI Ticket Classification System")
    print("=" * 60)
    
    # Handle SSL certificate issues (common on Windows/Mac)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # List of required NLTK packages
    nltk_packages = [
        ('punkt_tab', 'Modern sentence tokenizer'),
        ('punkt', 'Sentence tokenizer (fallback)'),
        ('stopwords', 'Stop words for text filtering'),
        ('wordnet', 'WordNet lexical database'),
        ('omw-1.4', 'Open Multilingual Wordnet'),
        ('averaged_perceptron_tagger', 'Part-of-speech tagger'),
        ('vader_lexicon', 'Sentiment analysis lexicon')
    ]
    
    successful_downloads = 0
    failed_downloads = []
    
    print("\n📥 Downloading NLTK packages...")
    print("-" * 40)
    
    for package, description in nltk_packages:
        try:
            print(f"⏳ Downloading {package}... ", end="", flush=True)
            nltk.download(package, quiet=True)
            print("✅ Success")
            successful_downloads += 1
        except Exception as e:
            print(f"❌ Failed: {str(e)[:50]}...")
            failed_downloads.append((package, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Download Summary:")
    print(f"   ✅ Successful: {successful_downloads}/{len(nltk_packages)}")
    print(f"   ❌ Failed: {len(failed_downloads)}")
    
    if failed_downloads:
        print(f"\n⚠️  Failed Downloads:")
        for package, error in failed_downloads:
            print(f"   - {package}: {error[:80]}...")
    
    # Test installations
    print("\n🧪 Testing NLTK installations...")
    print("-" * 40)
    
    # Test tokenization
    try:
        from nltk.tokenize import word_tokenize
        test_text = "This is a test sentence for tokenization."
        tokens = word_tokenize(test_text)
        print(f"✅ Tokenization: {len(tokens)} tokens from test sentence")
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
    
    # Test stopwords
    try:
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        print(f"✅ Stopwords: {len(stop_words)} English stopwords loaded")
    except Exception as e:
        print(f"❌ Stopwords failed: {e}")
    
    # Test lemmatization
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        test_word = lemmatizer.lemmatize("running", pos='v')
        print(f"✅ Lemmatization: 'running' → '{test_word}'")
    except Exception as e:
        print(f"❌ Lemmatization failed: {e}")
    
    print("\n" + "=" * 60)
    
    if len(failed_downloads) == 0:
        print("🎉 NLTK setup completed successfully!")
        print("✅ You can now run the AI Ticket Classification System")
        print("\n💡 Next steps:")
        print("   1. Run: python app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Start classifying tickets!")
    else:
        print("⚠️  NLTK setup completed with some issues")
        print("🔧 The system will use fallback methods for failed components")
        print("💡 You can still run the system, but some features may be limited")
    
    return len(failed_downloads) == 0

def create_nltk_data_dir():
    """Create NLTK data directory if it doesn't exist"""
    try:
        nltk_data_dir = Path.home() / 'nltk_data'
        nltk_data_dir.mkdir(exist_ok=True)
        print(f"📁 NLTK data directory: {nltk_data_dir}")
        return str(nltk_data_dir)
    except Exception as e:
        print(f"⚠️ Could not create NLTK data directory: {e}")
        return None

def manual_download_instructions():
    """Provide manual download instructions if automated download fails"""
    print("\n🔧 Manual Download Instructions:")
    print("-" * 40)
    print("If automated download failed, try these steps:")
    print("")
    print("1. Open Python interpreter:")
    print("   python")
    print("")
    print("2. Run these commands:")
    print("   import nltk")
    print("   nltk.download('all')  # Downloads everything")
    print("   # OR download specific packages:")
    print("   nltk.download('punkt_tab')")
    print("   nltk.download('punkt')")
    print("   nltk.download('stopwords')")
    print("   nltk.download('wordnet')")
    print("")
    print("3. Alternative - GUI downloader:")
    print("   import nltk")
    print("   nltk.download()  # Opens GUI downloader")
    print("")

if __name__ == "__main__":
    # Create NLTK data directory
    create_nltk_data_dir()
    
    # Setup NLTK
    success = setup_nltk()
    
    if not success:
        manual_download_instructions()
    
    print("\n🔍 System Information:")
    print(f"   Python version: {nltk.sys.version}")
    print(f"   NLTK version: {nltk.__version__}")
    print(f"   NLTK data path: {nltk.data.path}")
    
    print("\n" + "=" * 60)
    print("Setup script completed! 🚀")