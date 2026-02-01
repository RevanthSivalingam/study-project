"""
Setup script for RAG Chatbot
Downloads required NLTK data
"""

import nltk
import os

def setup_nltk():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        nltk.download('punkt', quiet=False)
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("📁 Creating directories...")
    os.makedirs("inputfiles", exist_ok=True)
    print("✅ Directories created!")

def main():
    print("=" * 50)
    print("  RAG Chatbot Setup")
    print("=" * 50)
    print()

    create_directories()
    setup_nltk()

    print()
    print("=" * 50)
    print("Setup complete! 🎉")
    print()
    print("Next steps:")
    print("1. Place your PDF/TXT/JSON files in the 'inputfiles' folder")
    print("2. Run: ./start_ui.sh")
    print("3. Open http://localhost:8501 in your browser")
    print("=" * 50)

if __name__ == "__main__":
    main()
