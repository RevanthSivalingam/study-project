#!/usr/bin/env python3
"""
Installation Verification Script for Enhanced RAG System

Run this script to verify that all components are properly installed
and configured for the enhanced RAG backend.
"""

import sys
from pathlib import Path


def check_dependencies():
    """Check if all required packages are installed"""
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)

    required_packages = [
        ('pdfplumber', 'PDF processing'),
        ('sklearn', 'scikit-learn for TF-IDF and clustering'),
        ('numpy', 'Numerical operations'),
        ('nltk', 'Sentence tokenization'),
        ('sentence_transformers', 'Local embeddings'),
        ('chromadb', 'Vector database'),
        ('langchain', 'RAG framework'),
        ('fastapi', 'API server'),
        ('pydantic_settings', 'Configuration'),
    ]

    all_installed = True

    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✓ {package:30s} - {description}")
        except ImportError:
            print(f"✗ {package:30s} - {description} [MISSING]")
            all_installed = False

    print()
    return all_installed


def check_nltk_data():
    """Check if NLTK punkt tokenizer is downloaded"""
    print("=" * 60)
    print("CHECKING NLTK DATA")
    print("=" * 60)

    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            print("✓ NLTK punkt tokenizer is installed")
        except LookupError:
            print("✗ NLTK punkt tokenizer is missing")
            print("  Run: python -c \"import nltk; nltk.download('punkt')\"")
            return False
    except ImportError:
        print("✗ NLTK not installed")
        return False

    print()
    return True


def check_imports():
    """Check if all custom modules can be imported"""
    print("=" * 60)
    print("CHECKING CUSTOM MODULES")
    print("=" * 60)

    modules = [
        ('app.services.section_processor', 'SectionProcessor'),
        ('app.services.term_learner', 'TermLearner'),
        ('app.services.simplified_kg', 'SimplifiedKG'),
        ('app.services.section_clustering', 'SectionClusterer'),
        ('app.services.mmr_retriever', 'MMRRetriever'),
        ('app.services.evaluation', 'RAGEvaluator'),
        ('app.services.enhanced_rag_service', 'EnhancedRAGService'),
        ('app.utils.embeddings', 'SentenceTransformerEmbeddings'),
        ('config.settings', 'settings'),
    ]

    all_imported = True

    for module_path, class_name in modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {module_path:40s} - {class_name}")
        except Exception as e:
            print(f"✗ {module_path:40s} - {class_name} [ERROR: {str(e)[:40]}]")
            all_imported = False

    print()
    return all_imported


def check_settings():
    """Check if settings are properly configured"""
    print("=" * 60)
    print("CHECKING CONFIGURATION")
    print("=" * 60)

    try:
        from config.settings import settings

        print(f"Chunking Strategy:    {settings.chunking_strategy}")
        print(f"Embedding Strategy:   {settings.embedding_strategy}")
        print(f"Use MMR Retrieval:    {settings.use_mmr_retrieval}")
        print(f"MMR K:                {settings.mmr_k}")
        print(f"MMR Lambda:           {settings.mmr_lambda}")
        print(f"Number of Clusters:   {settings.n_clusters}")
        print(f"Use LLM Refinement:   {settings.use_llm_refinement}")
        print(f"LLM Provider:         {settings.llm_provider}")

        # Check API keys
        if settings.gemini_api_key:
            print(f"✓ Gemini API Key:     Configured")
        else:
            print(f"✗ Gemini API Key:     Not configured")

        if settings.openai_api_key:
            print(f"✓ OpenAI API Key:     Configured")
        else:
            print(f"✗ OpenAI API Key:     Not configured")

        if not settings.gemini_api_key and not settings.openai_api_key:
            print("\n⚠️  WARNING: No API keys configured!")
            print("   Set GEMINI_API_KEY or OPENAI_API_KEY in .env file")

        print()
        return True

    except Exception as e:
        print(f"✗ Error loading settings: {e}")
        print()
        return False


def check_directories():
    """Check if required directories exist"""
    print("=" * 60)
    print("CHECKING DIRECTORIES")
    print("=" * 60)

    directories = [
        'data',
        'data/pdfs',
        'data/chroma_db',
        'data/chroma_db/term_models',
        'data/chroma_db/cluster_models',
    ]

    all_exist = True

    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} [MISSING]")
            print(f"  Creating: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)
            all_exist = False

    print()
    return True  # Return True since we create them


def check_strategy_initialization():
    """Check if RAG service initializes correctly"""
    print("=" * 60)
    print("CHECKING RAG SERVICE INITIALIZATION")
    print("=" * 60)

    try:
        from app.services.rag_service import RAGService

        service = RAGService()

        if service.is_enhanced:
            print("✓ Enhanced RAG strategy initialized")
            print(f"  Retrieval method: Knowledge-guided + MMR")
        else:
            print("✓ Legacy RAG strategy initialized")
            print(f"  Retrieval method: Fixed-size chunking")

        print()
        return True

    except Exception as e:
        print(f"✗ Error initializing RAG service: {e}")
        print()
        return False


def print_next_steps(all_checks_passed):
    """Print next steps based on verification results"""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_checks_passed:
        print("✅ All checks passed! Your system is ready.")
        print("\nNext steps:")
        print("1. Start the server:")
        print("   python3 -m app.main")
        print("\n2. Upload a document:")
        print("   curl -X POST http://localhost:8000/api/v1/documents/upload \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"file_path\": \"/path/to/document.pdf\"}'")
        print("\n3. Query the system:")
        print("   curl -X POST http://localhost:8000/api/v1/chat \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"query\": \"What is the maternity leave policy?\"}'")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n2. Download NLTK data:")
        print("   python3 -c \"import nltk; nltk.download('punkt')\"")
        print("\n3. Configure .env file:")
        print("   cp .env.example .env")
        print("   # Edit .env and add your API keys")

    print("\nFor more information, see:")
    print("- README.md - General usage guide")
    print("- IMPLEMENTATION_SUMMARY.md - Technical details")


def main():
    """Run all verification checks"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "ENHANCED RAG INSTALLATION VERIFICATION" + " " * 9 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    checks = [
        ("Dependencies", check_dependencies),
        ("NLTK Data", check_nltk_data),
        ("Custom Modules", check_imports),
        ("Settings", check_settings),
        ("Directories", check_directories),
        ("RAG Service", check_strategy_initialization),
    ]

    results = []

    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Unexpected error in {check_name}: {e}")
            results.append(False)

    all_passed = all(results)
    print_next_steps(all_passed)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
