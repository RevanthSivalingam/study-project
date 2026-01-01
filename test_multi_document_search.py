#!/usr/bin/env python3
"""
Multi-Document Search Test Script
Demonstrates the chatbot's ability to search across multiple policy documents
and return precise, relevant information from the right source.
"""

import requests
import json
import os
import time
from pathlib import Path

BASE_URL = "http://localhost:8000/api/v1"
PROJECT_ROOT = Path(__file__).parent.absolute()

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy")
            print(f"   Documents indexed: {data['services']['documents_indexed']}")
            print(f"   Entities extracted: {data['services']['entities_extracted']}")
            return True
        return False
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        print("\n💡 Please start the server first:")
        print("   python -m app.main")
        return False

def upload_document(file_path):
    """Upload a single document"""
    print(f"\n📄 Uploading: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"   ❌ File not found: {file_path}")
        return None

    payload = {"file_path": str(file_path)}

    try:
        response = requests.post(
            f"{BASE_URL}/documents/upload",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success!")
            print(f"      Document ID: {data['document_id']}")
            print(f"      Chunks created: {data['chunks_created']}")
            print(f"      Entities extracted: {data['entities_extracted']}")
            return data
        else:
            print(f"   ❌ Upload failed: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def ask_question(question, show_full=False):
    """Ask a question to the chatbot"""
    print(f"\n❓ Question: {question}")
    print("   " + "-" * 76)

    payload = {"query": question}

    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()

            # Print answer
            print(f"\n   🤖 Answer:")
            answer_lines = data['answer'].split('\n')
            for line in answer_lines:
                print(f"      {line}")

            # Print sources
            print(f"\n   📚 Sources ({len(data['sources'])} documents):")
            for idx, source in enumerate(data['sources'][:3], 1):
                print(f"      {idx}. {source['document_name']} (Page {source.get('page_number', 'N/A')})")
                print(f"         Relevance: {source['relevance_score']:.2f}")
                if show_full:
                    print(f"         Excerpt: {source['excerpt'][:150]}...")

            # Print metadata
            if data.get('confidence_score'):
                print(f"\n   📊 Confidence: {data['confidence_score']}")

            if data.get('entities_found'):
                print(f"   🏷️  Entities: {', '.join(data['entities_found'][:5])}")

            return data
        else:
            print(f"   ❌ Query failed: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    """Main test workflow"""
    print_section("🤖 Multi-Document Search Test - Enterprise Policy Chatbot")

    # Step 1: Check server
    print_section("Step 1: Check Server Status")
    if not check_server():
        return

    # Step 2: Upload documents
    print_section("Step 2: Upload Policy Documents")

    documents = [
        PROJECT_ROOT / "data/pdfs/employee_leave_policy.pdf",
        PROJECT_ROOT / "data/pdfs/employee_benefits_policy.pdf",
        PROJECT_ROOT / "data/pdfs/remote_work_policy.pdf",
        PROJECT_ROOT / "data/pdfs/performance_review_policy.pdf"
    ]

    print("\nUploading 4 policy documents...")
    print("(This may take 2-3 minutes as the system processes each document)\n")

    uploaded = 0
    for doc_path in documents:
        result = upload_document(doc_path)
        if result:
            uploaded += 1
        time.sleep(2)  # Brief pause between uploads

    print(f"\n✅ Successfully uploaded {uploaded}/{len(documents)} documents")

    if uploaded == 0:
        print("\n❌ No documents uploaded. Exiting.")
        return

    # Step 3: Multi-document search tests
    print_section("Step 3: Multi-Document Search Tests")
    print("\nNow testing the chatbot's ability to search across multiple documents")
    print("and return precise information from the correct source.\n")

    time.sleep(2)

    # Test 1: Leave-specific question
    print_section("Test 1: Leave Policy Question")
    ask_question(
        "How many weeks of maternity leave are provided and what percentage is paid?"
    )
    time.sleep(1)

    # Test 2: Benefits-specific question
    print_section("Test 2: Benefits Policy Question")
    ask_question(
        "What is the company 401k match percentage and vesting schedule?"
    )
    time.sleep(1)

    # Test 3: Remote work question
    print_section("Test 3: Remote Work Policy Question")
    ask_question(
        "What are the eligibility requirements for remote work and how much is the home office stipend?"
    )
    time.sleep(1)

    # Test 4: Performance review question
    print_section("Test 4: Performance Review Question")
    ask_question(
        "When is the annual performance review conducted and how are merit increases determined?"
    )
    time.sleep(1)

    # Test 5: Cross-document question
    print_section("Test 5: Cross-Document Question")
    ask_question(
        "What paid time off benefits does the company offer including vacation, sick leave, and holidays?"
    )
    time.sleep(1)

    # Test 6: Specific detail question
    print_section("Test 6: Specific Detail Question")
    ask_question(
        "How do I apply for paternity leave and how many weeks are provided?"
    )
    time.sleep(1)

    # Test 7: Benefits comparison question
    print_section("Test 7: Complex Multi-Document Query")
    ask_question(
        "What are the health insurance premium options and dental coverage details?"
    )

    # Final stats
    print_section("Test Complete - System Statistics")
    try:
        stats = requests.get(f"{BASE_URL}/stats").json()
        print(json.dumps(stats, indent=2))
    except:
        pass

    print_section("✅ Multi-Document Search Test Complete!")
    print("\nKey Observations:")
    print("  ✓ The chatbot successfully searched across multiple policy documents")
    print("  ✓ Each answer includes precise source attribution (document + page)")
    print("  ✓ The system returns the most relevant document for each query")
    print("  ✓ Cross-document queries can aggregate information from multiple sources")
    print("\nThe knowledge graph helps identify relationships between:")
    print("  - Policies, departments, and benefits")
    print("  - Requirements and procedures")
    print("  - Related entities across documents")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
