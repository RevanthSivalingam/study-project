"""
Example usage script for the Enterprise Policy Chatbot

This script demonstrates how to:
1. Upload a policy document
2. Query the chatbot
3. View the response with sources

Requirements:
- Server must be running (python -m app.main)
- PDF files in data/pdfs/ directory
"""

import requests
import json
import os

# API base URL
BASE_URL = "http://localhost:8000/api/v1"


def health_check():
    """Check if the API is running"""
    print("🔍 Checking API health...")
    response = requests.get(f"{BASE_URL}/health")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ API is healthy")
        print(f"   App: {data['app_name']}")
        print(f"   Version: {data['version']}")
        print(f"   Documents indexed: {data['services']['documents_indexed']}")
        print(f"   Entities extracted: {data['services']['entities_extracted']}")
        return True
    else:
        print(f"❌ API health check failed")
        return False


def upload_document(file_path):
    """Upload a policy document"""
    print(f"\n📄 Uploading document: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    payload = {
        "file_path": file_path,
        "document_type": "policy"
    }

    response = requests.post(
        f"{BASE_URL}/documents/upload",
        json=payload
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Document uploaded successfully!")
        print(f"   Document ID: {data['document_id']}")
        print(f"   File: {data['file_name']}")
        print(f"   Chunks created: {data['chunks_created']}")
        print(f"   Entities extracted: {data['entities_extracted']}")
        return data
    else:
        print(f"❌ Upload failed: {response.text}")
        return None


def chat_query(question):
    """Ask a question to the chatbot"""
    print(f"\n💬 Question: {question}")

    payload = {
        "query": question
    }

    response = requests.post(
        f"{BASE_URL}/chat",
        json=payload
    )

    if response.status_code == 200:
        data = response.json()
        print(f"\n🤖 Answer:")
        print(f"   {data['answer']}")
        print(f"\n📚 Sources ({len(data['sources'])}):")

        for idx, source in enumerate(data['sources'], 1):
            print(f"   {idx}. {source['document_name']} (Page {source['page_number']})")
            print(f"      Relevance: {source['relevance_score']:.2f}")
            print(f"      Excerpt: {source['excerpt'][:100]}...")

        if data.get('entities_found'):
            print(f"\n🏷️  Entities found: {', '.join(data['entities_found'])}")

        print(f"\n📊 Confidence: {data.get('confidence_score', 'N/A')}")

        return data
    else:
        print(f"❌ Query failed: {response.text}")
        return None


def get_stats():
    """Get system statistics"""
    print("\n📊 System Statistics:")
    response = requests.get(f"{BASE_URL}/stats")

    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data['data'], indent=2))
    else:
        print(f"❌ Failed to get stats")


def main():
    """Main example workflow"""
    print("=" * 60)
    print("Enterprise Policy Chatbot - Example Usage")
    print("=" * 60)

    # Step 1: Health check
    if not health_check():
        print("\n⚠️  Please start the server first:")
        print("   python -m app.main")
        return

    # Step 2: Upload a document (update this path to your actual PDF)
    pdf_path = os.path.abspath("data/pdfs/sample_policy.pdf")

    print(f"\n💡 To upload a document, place your PDF in: data/pdfs/")
    print(f"   Then update the 'pdf_path' variable in this script")

    if os.path.exists(pdf_path):
        upload_document(pdf_path)
    else:
        print(f"\n⚠️  Sample PDF not found at: {pdf_path}")
        print("   Skipping upload demo. Using existing documents if any.")

    # Step 3: Query examples
    example_queries = [
        "What is the maternity leave policy?",
        "How many vacation days do employees get?",
        "What are the requirements for sick leave?",
        "How do I apply for paternity leave?"
    ]

    print("\n" + "=" * 60)
    print("Example Queries")
    print("=" * 60)

    for query in example_queries[:2]:  # Demo with first 2 queries
        chat_query(query)
        print("\n" + "-" * 60)

    # Step 4: Show stats
    get_stats()

    print("\n" + "=" * 60)
    print("✅ Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
