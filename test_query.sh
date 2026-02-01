#!/bin/bash
# Quick test script for querying the Enhanced RAG system

if [ -z "$1" ]; then
    echo "Usage: ./test_query.sh \"Your question here\""
    echo ""
    echo "Examples:"
    echo "  ./test_query.sh \"What is the maternity leave policy?\""
    echo "  ./test_query.sh \"How many days of sick leave?\""
    echo "  ./test_query.sh \"What are the vacation benefits?\""
    exit 1
fi

QUERY="$1"

echo "📋 Querying Enhanced RAG system..."
echo "Question: $QUERY"
echo ""

curl -s -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\"}" | python3 -c "
import sys, json

try:
    data = json.load(sys.stdin)

    print('═' * 70)
    print('ANSWER:')
    print('═' * 70)
    print(data['answer'])
    print()

    print('═' * 70)
    print('METADATA:')
    print('═' * 70)
    print(f\"Retrieval Method: {data.get('retrieval_method', 'N/A')}\")
    print(f\"Confidence Score: {data.get('confidence_score', 'N/A')}\")
    print(f\"MMR Sentences: {data.get('mmr_sentences_used', 'N/A')}\")
    print(f\"Section Title: {data.get('section_title', 'N/A')}\")

    if data.get('precision_at_k'):
        print(f\"Precision@k: {data['precision_at_k']:.2f}\")
    if data.get('recall_at_k'):
        print(f\"Recall@k: {data['recall_at_k']:.2f}\")
    if data.get('mrr'):
        print(f\"MRR: {data['mrr']:.2f}\")

    if data.get('sources'):
        print()
        print('═' * 70)
        print('SOURCES:')
        print('═' * 70)
        for source in data['sources']:
            print(f\"Document: {source['document_name']}\")
            print(f\"Page: {source.get('page_number', 'N/A')}\")
            print(f\"Relevance: {source['relevance_score']:.2f}\")

    print('═' * 70)

except Exception as e:
    print(f'Error: {e}')
    print(sys.stdin.read())
"
