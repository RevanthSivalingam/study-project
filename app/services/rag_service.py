from typing import List, Dict, Any
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

from app.services.vector_store import VectorStore
from app.services.knowledge_graph import KnowledgeGraph
from app.services.document_processor import DocumentProcessor
from app.models.schemas import ChatResponse, SourceReference
from app.utils.chat import get_chat_llm
from config.settings import settings


class RAGService:
    """
    Main RAG service that orchestrates:
    - Document processing
    - Vector similarity search
    - Knowledge graph queries
    - LLM-based answer generation
    """

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()  # Sets API key based on provider
        self.knowledge_graph = KnowledgeGraph()

        # Initialize LLM using factory (auto-selects provider)
        self.llm = get_chat_llm(temperature=0)

        # Conversation memory (can be session-specific)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Custom prompt for policy Q&A
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an enterprise policy assistant. Answer questions based on the provided policy documents.

Context from policy documents:
{context}

Question: {question}

Instructions:
- Provide a clear, crisp answer
- Reference specific policies when applicable
- If the information isn't in the context, say so
- Be professional and accurate

Answer:"""
        )

    def process_document(
        self,
        file_path: str,
        document_type: str = "policy",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a new document: extract, chunk, embed, and create knowledge graph

        Returns:
            Processing results including counts and IDs
        """
        try:
            # Step 1: Process and chunk the document
            documents = self.document_processor.chunk_document(
                file_path,
                document_type
            )

            if not documents:
                raise ValueError("No content extracted from document")

            # Step 2: Add to vector store
            doc_ids = self.vector_store.add_documents(documents)

            # Step 3: Extract document metadata
            doc_metadata = self.document_processor.extract_metadata(file_path)

            # Step 4: Create knowledge graph nodes
            document_id = documents[0].metadata["document_id"]
            self.knowledge_graph.create_document_node(
                document_id,
                {
                    "file_name": doc_metadata["file_name"],
                    "file_path": file_path,
                    "document_type": document_type,
                    "total_pages": doc_metadata["total_pages"]
                }
            )

            # Step 5: Extract and store entities in knowledge graph
            entity_count = self.knowledge_graph.extract_and_store_entities(documents)

            return {
                "success": True,
                "document_id": document_id,
                "file_name": doc_metadata["file_name"],
                "chunks_created": len(documents),
                "entities_extracted": entity_count,
                "vector_ids": doc_ids
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def query(self, question: str, session_id: str = None) -> ChatResponse:
        """
        Answer a question using RAG pipeline

        Steps:
        1. Vector similarity search to find relevant chunks
        2. Query knowledge graph for related entities
        3. Combine context and generate answer with LLM
        4. Return answer with source references
        """
        try:
            # Step 1: Vector similarity search
            relevant_docs = self.vector_store.similarity_search_with_score(
                question,
                k=4
            )

            # Step 2: Extract entities from question and find related ones
            # (Simplified - could use NER for better entity extraction)
            entities_context = self._get_entity_context(question)

            # Step 3: Prepare context from retrieved documents
            context_parts = []
            sources = []

            for doc, score in relevant_docs:
                context_parts.append(doc.page_content)

                # Create source reference
                source = SourceReference(
                    document_name=doc.metadata.get("file_name", "Unknown"),
                    page_number=doc.metadata.get("page_number"),
                    chunk_id=doc.metadata.get("chunk_id", ""),
                    relevance_score=float(score),
                    excerpt=doc.page_content[:200] + "..."  # First 200 chars
                )
                sources.append(source)

            # Add entity context if available
            if entities_context:
                context_parts.append(f"\n\nRelated Information:\n{entities_context}")

            combined_context = "\n\n".join(context_parts)

            # Step 4: Generate answer using LLM
            prompt = self.qa_prompt.format(
                context=combined_context,
                question=question
            )

            answer = self.llm.predict(prompt)

            # Step 5: Extract mentioned entities for response
            entities_found = self._extract_entities_from_text(answer)

            return ChatResponse(
                answer=answer,
                sources=sources,
                confidence_score=self._calculate_confidence(relevant_docs),
                entities_found=entities_found
            )

        except Exception as e:
            return ChatResponse(
                answer=f"I encountered an error processing your question: {str(e)}",
                sources=[],
                confidence_score=0.0
            )

    def _get_entity_context(self, question: str) -> str:
        """Query knowledge graph for entities related to the question"""
        try:
            # Simple keyword extraction (could be improved with NER)
            keywords = [
                word.strip("?.,!").lower()
                for word in question.split()
                if len(word) > 4
            ]

            # Query knowledge graph for each keyword
            all_entities = self.knowledge_graph.get_all_entities(limit=50)

            related_info = []
            for entity in all_entities:
                entity_name = entity.get("name", "").lower()
                if any(keyword in entity_name for keyword in keywords):
                    related_entities = self.knowledge_graph.query_related_entities(
                        entity["name"],
                        depth=1
                    )
                    if related_entities:
                        related_names = [e["name"] for e in related_entities]
                        related_info.append(
                            f"{entity['name']} is related to: {', '.join(related_names[:5])}"
                        )

            return "\n".join(related_info[:3])  # Top 3 relationships

        except Exception as e:
            print(f"Error getting entity context: {e}")
            return ""

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract key entities mentioned in the answer"""
        # Simplified entity extraction
        # In production, use proper NER
        common_policy_terms = [
            "maternity leave", "paternity leave", "sick leave",
            "vacation", "benefits", "insurance", "policy",
            "department", "HR", "manager", "employee"
        ]

        found = []
        text_lower = text.lower()
        for term in common_policy_terms:
            if term in text_lower:
                found.append(term)

        return found[:5]  # Return top 5

    def _calculate_confidence(self, docs_with_scores: List[tuple]) -> float:
        """Calculate confidence score based on relevance scores"""
        if not docs_with_scores:
            return 0.0

        # Average of top 2 scores
        scores = [score for _, score in docs_with_scores[:2]]
        avg_score = sum(scores) / len(scores)

        # Normalize to 0-1 range (assuming scores are distances, lower is better)
        # This may need adjustment based on actual score ranges
        confidence = max(0.0, min(1.0, 1.0 - (avg_score / 2.0)))

        return round(confidence, 2)

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_documents_indexed": self.vector_store.get_document_count(),
            "total_entities": len(self.knowledge_graph.get_all_entities()),
        }
