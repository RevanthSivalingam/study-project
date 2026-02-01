"""
Enhanced RAG Service with knowledge-guided two-stage retrieval.

Implements sophisticated retrieval pipeline:
1. Section-based chunking (header detection)
2. TF-IDF term learning for query normalization
3. Simplified KG for term→section mapping
4. Two-stage retrieval: KG-guided → semantic fallback
5. MMR sentence selection
6. Optional LLM refinement
"""
import hashlib
import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from app.services.section_processor import Section, SectionProcessor
from app.services.term_learner import TermLearner
from app.services.simplified_kg import SimplifiedKG
from app.services.section_clustering import SectionClusterer
from app.services.mmr_retriever import MMRRetriever, tokenize_sentences
from app.services.evaluation import RAGEvaluator
from app.services.vector_store import VectorStore
from app.models.schemas import ChatResponse, SourceReference
from app.utils.embeddings import get_embeddings
from app.utils.chat import get_chat_llm
from config.settings import settings


class EnhancedRAGService:
    """
    Enhanced RAG with knowledge-guided retrieval and MMR sentence selection.

    Pipeline:
    - Document Processing: Section extraction → TF-IDF learning → KG building → Clustering
    - Query Processing: Query normalization → KG retrieval → Fallback clustering → MMR selection
    """

    def __init__(self):
        """Initialize enhanced RAG components."""
        self.section_processor = SectionProcessor()
        self.term_learner = TermLearner(settings.chroma_persist_directory)
        self.simplified_kg = SimplifiedKG()
        self.section_clusterer = SectionClusterer(
            n_clusters=settings.n_clusters,
            persist_directory=settings.chroma_persist_directory
        )
        self.embeddings = get_embeddings()
        self.mmr_retriever = MMRRetriever(self.embeddings)
        self.vector_store = VectorStore()
        self.evaluator = RAGEvaluator()

        # LLM for optional refinement
        self.llm = None
        if settings.use_llm_refinement:
            self.llm = get_chat_llm(temperature=0.2)

        # Load learned models if available
        self.term_learner.load()
        self.section_clusterer.load()

        # In-memory section storage (for fast retrieval)
        self.sections: List[Section] = []

    def process_document(
        self,
        file_path: str,
        document_type: str = "policy",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process document using section-based strategy.

        Steps:
        1. Extract sections (header-based)
        2. Learn TF-IDF terms
        3. Build simplified KG
        4. Cluster sections
        5. Store in vector database
        6. Persist learned models

        Args:
            file_path: Path to the document
            document_type: Type of document
            metadata: Additional metadata

        Returns:
            Processing results dictionary
        """
        try:
            # Generate document ID
            doc_id = self._generate_document_id(file_path)
            file_name = os.path.basename(file_path)

            # Step 1: Extract sections
            print(f"Extracting sections from {file_name}...")
            sections = self.section_processor.extract_sections_from_pdf(
                file_path,
                doc_id
            )

            if not sections:
                return {
                    "success": False,
                    "error": "No sections extracted from document"
                }

            print(f"Extracted {len(sections)} sections")

            # Step 2: Learn TF-IDF terms
            print("Learning generic and key terms...")
            self.term_learner.learn_generic_terms(sections)
            self.term_learner.learn_key_terms(sections, top_n=40)

            # Step 3: Build simplified KG
            print("Building knowledge graph...")
            self.simplified_kg.build_from_sections(
                sections,
                self.term_learner.key_terms
            )

            # Step 4: Create embeddings for sections
            print("Creating embeddings...")
            section_texts = [section.content for section in sections]
            embeddings = self.embeddings.embed_documents(section_texts)

            # Step 5: Cluster sections
            print("Clustering sections...")
            self.section_clusterer.fit(sections, embeddings)

            # Step 6: Store in ChromaDB
            print("Storing in vector database...")
            documents = self._sections_to_documents(sections)
            doc_ids = self.vector_store.add_documents(documents)

            # Step 7: Store sections in memory
            self.sections.extend(sections)

            # Step 8: Persist learned models
            print("Saving models...")
            self.term_learner.save()
            self.section_clusterer.save()

            return {
                "success": True,
                "document_id": doc_id,
                "file_name": file_name,
                "chunks_created": len(sections),  # Backward compatible field name
                "sections_created": len(sections),  # Also include new field name
                "entities_extracted": len(self.term_learner.key_terms),  # Backward compatible
                "key_terms_learned": len(self.term_learner.key_terms),
                "clusters_created": self.section_clusterer.n_clusters,
                "vector_ids": doc_ids,
                "generic_terms_count": len(self.term_learner.generic_terms)
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def query(
        self,
        question: str,
        session_id: str = None
    ) -> ChatResponse:
        """
        Answer question using enhanced RAG pipeline.

        Pipeline:
        1. Normalize query (remove generic terms)
        2. KG-guided retrieval
        3. Semantic fallback (clustering) if needed
        4. MMR sentence selection
        5. Optional LLM refinement
        6. Calculate evaluation metrics

        Args:
            question: User's question
            session_id: Optional session ID

        Returns:
            ChatResponse with answer and metadata
        """
        try:
            # ===== STAGE 1: Query Normalization =====
            normalized_query = self.term_learner.normalize_query(question)
            print(f"Normalized query: '{normalized_query}'")

            # ===== STAGE 2: KG-Guided Retrieval =====
            kg_candidates = self.simplified_kg.query(normalized_query)
            print(f"KG found {len(kg_candidates)} candidate sections")

            selected_section = None
            retrieval_method = None
            cluster_id = None

            if kg_candidates:
                # Group by section and select best
                section_map = self.simplified_kg.group_by_section(
                    self.sections,
                    kg_candidates
                )

                # Detect broad queries (many candidates = need overview)
                # Use threshold of 15 to avoid triggering on moderately specific queries
                is_broad_query = len(kg_candidates) > 15

                if is_broad_query:
                    print(f"Broad query detected ({len(kg_candidates)} candidates)")
                    # For broad queries, combine top sections using clustering
                    retrieval_method = "broad_query_multi_section"

                    # Get diverse sections from different clusters
                    query_embedding = self.embeddings.embed_query(normalized_query)

                    # Combine content from multiple top sections
                    top_sections = []
                    scored_sections = []

                    for title, content in section_map.items():
                        section_obj = self.simplified_kg.get_section_by_title(title)
                        if section_obj:
                            # Simple lexical score for ranking
                            query_terms = set(normalized_query.lower().split())
                            content_terms = set(content.lower().split())
                            score = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
                            scored_sections.append((section_obj, score))

                    # Sort by score and take top 3 diverse sections
                    scored_sections.sort(key=lambda x: x[1], reverse=True)
                    top_sections = [s[0] for s in scored_sections[:3]]

                    if top_sections:
                        # Combine content from top sections
                        combined_content = "\n\n".join([
                            f"**{section.title}**\n{section.content}"
                            for section in top_sections
                        ])

                        # Create a virtual combined section
                        from app.services.section_processor import Section
                        selected_section = Section(
                            title="Combined: " + ", ".join([s.title for s in top_sections[:2]]),
                            content=combined_content,
                            page_number=top_sections[0].page_number,
                            section_index=top_sections[0].section_index,
                            document_id=top_sections[0].document_id,
                            file_name=top_sections[0].file_name,
                            section_type="combined"
                        )
                        print(f"Combined {len(top_sections)} sections for broad query")
                else:
                    # Select best single section using hybrid scoring
                    best_title, score = self.simplified_kg.select_best_section(
                        normalized_query,
                        section_map,
                        embeddings_func=self.embeddings.embed_query
                    )

                    if best_title:
                        selected_section = self.simplified_kg.get_section_by_title(best_title)

                        # Validate content richness
                        if selected_section and not self.section_processor.is_content_rich(
                            selected_section.content
                        ):
                            print("KG section not rich enough, falling back...")
                            selected_section = None
                        else:
                            retrieval_method = "kg_guided"
                            print(f"KG selected: {best_title} (score: {score:.3f})")

            # ===== STAGE 3: Semantic Fallback =====
            if selected_section is None:
                print("Using semantic fallback (clustering)...")

                # Get query embedding
                query_embedding = self.embeddings.embed_query(normalized_query)

                # Select cluster
                cluster_id = self.section_clusterer.select_cluster(query_embedding)
                cluster_sections = self.section_clusterer.get_cluster_sections(cluster_id)
                print(f"Selected cluster {cluster_id} with {len(cluster_sections)} sections")

                # Get section objects for this cluster
                cluster_section_objs = [
                    s for s in self.sections if s.title in cluster_sections
                ]

                if cluster_section_objs:
                    # Select best section from cluster
                    section_map = {
                        s.title: s.content for s in cluster_section_objs
                    }
                    best_title, score = self.simplified_kg.select_best_section(
                        normalized_query,
                        section_map,
                        embeddings_func=self.embeddings.embed_query
                    )

                    if best_title:
                        selected_section = next(
                            s for s in cluster_section_objs if s.title == best_title
                        )
                        retrieval_method = "semantic_fallback"
                        print(f"Cluster selected: {best_title} (score: {score:.3f})")

            # If still no section, use first section as fallback
            if selected_section is None and self.sections:
                selected_section = self.sections[0]
                retrieval_method = "default_fallback"
                print("Using default fallback (first section)")

            if selected_section is None:
                return ChatResponse(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    confidence_score=0.0,
                    retrieval_method="none"
                )

            # ===== STAGE 4: MMR Sentence Selection =====
            print("Selecting sentences with MMR...")
            sentences = tokenize_sentences(selected_section.content)
            print(f"Total sentences: {len(sentences)}")

            if not sentences:
                return ChatResponse(
                    answer="The retrieved section contains no sentences.",
                    sources=[],
                    confidence_score=0.0
                )

            mmr_sentences, mmr_indices = self.mmr_retriever.select_sentences_with_indices(
                normalized_query,
                sentences,
                k=min(settings.mmr_k, len(sentences)),
                lambda_param=settings.mmr_lambda
            )
            print(f"MMR selected {len(mmr_sentences)} sentences")

            # ===== STAGE 5: Answer Generation =====
            if settings.use_llm_refinement and self.llm:
                # Use LLM to refine answer
                context = "\n".join(mmr_sentences)
                # Clean up context (remove separator lines)
                context = self._clean_text(context)

                prompt = f"""Based on the following context, answer the question concisely and precisely.

Extract only the most relevant information. Remove any formatting artifacts or unnecessary details.

Context:
{context}

Question: {question}

Provide a clear, direct answer without extra formatting or separators."""
                answer = self.llm.predict(prompt)
                # Clean the final answer
                answer = self._clean_text(answer)
            else:
                # Join MMR sentences directly
                answer = " ".join(mmr_sentences)
                # Clean up separators and formatting
                answer = self._clean_text(answer)

            # ===== STAGE 6: Calculate Evaluation Metrics =====
            # For demo, assume all retrieved sentences are relevant
            # In production, would need ground truth labels
            relevant_indices = mmr_indices  # Simplified assumption

            precision = self.evaluator.calculate_precision_at_k(
                mmr_indices,
                relevant_indices,
                k=min(3, len(mmr_indices))
            )
            recall = self.evaluator.calculate_recall_at_k(
                mmr_indices,
                relevant_indices,
                k=min(3, len(mmr_indices))
            )
            mrr = self.evaluator.calculate_mrr(mmr_indices, relevant_indices)

            # ===== STAGE 7: Create Response =====
            source = SourceReference(
                document_name=selected_section.file_name,
                page_number=selected_section.page_number,
                chunk_id=f"section_{selected_section.section_index}",
                relevance_score=0.9,  # High score for selected section
                excerpt=selected_section.content[:200] + "..."
            )

            confidence = self.evaluator.calculate_confidence(len(mmr_sentences), k=5)

            return ChatResponse(
                answer=answer,
                sources=[source],
                confidence_score=confidence,
                retrieval_method=retrieval_method,
                mmr_sentences_used=len(mmr_sentences),
                cluster_id=cluster_id,
                section_title=selected_section.title,
                precision_at_k=precision,
                recall_at_k=recall,
                mrr=mrr
            )

        except Exception as e:
            import traceback
            print(f"Query error: {e}")
            print(traceback.format_exc())

            return ChatResponse(
                answer=f"I encountered an error processing your question: {str(e)}",
                sources=[],
                confidence_score=0.0,
                retrieval_method="error"
            )

    def _sections_to_documents(self, sections: List[Section]) -> List[Document]:
        """
        Convert Section objects to LangChain Document objects.

        Args:
            sections: List of Section objects

        Returns:
            List of Document objects
        """
        documents = []

        for section in sections:
            doc = Document(
                page_content=section.content,
                metadata={
                    "document_id": section.document_id,
                    "file_name": section.file_name,
                    "page_number": section.page_number,
                    "section_index": section.section_index,
                    "section_title": section.title,
                    "section_type": section.section_type,
                    "chunk_id": f"{section.document_id}_section_{section.section_index}",
                    "document_type": "policy"
                }
            )
            documents.append(doc)

        return documents

    def _generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID based on file path."""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]

    def _clean_text(self, text: str) -> str:
        """
        Clean up text by removing separator lines and excessive formatting.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        import re

        # Remove separator lines (=== or ---)
        text = re.sub(r'[=\-]{4,}', '', text)

        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the enhanced RAG system.

        Returns:
            Dictionary with system statistics
        """
        term_stats = self.term_learner.get_stats()
        kg_stats = self.simplified_kg.get_stats()
        cluster_stats = self.section_clusterer.get_stats()

        return {
            "total_sections": len(self.sections),
            "term_learner": term_stats,
            "knowledge_graph": kg_stats,
            "clustering": cluster_stats,
            "settings": {
                "chunking_strategy": settings.chunking_strategy,
                "embedding_strategy": settings.embedding_strategy,
                "use_mmr_retrieval": settings.use_mmr_retrieval,
                "mmr_k": settings.mmr_k,
                "mmr_lambda": settings.mmr_lambda,
                "n_clusters": settings.n_clusters,
                "use_llm_refinement": settings.use_llm_refinement
            }
        }
