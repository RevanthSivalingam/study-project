"""
RAG System Logic - Converted from Jupyter Notebook
Handles document loading, chunking, embeddings, and query answering
"""

import os
import json
import re
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
from llm_provider import LLMProviderFactory, LLMProvider


class RAGSystem:
    def __init__(self, api_key=None, input_folder="inputfiles", llm_config=None):
        """Initialize the RAG system"""
        self.input_folder = input_folder
        self.embedder = SentenceTransformer("all-mpnet-base-v2")

        # Load environment variables as defaults
        load_dotenv()

        # Build configuration with priority: UI input > function args > .env > defaults
        config = {
            'provider': os.getenv('LLM_PROVIDER', 'none'),
            'openai_api_key': api_key or os.getenv('OPENAI_API_KEY'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY'),
            'openai_model': os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview'),
            'gemini_model': os.getenv('GEMINI_MODEL', 'gemini-1.5-pro'),
            'temperature': float(os.getenv('LLM_TEMPERATURE', '0.3')),
            'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '500')),
            'fallback_to_mmr': os.getenv('FALLBACK_TO_MMR', 'true').lower() == 'true'
        }

        # Override with UI-provided llm_config (highest priority)
        if llm_config:
            # Merge UI config, prioritizing non-None values
            for key, value in llm_config.items():
                if value is not None:
                    config[key] = value

            # Special handling: If provider is set in UI, use that provider's API key from UI
            if 'provider' in llm_config and llm_config['provider'] != 'none':
                provider = llm_config['provider']
                # If UI provided an API key for this provider, use it
                if provider == 'openai' and llm_config.get('openai_api_key'):
                    config['openai_api_key'] = llm_config['openai_api_key']
                elif provider == 'gemini' and llm_config.get('gemini_api_key'):
                    config['gemini_api_key'] = llm_config['gemini_api_key']

        self.llm_provider = LLMProviderFactory.create_provider(config)
        self.client = None  # Deprecated - keeping for backward compatibility

        # Storage
        self.documents = []
        self.sections = []
        self.section_texts = []
        self.section_embeddings = None
        self.corpus = []
        self.knowledge_graph = defaultdict(lambda: defaultdict(list))
        self.learned_terms = []
        self.generic_terms = set()

        # Clustering
        self.NUM_CLUSTERS = 6
        self.kmeans = None
        self.cluster_ids = None
        self.cluster_map = {}
        self.cluster_centroids = {}

        self.initialized = False

    def load_documents(self):
        """Load PDF, TXT, and JSON files from folder"""
        docs = []

        if not os.path.exists(self.input_folder):
            os.makedirs(self.input_folder)
            return docs

        for filename in os.listdir(self.input_folder):
            filepath = os.path.join(self.input_folder, filename)

            # PDF FILES
            if filename.lower().endswith(".pdf"):
                with pdfplumber.open(filepath) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    docs.append(text)

            # TEXT FILES
            elif filename.lower().endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    docs.append(f.read())

            # JSON FILES
            elif filename.lower().endswith(".json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    docs.append(json.dumps(data, indent=2))

        self.documents = docs
        return docs

    def section_chunk(self, text):
        """Section-based text chunking using rule-based header detection"""
        sections = []
        current_title = None
        buffer = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Detect top-level section headers
            is_header = (
                re.match(r"^PAGE\s+\d+", line, re.IGNORECASE) or
                re.match(r"^\d+\.\s+[A-Za-z]", line)
            )

            if is_header:
                if current_title and buffer:
                    sections.append({
                        "title": current_title,
                        "content": "\n".join(buffer)
                    })

                current_title = line
                buffer = [line]
            else:
                buffer.append(line)

        if current_title and buffer:
            sections.append({
                "title": current_title,
                "content": "\n".join(buffer)
            })

        return sections

    def build_generic_terms(self, sections, top_percent=0.15):
        """Identify generic terms using TF-IDF (corpus-level)"""
        texts = [s["title"] + " " + s["content"] for s in sections]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.95,
            min_df=2
        )

        tfidf = vectorizer.fit_transform(texts)
        terms = np.array(vectorizer.get_feature_names_out())

        # Mean TF-IDF per term across corpus
        mean_scores = tfidf.mean(axis=0).A1

        # Lowest TF-IDF → most generic
        cutoff = int(len(terms) * top_percent)
        generic_terms = set(terms[np.argsort(mean_scores)[:cutoff]])

        return generic_terms

    def learn_key_terms(self, sections, top_k=40):
        """Learn key terms using TF-IDF"""
        texts = [s["content"] for s in sections if s["content"]]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.85
        )
        X = vectorizer.fit_transform(texts)

        terms = vectorizer.get_feature_names_out()
        scores = X.mean(axis=0).A1

        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ranked[:top_k]]

    def build_kg_automatically(self, sections, learned_terms):
        """Build knowledge graph automatically"""
        knowledge_graph = defaultdict(lambda: defaultdict(list))

        for sec in sections:
            content = sec["content"]
            if not content:
                continue

            for line in content.split("\n"):
                line_l = line.lower()

                matched_terms = [t for t in learned_terms if t in line_l]
                if not matched_terms:
                    continue

                numbers = re.findall(r"\d+\s+(days|weeks|months)", line_l)

                for term in matched_terms:
                    if numbers:
                        knowledge_graph[term]["limits"].append(line.strip())
                    else:
                        knowledge_graph[term]["description"].append(line.strip())

        return knowledge_graph

    def query_knowledge_graph(self, query):
        """Query the knowledge graph"""
        q = query.lower()
        collected = []

        for entity, facts in self.knowledge_graph.items():
            if entity in q:
                for v in facts.values():
                    collected.extend(v)

        return collected if collected else None

    def build_corpus(self, sections):
        """Build corpus for semantic retrieval"""
        return [(s["title"] + " " + s["content"]).lower() for s in sections]

    def cluster_sections(self):
        """Cluster document sections using K-Means"""
        self.kmeans = KMeans(n_clusters=self.NUM_CLUSTERS, random_state=42)
        self.cluster_ids = self.kmeans.fit_predict(self.section_embeddings)

        self.cluster_map = {}
        for i, cid in enumerate(self.cluster_ids):
            self.cluster_map.setdefault(cid, []).append(self.sections[i])

        self.cluster_centroids = {
            cid: np.mean(
                self.embedder.encode([s["title"] + " " + s["content"] for s in sec]),
                axis=0
            )
            for cid, sec in self.cluster_map.items()
        }

    def select_cluster(self, query):
        """Select relevant cluster using semantic similarity"""
        q_emb = self.embedder.encode([query])[0]
        scores = {
            cid: cosine_similarity([q_emb], [centroid])[0][0]
            for cid, centroid in self.cluster_centroids.items()
        }
        return max(scores, key=scores.get)

    def normalize_query_for_retrieval(self, query):
        """Remove generic terms from query"""
        tokens = query.lower().split()
        filtered = [t for t in tokens if t not in self.generic_terms]
        return " ".join(filtered) if filtered else query

    def select_best_section(self, sections, query, threshold=0.45):
        """Select best section using similarity threshold"""
        q_emb = self.embedder.encode([query])[0]

        # TITLE MATCHING
        titles = [s["title"] for s in sections]
        title_embeddings = self.embedder.encode(titles)
        title_scores = cosine_similarity([q_emb], title_embeddings)[0]

        # CONTENT MATCHING
        contents = [s["content"] for s in sections]
        content_embeddings = self.embedder.encode(contents)
        content_scores = cosine_similarity([q_emb], content_embeddings)[0]

        # COMBINE SCORES (Weighted)
        combined_scores = [
            0.6 * title_scores[i] + 0.4 * content_scores[i]
            for i in range(len(sections))
        ]

        best_idx = int(np.argmax(combined_scores))
        return sections[best_idx]

    def is_section_relevant(self, section, query, threshold=0.55):
        """Check if section is relevant with length normalization"""
        section_text = (section["title"] + " " + section["content"]).strip()

        sec_emb = self.embedder.encode(section_text)
        qry_emb = self.embedder.encode(query)

        score = cosine_similarity([sec_emb], [qry_emb])[0][0]

        # LENGTH NORMALIZATION
        length_factor = min(len(section_text) / 300, 1.0)
        score = score * (0.7 + 0.3 * length_factor)

        return score >= threshold

    def group_by_section(self, sections, candidate_lines):
        """Group knowledge graph facts by section"""
        section_map = {}
        for sec in sections:
            sec_lines = sec["content"].split("\n")
            matched = [l for l in sec_lines if l in candidate_lines]
            if matched:
                section_map[sec["title"]] = matched
        return section_map

    def select_best_kg_section(self, query, section_map):
        """Select best section from knowledge graph results"""
        if not section_map:
            return None, None

        query_l = query.lower()
        query_terms = set(query_l.split())

        titles = list(section_map.keys())
        title_embs = self.embedder.encode(titles)
        q_emb = self.embedder.encode([query])[0]

        sims = cosine_similarity([q_emb], title_embs)[0]

        final_scores = []

        for i, title in enumerate(titles):
            title_l = title.lower()

            # lexical overlap bonus
            overlap = sum(1 for t in query_terms if t in title_l)

            # final score
            score = sims[i] + (0.25 * overlap)
            final_scores.append(score)

        best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])

        return titles[best_idx], section_map[titles[best_idx]]

    def mmr(self, query, sentences, k=6, lambda_param=0.7, return_scores=False):
        """
        Maximal Marginal Relevance for diverse sentence selection

        Args:
            query: Search query
            sentences: List of candidate sentences
            k: Number of sentences to select
            lambda_param: Balance between relevance and diversity
            return_scores: If True, return (sentences, scores) tuple

        Returns:
            List of selected sentences, or (sentences, scores) if return_scores=True
        """
        if not sentences:
            return ([], []) if return_scores else []

        sent_embs = self.embedder.encode(sentences)
        q_emb = self.embedder.encode([query])[0]

        selected = []
        selected_scores = []
        used = set()

        for _ in range(min(k, len(sentences))):
            scores = []
            for i, emb in enumerate(sent_embs):
                if i in used:
                    continue
                relevance = cosine_similarity([q_emb], [emb])[0][0]
                diversity = max(
                    [cosine_similarity([emb], [sent_embs[j]])[0][0] for j in used],
                    default=0
                )
                score = lambda_param * relevance - (1 - lambda_param) * diversity
                scores.append((score, i, relevance))

            if not scores:
                break

            best_score, best_idx, best_relevance = max(scores)
            used.add(best_idx)
            selected.append(sentences[best_idx])
            selected_scores.append(float(best_relevance))

        if return_scores:
            return selected, selected_scores
        return selected

    def retrieve_with_strategy(self, query: str, sentences: list, section_title: str, use_llm: bool = False):
        """
        Retrieve sentences using LLM-aware strategy

        Args:
            query: User query
            sentences: Candidate sentences from section
            section_title: Title of source section
            use_llm: Whether LLM is available for synthesis

        Returns:
            dict with retrieved_sentences, scores, sources, and metadata
        """
        if use_llm:
            # With LLM: Retrieve MORE context (8-10 sentences)
            # LLM will synthesize and extract relevant info
            k = min(10, len(sentences))
            selected, scores = self.mmr(query, sentences, k=k, lambda_param=0.6, return_scores=True)
        else:
            # Without LLM: Precise retrieval (4-5 sentences)
            # Direct concatenation, so be selective
            k = min(5, len(sentences))
            selected, scores = self.mmr(query, sentences, k=k, lambda_param=0.75, return_scores=True)

        # Build source attributions
        sources = []
        for i, (sentence, score) in enumerate(zip(selected, scores)):
            sources.append({
                "text": sentence,
                "section": section_title,
                "relevance_score": score,
                "rank": i + 1
            })

        return {
            "sentences": selected,
            "scores": scores,
            "sources": sources,
            "avg_relevance": float(np.mean(scores)) if scores else 0.0,
            "num_sources": len(selected)
        }

    def calculate_confidence(self, method: str, avg_relevance: float, num_sources: int,
                            section_match_score: float = 0.0) -> dict:
        """
        Calculate actual confidence/trust score

        Args:
            method: Retrieval method used ('knowledge_graph' or 'semantic_search')
            avg_relevance: Average relevance score of retrieved sentences
            num_sources: Number of sources retrieved
            section_match_score: How well the section matched the query

        Returns:
            dict with confidence score and confidence level
        """
        # Base score from relevance
        base_score = avg_relevance

        # Method bonus (knowledge graph is more reliable)
        if method == "knowledge_graph":
            method_bonus = 0.15
        elif method == "semantic_search":
            method_bonus = 0.08
        else:
            method_bonus = 0.0

        # Source quantity factor (more sources = more confident, up to a point)
        source_factor = min(num_sources / 5.0, 1.0) * 0.1

        # Section match factor
        section_factor = section_match_score * 0.1

        # Calculate final confidence
        confidence = min(base_score + method_bonus + source_factor + section_factor, 1.0)

        # Determine confidence level
        if confidence >= 0.85:
            level = "Very High"
        elif confidence >= 0.70:
            level = "High"
        elif confidence >= 0.55:
            level = "Medium"
        elif confidence >= 0.40:
            level = "Low"
        else:
            level = "Very Low"

        return {
            "score": float(confidence),
            "level": level,
            "breakdown": {
                "base_relevance": float(base_score),
                "method_bonus": float(method_bonus),
                "source_factor": float(source_factor),
                "section_factor": float(section_factor)
            }
        }

    def clean_answer(self, answer: str) -> str:
        """
        Clean up answer text by removing unnecessary characters and formatting

        Args:
            answer: Raw answer text

        Returns:
            Cleaned answer text
        """
        # Remove excessive newlines
        answer = re.sub(r'\n{3,}', '\n\n', answer)

        # Remove leading/trailing whitespace
        answer = answer.strip()

        # Remove markdown artifacts if present
        answer = re.sub(r'\*\*\*+', '', answer)
        answer = re.sub(r'___+', '', answer)

        # Fix spacing around punctuation
        answer = re.sub(r'\s+([.,;:!?])', r'\1', answer)
        answer = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', answer)

        # Remove multiple spaces
        answer = re.sub(r' {2,}', ' ', answer)

        # Ensure proper sentence spacing
        answer = re.sub(r'\.(?=[A-Z])', '. ', answer)

        return answer.strip()

    def synthesize_with_llm(self, query: str, retrieved_sentences: list) -> dict:
        """Use LLM to synthesize natural answer from retrieved sentences"""
        try:
            result = self.llm_provider.generate_answer(query, retrieved_sentences)
            return {
                "answer": result['answer'],
                "provider": result['provider'],
                "tokens_used": result.get('tokens_used', 0),
                "fallback_used": result.get('fallback_used', False)
            }
        except Exception as e:
            # Fallback to MMR on any error
            return {
                "answer": " ".join(retrieved_sentences),
                "provider": "mmr_fallback",
                "error": str(e),
                "fallback_used": True,
                "tokens_used": 0
            }

    def is_content_rich(self, section):
        """Check if section has sufficient content"""
        return (
            len(section["content"].strip()) > 60 and
            len(section["content"].split()) > 10
        )

    def answer_query(self, query):
        """
        End-to-end query answering with LLM-aware retrieval

        Uses different retrieval strategies based on LLM availability:
        - With LLM: More context (8-10 sentences), LLM synthesis
        - Without LLM: Precise retrieval (4-5 sentences), direct concatenation
        """
        if not self.initialized:
            return {
                "answer": "System not initialized. Please load documents first.",
                "section_title": None,
                "retrieved_sentences": [],
                "sources": [],
                "method": "error",
                "confidence": {"score": 0.0, "level": "None"}
            }

        # Determine if LLM is available (not MMR provider)
        use_llm = self.llm_provider.__class__.__name__ != 'MMRProvider'

        # Normalize query
        normalized_query = self.normalize_query_for_retrieval(query)

        # STEP 1: Try Knowledge Graph retrieval
        kg_candidates = self.query_knowledge_graph(normalized_query)
        method = None
        full_section = None
        section_match_score = 0.0

        if kg_candidates:
            grouped = self.group_by_section(self.sections, kg_candidates)
            best_title, _ = self.select_best_kg_section(normalized_query, grouped)

            if best_title:
                candidate = next(s for s in self.sections if s["title"] == best_title)

                if not self.is_content_rich(candidate):
                    # Fallback: closest section with real content
                    content_sections = [s for s in self.sections if self.is_content_rich(s)]
                    candidate = self.select_best_section(content_sections, normalized_query)

                full_section = candidate
                method = "knowledge_graph"

                # Calculate section match score
                section_text = full_section["title"] + " " + full_section["content"]
                sec_emb = self.embedder.encode([section_text])[0]
                q_emb = self.embedder.encode([normalized_query])[0]
                section_match_score = float(cosine_similarity([sec_emb], [q_emb])[0][0])

        # STEP 2: Semantic search fallback
        if not full_section:
            cluster_id = self.select_cluster(normalized_query)
            cluster_sections = self.cluster_map[cluster_id]
            full_section = self.select_best_section(cluster_sections, normalized_query)

            if not full_section:
                return {
                    "answer": "I apologize, but I cannot find relevant information about your question in the available documents.",
                    "section_title": None,
                    "retrieved_sentences": [],
                    "sources": [],
                    "method": "not_found",
                    "confidence": {"score": 0.0, "level": "None"},
                    "llm_provider": "none",
                    "tokens_used": 0
                }

            method = "semantic_search"

            # Calculate section match score
            section_text = full_section["title"] + " " + full_section["content"]
            sec_emb = self.embedder.encode([section_text])[0]
            q_emb = self.embedder.encode([normalized_query])[0]
            section_match_score = float(cosine_similarity([sec_emb], [q_emb])[0][0])

        # STEP 3: Retrieve sentences with LLM-aware strategy
        sentences = sent_tokenize(full_section["content"])
        retrieval_result = self.retrieve_with_strategy(
            normalized_query,
            sentences,
            full_section["title"],
            use_llm=use_llm
        )

        # STEP 4: Generate answer
        if use_llm:
            # Use LLM for synthesis
            synthesis_result = self.synthesize_with_llm(
                query,  # Use original query for LLM
                retrieval_result["sentences"]
            )
            raw_answer = synthesis_result['answer']

            # Apply the same beautification rules as MMR mode
            # This ensures consistent formatting across both modes
            answer = self._beautify_text(raw_answer)
            answer = self.clean_answer(answer)

            llm_provider = synthesis_result.get('provider', 'unknown')
            tokens_used = synthesis_result.get('tokens_used', 0)
            fallback_used = synthesis_result.get('fallback_used', False)
        else:
            # Direct concatenation with proper formatting
            answer = self._format_mmr_answer(retrieval_result["sentences"])
            llm_provider = "mmr"
            tokens_used = 0
            fallback_used = False

        # STEP 5: Calculate confidence score
        confidence = self.calculate_confidence(
            method=method,
            avg_relevance=retrieval_result["avg_relevance"],
            num_sources=retrieval_result["num_sources"],
            section_match_score=section_match_score
        )

        # STEP 6: Return comprehensive result
        return {
            "answer": answer,
            "section_title": full_section["title"],
            "retrieved_sentences": retrieval_result["sentences"],
            "sources": retrieval_result["sources"],
            "method": method,
            "confidence": confidence,
            "llm_provider": llm_provider,
            "tokens_used": tokens_used,
            "fallback_used": fallback_used,
            "metadata": {
                "query": query,
                "normalized_query": normalized_query,
                "section_match_score": section_match_score,
                "avg_sentence_relevance": retrieval_result["avg_relevance"],
                "num_sentences_retrieved": retrieval_result["num_sources"],
                "use_llm": use_llm
            }
        }

    def _format_mmr_answer(self, sentences: list) -> str:
        """
        Format MMR sentences into a coherent answer without LLM

        Args:
            sentences: List of retrieved sentences

        Returns:
            Formatted answer string with beautification
        """
        if not sentences:
            return "No relevant information found."

        # Apply beautification
        answer = self._beautify_mmr_answer(sentences)

        # Final cleanup
        answer = self.clean_answer(answer)

        return answer

    def _beautify_text(self, text: str) -> str:
        """
        Apply programmatic formatting to any text (LLM or MMR output)

        Detects patterns and adds structure:
        - Lists (numbered/bulleted items)
        - Key terms (bold)
        - Paragraph breaks
        - Section headers

        Args:
            text: Raw text to beautify (can be LLM answer or concatenated sentences)

        Returns:
            Formatted markdown string
        """
        if not text or not text.strip():
            return "No relevant information found."

        # Split text into lines for processing
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        formatted_lines = []
        current_paragraph = []

        # Patterns for detection
        section_header_pattern = re.compile(r'^\d+\.\d+\s+[A-Z].*:$|^[A-Z][A-Z\s]{8,}:?$')  # "2.1 Title:" or "SECTION TITLE"
        subsection_pattern = re.compile(r'^(Review Period|Review Window|Purpose|30-Day Review|60-Day Review|90-Day Review):')  # Field labels
        list_pattern = re.compile(r'^[\-\*\•]\s+|^\d+[\.\)]\s+[a-z]|first|second|third', re.IGNORECASE)
        key_term_pattern = re.compile(r'\b(policy|procedure|required|mandatory|eligible|must|shall|within \d+ days|minimum|maximum)\b', re.IGNORECASE)
        topic_shift_pattern = re.compile(r'\b(however|additionally|furthermore|in contrast|alternatively|note that)\b', re.IGNORECASE)

        def bold_key_terms(text: str) -> str:
            """Add bold formatting to key policy terms"""
            terms_to_bold = [
                'eligible', 'required', 'mandatory', 'must', 'shall',
                'policy', 'procedure', 'approval', 'within \\d+ (days|weeks|months)'
            ]
            for term in terms_to_bold:
                text = re.sub(f'\\b({term})\\b', r'**\1**', text, flags=re.IGNORECASE)
            return text

        def detect_section_header(text: str) -> bool:
            """Check if line is a section header"""
            return bool(section_header_pattern.match(text.strip()))

        def detect_subsection(text: str) -> bool:
            """Check if line is a subsection/field label"""
            return bool(subsection_pattern.match(text.strip()))

        def detect_list_item(text: str) -> bool:
            """Check if line appears to be a list item"""
            return bool(list_pattern.search(text.strip()[:30]))

        def detect_topic_shift(text: str) -> bool:
            """Check if line signals a topic shift"""
            return bool(topic_shift_pattern.search(text.strip()[:50]))

        # Process lines
        in_list = False

        for i, line in enumerate(lines):
            if not line:
                continue

            # Skip separator lines
            if re.match(r'^[=\-_]{4,}$', line):
                continue

            is_section_header = detect_section_header(line)
            is_subsection = detect_subsection(line)
            is_list_item = detect_list_item(line)
            is_topic_shift = detect_topic_shift(line)

            if is_section_header:
                # Flush current paragraph
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    formatted_lines.append('')
                    current_paragraph = []

                # Add section header with spacing
                if formatted_lines:  # Add space before if not first item
                    formatted_lines.append('')
                formatted_lines.append(f"**{line}**")
                formatted_lines.append('')
                in_list = False

            elif is_subsection:
                # Flush current paragraph
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    current_paragraph = []

                # Add subsection as its own line
                formatted_lines.append(bold_key_terms(line))
                in_list = False

            elif is_list_item:
                # Flush current paragraph if starting a list
                if current_paragraph and not in_list:
                    formatted_lines.append(' '.join(current_paragraph))
                    formatted_lines.append('')  # Blank line before list
                    current_paragraph = []

                # Format as bullet point
                clean_line = re.sub(r'^\d+[\.\)]\s+|^[\-\*\•]\s+', '', line)
                formatted_lines.append(f"- {bold_key_terms(clean_line)}")
                in_list = True

            elif is_topic_shift and current_paragraph:
                # End current paragraph and start new one
                formatted_lines.append(' '.join(current_paragraph))
                formatted_lines.append('')  # Paragraph break
                current_paragraph = [bold_key_terms(line)]
                in_list = False

            else:
                # Regular line - add to current paragraph
                if in_list:
                    formatted_lines.append('')
                    in_list = False

                current_paragraph.append(bold_key_terms(line))

                # Break paragraphs every 2-3 lines
                if len(current_paragraph) >= 2 and i < len(lines) - 1:
                    formatted_lines.append(' '.join(current_paragraph))
                    formatted_lines.append('')
                    current_paragraph = []

        # Flush remaining paragraph
        if current_paragraph:
            formatted_lines.append(' '.join(current_paragraph))

        # Join and clean up
        result = '\n'.join(formatted_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 newlines

        return result.strip()

    def _beautify_mmr_answer(self, sentences: list) -> str:
        """
        Apply programmatic formatting to MMR concatenated sentences

        This is a wrapper that converts sentence list to text and applies beautification.

        Args:
            sentences: List of retrieved sentences (may contain embedded newlines)

        Returns:
            Formatted markdown string
        """
        if not sentences:
            return "No relevant information found."

        # Join sentences into text
        text = '\n'.join(str(s) for s in sentences if s)

        # Apply unified beautification
        return self._beautify_text(text)

    def initialize(self):
        """Initialize the RAG system - load and process documents"""
        # Load documents
        self.documents = self.load_documents()

        if not self.documents:
            return False, "No documents found in inputfiles folder"

        # Extract sections
        for doc in self.documents:
            self.sections.extend(self.section_chunk(doc))

        if not self.sections:
            return False, "No sections extracted from documents"

        # Create embeddings
        self.section_texts = [s["title"] + " " + s["content"] for s in self.sections]
        self.section_embeddings = self.embedder.encode(self.section_texts)

        # Learn key terms
        self.learned_terms = self.learn_key_terms(self.sections)

        # Build knowledge graph
        self.knowledge_graph = self.build_kg_automatically(self.sections, self.learned_terms)

        # Build corpus
        self.corpus = self.build_corpus(self.sections)

        # Cluster sections
        self.cluster_sections()

        # Learn generic terms
        self.generic_terms = self.build_generic_terms(self.sections)

        self.initialized = True
        return True, f"Successfully initialized with {len(self.documents)} documents and {len(self.sections)} sections"
