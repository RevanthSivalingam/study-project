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


class RAGSystem:
    def __init__(self, api_key=None, input_folder="inputfiles"):
        """Initialize the RAG system"""
        self.input_folder = input_folder
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.client = OpenAI(api_key=api_key) if api_key else None

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

    def mmr(self, query, sentences, k=6, lambda_param=0.7):
        """Maximal Marginal Relevance for diverse sentence selection"""
        sent_embs = self.embedder.encode(sentences)
        q_emb = self.embedder.encode([query])[0]

        selected = []
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
                scores.append((score, i))

            if not scores:
                break

            best = max(scores)[1]
            used.add(best)
            selected.append(sentences[best])

        return selected

    def is_content_rich(self, section):
        """Check if section has sufficient content"""
        return (
            len(section["content"].strip()) > 60 and
            len(section["content"].split()) > 10
        )

    def answer_query(self, query):
        """End-to-end query answering with knowledge-guided RAG"""
        if not self.initialized:
            return {
                "answer": "System not initialized. Please load documents first.",
                "section_title": None,
                "retrieved_sentences": [],
                "method": "error"
            }

        # STEP 1: Knowledge Graph candidate generation
        normalized_query = self.normalize_query_for_retrieval(query)
        kg_candidates = self.query_knowledge_graph(normalized_query)

        if kg_candidates:
            grouped = self.group_by_section(self.sections, kg_candidates)
            best_title, _ = self.select_best_kg_section(normalized_query, grouped)

            if best_title:
                candidate = next(s for s in self.sections if s["title"] == best_title)

                if not self.is_content_rich(candidate):
                    # fallback: closest section with real content
                    content_sections = [s for s in self.sections if self.is_content_rich(s)]
                    candidate = self.select_best_section(content_sections, normalized_query)

                full_section = candidate
                sentences = sent_tokenize(full_section["content"])
                final = self.mmr(normalized_query, sentences)

                answer = " ".join(final)

                return {
                    "answer": answer,
                    "section_title": best_title,
                    "retrieved_sentences": final,
                    "method": "knowledge_graph"
                }

        # STEP 2: Semantic fallback
        cluster_id = self.select_cluster(normalized_query)
        cluster_sections = self.cluster_map[cluster_id]

        full_section = self.select_best_section(cluster_sections, normalized_query)

        if not full_section:
            return {
                "answer": "The requested information is not available in the current knowledge base.",
                "section_title": None,
                "retrieved_sentences": [],
                "method": "not_found"
            }

        sentences = sent_tokenize(full_section["content"])
        final = self.mmr(normalized_query, sentences)

        answer = " ".join(final)

        return {
            "answer": answer,
            "section_title": full_section['title'],
            "retrieved_sentences": final,
            "method": "semantic_search"
        }

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
