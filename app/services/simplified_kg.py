"""
Simplified Knowledge Graph for term-based section retrieval.

This lightweight KG maps key terms to document sections using pattern matching,
enabling fast knowledge-guided retrieval without complex graph structures.
"""
import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from app.services.section_processor import Section


class SimplifiedKG:
    """
    Lightweight knowledge graph using term→section mappings.

    Structure: {term: {"limits": [section_titles], "description": [section_titles]}}
    """

    # Patterns for extracting numeric constraints
    NUMERIC_CONSTRAINT = re.compile(
        r'(\d+)\s*(days?|weeks?|months?|years?|hours?|minutes?|percent|%)',
        re.IGNORECASE
    )

    # Patterns for key phrases
    LIMIT_PHRASES = [
        'maximum', 'minimum', 'limit', 'up to', 'at least',
        'no more than', 'no less than', 'cannot exceed'
    ]

    def __init__(self):
        """Initialize empty knowledge graph."""
        self.graph: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: {"limits": [], "description": []}
        )
        self.section_map: Dict[str, Section] = {}  # title -> Section
        self.term_sections: Dict[str, Set[str]] = defaultdict(set)  # term -> set of section titles

    def build_from_sections(
        self,
        sections: List[Section],
        key_terms: List[str]
    ) -> None:
        """
        Build knowledge graph from sections and key terms.

        Args:
            sections: List of document sections
            key_terms: List of important terms to track
        """
        # Clear existing graph
        self.graph.clear()
        self.section_map.clear()
        self.term_sections.clear()

        # Index sections by title
        for section in sections:
            self.section_map[section.title] = section

        # Build term→section mappings
        for section in sections:
            content_lower = section.content.lower()

            for term in key_terms:
                term_lower = term.lower()

                # Check if term appears in section
                if term_lower in content_lower:
                    self.term_sections[term_lower].add(section.title)

                    # Classify the relationship type
                    relation_type = self._classify_relation(content_lower, term_lower)

                    if relation_type:
                        self.graph[term_lower][relation_type].append(section.title)

        print(f"Built KG with {len(self.graph)} terms across {len(sections)} sections")

    def _classify_relation(self, content: str, term: str) -> Optional[str]:
        """
        Classify the type of relationship between term and content.

        Args:
            content: Section content (lowercased)
            term: The term (lowercased)

        Returns:
            'limits' if content contains constraints, 'description' otherwise
        """
        # Extract context around the term (±100 characters)
        try:
            term_pos = content.index(term)
            start = max(0, term_pos - 100)
            end = min(len(content), term_pos + len(term) + 100)
            context = content[start:end]
        except ValueError:
            return "description"

        # Check for numeric constraints
        if self.NUMERIC_CONSTRAINT.search(context):
            return "limits"

        # Check for limit-related phrases
        for phrase in self.LIMIT_PHRASES:
            if phrase in context:
                return "limits"

        # Default to description
        return "description"

    def query(self, query: str, relation_type: Optional[str] = None) -> List[str]:
        """
        Query the knowledge graph for relevant section titles.

        Args:
            query: Search query (normalized)
            relation_type: Optional filter ('limits' or 'description')

        Returns:
            List of section titles containing query terms
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        matching_sections = set()

        for term in query_terms:
            # Check exact matches
            if term in self.graph:
                if relation_type:
                    matching_sections.update(self.graph[term][relation_type])
                else:
                    matching_sections.update(self.graph[term]["limits"])
                    matching_sections.update(self.graph[term]["description"])

            # Check partial matches (term is substring of key term)
            for key_term in self.term_sections:
                if term in key_term or key_term in term:
                    matching_sections.update(self.term_sections[key_term])

        return list(matching_sections)

    def group_by_section(
        self,
        sections: List[Section],
        candidate_titles: List[str]
    ) -> Dict[str, str]:
        """
        Group candidate sections and extract their content.

        Args:
            sections: All available sections
            candidate_titles: Titles of candidate sections

        Returns:
            Dictionary mapping section titles to their content
        """
        section_map = {}

        for section in sections:
            if section.title in candidate_titles:
                section_map[section.title] = section.content

        return section_map

    def select_best_section(
        self,
        query: str,
        section_map: Dict[str, str],
        embeddings_func=None
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Select the best section from candidates.

        Uses hybrid scoring:
        - Semantic similarity (if embeddings provided)
        - Lexical overlap (Jaccard similarity)

        Args:
            query: The search query
            section_map: Dictionary of section_title -> content
            embeddings_func: Optional function to get embeddings

        Returns:
            Tuple of (best_section_title, score)
        """
        if not section_map:
            return None, None

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scores = {}

        for title, content in section_map.items():
            content_lower = content.lower()
            content_terms = set(content_lower.split())

            # Lexical overlap (Jaccard similarity)
            intersection = query_terms.intersection(content_terms)
            union = query_terms.union(content_terms)
            lexical_score = len(intersection) / len(union) if union else 0

            # If embeddings provided, add semantic score
            if embeddings_func:
                try:
                    query_emb = embeddings_func(query)
                    content_emb = embeddings_func(content)
                    semantic_score = self._cosine_similarity(query_emb, content_emb)
                    # Weighted combination: 60% semantic, 40% lexical
                    final_score = 0.6 * semantic_score + 0.4 * lexical_score
                except Exception as e:
                    print(f"Embedding error: {e}")
                    final_score = lexical_score
            else:
                final_score = lexical_score

            scores[title] = final_score

        # Select section with highest score
        if scores:
            best_title = max(scores, key=scores.get)
            return best_title, scores[best_title]

        return None, None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        import numpy as np

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_related_terms(self, term: str, max_terms: int = 5) -> List[str]:
        """
        Find terms that appear in the same sections as the given term.

        Args:
            term: The term to find related terms for
            max_terms: Maximum number of related terms to return

        Returns:
            List of related terms
        """
        term_lower = term.lower()

        if term_lower not in self.term_sections:
            return []

        # Get sections containing this term
        term_section_titles = self.term_sections[term_lower]

        # Find other terms in these sections
        related_term_counts = defaultdict(int)

        for other_term, section_titles in self.term_sections.items():
            if other_term == term_lower:
                continue

            # Count overlap with term's sections
            overlap = len(term_section_titles.intersection(section_titles))
            if overlap > 0:
                related_term_counts[other_term] = overlap

        # Sort by overlap count
        sorted_terms = sorted(
            related_term_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [term for term, _ in sorted_terms[:max_terms]]

    def get_section_by_title(self, title: str) -> Optional[Section]:
        """
        Retrieve a section by its title.

        Args:
            title: The section title

        Returns:
            Section object or None
        """
        return self.section_map.get(title)

    def get_stats(self) -> Dict:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with KG statistics
        """
        total_mappings = sum(
            len(relations["limits"]) + len(relations["description"])
            for relations in self.graph.values()
        )

        return {
            'num_terms': len(self.graph),
            'num_sections': len(self.section_map),
            'total_mappings': total_mappings,
            'avg_sections_per_term': total_mappings / len(self.graph) if self.graph else 0
        }
