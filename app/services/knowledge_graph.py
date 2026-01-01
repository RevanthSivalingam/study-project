import os
import pickle
from typing import List, Dict, Any
import networkx as nx
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.utils.chat import get_chat_llm
from config.settings import settings


class KnowledgeGraph:
    """
    Manages knowledge graph operations using NetworkX (in-memory graph)

    This is a lightweight alternative to Neo4j that stores the graph in memory
    and can persist to disk. Perfect for small-to-medium datasets.
    """

    def __init__(self):
        # Initialize NetworkX directed graph
        self.graph = nx.DiGraph()

        # Graph persistence path
        self.graph_path = os.path.join(settings.chroma_persist_directory, "knowledge_graph.pkl")

        # Load existing graph if available
        self._load_graph()

        # Initialize LLM using factory (auto-selects provider)
        self.llm = get_chat_llm(temperature=0)

        # Entity extraction prompt
        self.entity_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Extract key entities and relationships from the following policy document text.
Focus on: policy names, departments, roles, benefits, requirements, deadlines, processes.

Text: {text}

Return the information in this format:
ENTITIES:
- Entity: [name], Type: [policy/department/role/benefit/requirement/process]

RELATIONSHIPS:
- [Entity1] -> [Relationship] -> [Entity2]

Be specific and accurate. Only extract explicitly mentioned information."""
        )

    def _load_graph(self):
        """Load graph from disk if it exists"""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
                print(f"✅ Loaded knowledge graph with {self.graph.number_of_nodes()} nodes")
            except Exception as e:
                print(f"⚠️  Could not load graph: {e}")
                self.graph = nx.DiGraph()

    def _save_graph(self):
        """Persist graph to disk"""
        try:
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.graph, f)
        except Exception as e:
            print(f"⚠️  Could not save graph: {e}")

    def close(self):
        """Save graph before closing"""
        self._save_graph()

    def create_document_node(self, document_id: str, metadata: Dict[str, Any]):
        """Create a document node in the knowledge graph"""
        self.graph.add_node(
            f"doc_{document_id}",
            node_type="document",
            document_id=document_id,
            file_name=metadata.get("file_name", ""),
            file_path=metadata.get("file_path", ""),
            document_type=metadata.get("document_type", "policy"),
            total_pages=metadata.get("total_pages", 0)
        )
        self._save_graph()

    def extract_and_store_entities(self, documents: List[Document]) -> int:
        """
        Extract entities from documents and store in knowledge graph

        Returns:
            Number of entities extracted
        """
        entity_count = 0

        # Skip entity extraction for now to avoid API compatibility issues
        # This can be re-enabled once LangChain versions are stabilized
        print("⚠️  Entity extraction temporarily disabled due to API compatibility")

        # Still save the graph (even if empty)
        self._save_graph()

        return entity_count

    def _parse_entities(self, extraction_result: str) -> List[Dict[str, str]]:
        """Parse entities from LLM extraction result"""
        entities = []
        lines = extraction_result.split("\n")

        in_entities_section = False
        for line in lines:
            if "ENTITIES:" in line:
                in_entities_section = True
                continue
            if "RELATIONSHIPS:" in line:
                in_entities_section = False
                break

            if in_entities_section and line.strip().startswith("-"):
                # Parse format: - Entity: [name], Type: [type]
                try:
                    parts = line.strip("- ").split(",")
                    name = parts[0].split(":")[1].strip().strip("[]")
                    entity_type = parts[1].split(":")[1].strip().strip("[]")
                    entities.append({"name": name, "type": entity_type})
                except Exception:
                    continue

        return entities

    def _parse_relationships(self, extraction_result: str) -> List[Dict[str, str]]:
        """Parse relationships from LLM extraction result"""
        relationships = []
        lines = extraction_result.split("\n")

        in_relationships_section = False
        for line in lines:
            if "RELATIONSHIPS:" in line:
                in_relationships_section = True
                continue

            if in_relationships_section and line.strip().startswith("-"):
                # Parse format: - [Entity1] -> [Relationship] -> [Entity2]
                try:
                    parts = line.strip("- ").split("->")
                    entity1 = parts[0].strip().strip("[]")
                    relationship = parts[1].strip().strip("[]")
                    entity2 = parts[2].strip().strip("[]")
                    relationships.append({
                        "entity1": entity1,
                        "relationship": relationship,
                        "entity2": entity2
                    })
                except Exception:
                    continue

        return relationships

    def query_related_entities(self, entity_name: str, depth: int = 2) -> List[Dict]:
        """
        Query related entities from the knowledge graph

        Args:
            entity_name: Name of the entity to start from
            depth: Depth of relationships to traverse

        Returns:
            List of related entities
        """
        entity_id = f"entity_{entity_name.lower().replace(' ', '_')}"

        if not self.graph.has_node(entity_id):
            return []

        related = []

        try:
            # Get all nodes within depth distance
            # Using BFS to find nodes up to specified depth
            visited = set()
            queue = [(entity_id, 0)]

            while queue:
                current_node, current_depth = queue.pop(0)

                if current_depth >= depth:
                    continue

                if current_node in visited:
                    continue

                visited.add(current_node)

                # Get neighbors (both incoming and outgoing)
                neighbors = list(self.graph.successors(current_node)) + \
                           list(self.graph.predecessors(current_node))

                for neighbor in neighbors:
                    if neighbor not in visited:
                        node_data = self.graph.nodes[neighbor]

                        # Only include entity nodes
                        if node_data.get("node_type") == "entity":
                            related.append({
                                "name": node_data.get("name", ""),
                                "type": node_data.get("entity_type", ""),
                                "document_id": node_data.get("document_id", "")
                            })

                        queue.append((neighbor, current_depth + 1))

            return related[:20]  # Limit to 20 results

        except Exception as e:
            print(f"Error querying related entities: {e}")
            return []

    def get_all_entities(self, limit: int = 100) -> List[Dict]:
        """Get all entities from the knowledge graph"""
        entities = []

        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("node_type") == "entity":
                entities.append({
                    "name": node_data.get("name", ""),
                    "type": node_data.get("entity_type", ""),
                    "document_id": node_data.get("document_id", "")
                })

                if len(entities) >= limit:
                    break

        return entities

    def get_stats(self) -> Dict[str, int]:
        """Get graph statistics"""
        entity_count = sum(1 for _, data in self.graph.nodes(data=True)
                          if data.get("node_type") == "entity")
        document_count = sum(1 for _, data in self.graph.nodes(data=True)
                            if data.get("node_type") == "document")

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_nodes": entity_count,
            "document_nodes": document_count
        }

    def reset_graph(self):
        """Clear the entire graph"""
        self.graph.clear()
        self._save_graph()
