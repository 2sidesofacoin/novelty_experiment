import numpy as np
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

from novelty_filter.db.connection import get_connection
from novelty_filter.embeddings.openai_embeddings import OpenAIEmbeddings

class FactComparisonSystem:
    def __init__(self, db_path="facts.duckdb", embedding_service=None):
        """
        Initialize the fact comparison system
        
        Args:
            db_path: Path to the DuckDB database
            embedding_service: Service for generating embeddings
        """
        self.conn = get_connection(db_path)
        self.embedding_service = embedding_service or OpenAIEmbeddings()
    
    def add_fact(self, entity_id: int, fact_text: str, source_url: Optional[str] = None, 
                 source_name: Optional[str] = None, timestamp_published: Optional[datetime] = None) -> int:
        """
        Add a new fact to the database
        
        Args:
            entity_id: ID of the entity this fact relates to
            fact_text: The text of the fact
            source_url: URL of the source document
            source_name: Name of the source
            timestamp_published: When the fact was published
            
        Returns:
            fact_id: ID of the newly added fact
        """
        # Generate vector embedding
        embedding = self._generate_embedding(fact_text)
        
        # Generate hash for exact matching
        hash_sig = self._generate_hash(fact_text)
        
        # Get the next fact_id
        result = self.conn.execute("SELECT COALESCE(MAX(fact_id), 0) + 1 AS next_id FROM facts").fetchone()
        fact_id = result[0]
        
        # Prepare timestamp
        timestamp_published = timestamp_published or datetime.utcnow()
        
        # Insert fact
        self.conn.execute("""
            INSERT INTO facts (
                fact_id, entity_id, fact_text, fact_vector, source_url, 
                source_name, timestamp_captured, timestamp_published, hash_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fact_id, 
            entity_id, 
            fact_text, 
            self._serialize_vector(embedding), 
            source_url, 
            source_name, 
            datetime.utcnow(), 
            timestamp_published, 
            hash_sig
        ))
        
        return fact_id
    
    def add_facts_batch(self, facts: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple facts to the database in batch mode
        
        Args:
            facts: List of fact dictionaries, each containing:
                - entity_id: ID of the entity this fact relates to
                - fact_text: The text of the fact
                - source_url: (Optional) URL of the source document
                - source_name: (Optional) Name of the source
                - timestamp_published: (Optional) When the fact was published
                
        Returns:
            List[int]: List of fact_ids for the newly added facts
        """
        if not facts:
            return []
        
        # Generate vector embeddings for all texts at once
        fact_texts = [f['fact_text'] for f in facts]
        all_embeddings = self._generate_embeddings_batch(fact_texts)
        
        # Generate hashes for all facts
        hash_signatures = [self._generate_hash(text) for text in fact_texts]
        
        # Get the next fact_id
        result = self.conn.execute("SELECT COALESCE(MAX(fact_id), 0) + 1 AS next_id FROM facts").fetchone()
        start_fact_id = result[0]
        
        # Prepare insert data
        fact_ids = []
        insert_data = []
        
        for i, fact in enumerate(facts):
            fact_id = start_fact_id + i
            fact_ids.append(fact_id)
            
            # Get fact details with defaults
            entity_id = fact['entity_id']
            fact_text = fact['fact_text']
            source_url = fact.get('source_url')
            source_name = fact.get('source_name')
            timestamp_published = fact.get('timestamp_published', datetime.utcnow())
            
            insert_data.append((
                fact_id, 
                entity_id, 
                fact_text, 
                self._serialize_vector(all_embeddings[i]), 
                source_url, 
                source_name, 
                datetime.utcnow(), 
                timestamp_published, 
                hash_signatures[i]
            ))
        
        # Batch insert into database
        self.conn.executemany("""
            INSERT INTO facts (
                fact_id, entity_id, fact_text, fact_vector, source_url, 
                source_name, timestamp_captured, timestamp_published, hash_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, insert_data)
        
        return fact_ids
    
    def is_novel_fact(self, entity_id: int, fact_text: str, similarity_threshold: float = 0.85) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Check if a fact is novel compared to existing facts for an entity
        
        Args:
            entity_id: ID of the entity to check against
            fact_text: The fact text to check
            similarity_threshold: Threshold above which facts are considered similar (0.85 is a good default for 
                                 OpenAI embeddings, may need adjustment based on testing)
            
        Returns:
            is_novel: Boolean indicating whether the fact is novel
            similar_facts: List of similar facts if not novel
        """
        # First, check for exact matches using hash
        hash_sig = self._generate_hash(fact_text)
        
        # Query for exact matches
        exact_matches = self.conn.execute("""
            SELECT * FROM facts 
            WHERE entity_id = ? AND hash_signature = ?
        """, (entity_id, hash_sig)).fetchall()
        
        if exact_matches:
            # Convert to dictionary for consistency
            columns = [col[0] for col in self.conn.description]
            exact_match_dict = {columns[i]: exact_matches[0][i] for i in range(len(columns))}
            return False, [exact_match_dict]
        
        # If no exact match, check for semantic similarity
        fact_vector = self._generate_embedding(fact_text)
        
        # Get facts for this entity
        entity_facts = self.conn.execute("""
            SELECT * FROM facts 
            WHERE entity_id = ?
        """, (entity_id,)).fetchall()
        
        if not entity_facts:
            return True, []
        
        # Convert to list of dictionaries
        columns = [col[0] for col in self.conn.description]
        entity_facts_dicts = []
        for fact in entity_facts:
            fact_dict = {columns[i]: fact[i] for i in range(len(columns))}
            entity_facts_dicts.append(fact_dict)
        
        # Compare embeddings
        similar_facts = []
        for existing_fact in entity_facts_dicts:
            existing_vector = self._deserialize_vector(existing_fact['fact_vector'])
            similarity = self._calculate_similarity(fact_vector, existing_vector)
            
            if similarity > similarity_threshold:
                similar_facts.append({
                    'fact': existing_fact,
                    'similarity': similarity
                })
        
        if similar_facts:
            # Sort by similarity
            similar_facts.sort(key=lambda x: x['similarity'], reverse=True)
            return False, [item['fact'] for item in similar_facts]
        
        return True, []
    
    def check_facts_novelty_batch(self, facts: List[Dict[str, Any]], 
                                similarity_threshold: float = 0.85) -> List[Tuple[bool, List[Dict[str, Any]]]]:
        """
        Check if multiple facts are novel compared to existing facts for their respective entities
        
        Args:
            facts: List of fact dictionaries, each containing:
                - entity_id: ID of the entity this fact relates to
                - fact_text: The text of the fact
            similarity_threshold: Threshold above which facts are considered similar
            
        Returns:
            List of tuples containing:
                - is_novel: Boolean indicating whether the fact is novel
                - similar_facts: List of similar facts if not novel
        """
        results = []
        
        # Group facts by entity_id to minimize database queries
        entity_facts_map = {}
        for fact in facts:
            entity_id = fact['entity_id']
            if entity_id not in entity_facts_map:
                entity_facts_map[entity_id] = []
            entity_facts_map[entity_id].append(fact['fact_text'])
        
        # Fetch existing facts for all entities at once
        entity_ids = list(entity_facts_map.keys())
        placeholders = ','.join(['?'] * len(entity_ids))
        
        if not entity_ids:
            return []
            
        query = f"SELECT * FROM facts WHERE entity_id IN ({placeholders})"
        db_facts = self.conn.execute(query, entity_ids).fetchall()
        
        # Group existing facts by entity_id
        columns = [col[0] for col in self.conn.description]
        existing_facts_by_entity = {}
        
        for fact in db_facts:
            fact_dict = {columns[i]: fact[i] for i in range(len(columns))}
            entity_id = fact_dict['entity_id']
            
            if entity_id not in existing_facts_by_entity:
                existing_facts_by_entity[entity_id] = []
                
            existing_facts_by_entity[entity_id].append(fact_dict)
        
        # Process each fact
        for fact in facts:
            entity_id = fact['entity_id']
            fact_text = fact['fact_text']
            
            # Check for hash match first
            hash_sig = self._generate_hash(fact_text)
            
            exact_match = False
            exact_match_facts = []
            
            if entity_id in existing_facts_by_entity:
                for existing_fact in existing_facts_by_entity[entity_id]:
                    if existing_fact['hash_signature'] == hash_sig:
                        exact_match = True
                        exact_match_facts.append(existing_fact)
                        break
            
            if exact_match:
                results.append((False, exact_match_facts))
                continue
            
            # If no exact match, check for semantic similarity
            if entity_id not in existing_facts_by_entity:
                # No facts for this entity
                results.append((True, []))
                continue
                
            # Generate embedding for this fact
            fact_vector = self._generate_embedding(fact_text)
            
            # Compare with existing facts
            similar_facts = []
            for existing_fact in existing_facts_by_entity[entity_id]:
                existing_vector = self._deserialize_vector(existing_fact['fact_vector'])
                similarity = self._calculate_similarity(fact_vector, existing_vector)
                
                if similarity > similarity_threshold:
                    similar_facts.append({
                        'fact': existing_fact,
                        'similarity': similarity
                    })
            
            if similar_facts:
                # Sort by similarity
                similar_facts.sort(key=lambda x: x['similarity'], reverse=True)
                results.append((False, [item['fact'] for item in similar_facts]))
            else:
                results.append((True, []))
        
        return results
    
    def get_facts_for_entity(self, entity_id: int, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all facts for an entity, optionally filtered by recency
        
        Args:
            entity_id: Entity ID to fetch facts for
            days_back: If provided, only return facts from last X days
            
        Returns:
            facts: List of facts for the entity
        """
        query = "SELECT * FROM facts WHERE entity_id = ?"
        params = [entity_id]
        
        if days_back:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            query += " AND timestamp_published >= ?"
            params.append(cutoff_date)
            
        query += " ORDER BY timestamp_published DESC"
        
        # Execute query and fetch results
        result = self.conn.execute(query, params).fetchall()
        
        # Convert to list of dictionaries
        columns = [col[0] for col in self.conn.description]
        facts = []
        for row in result:
            fact_dict = {columns[i]: row[i] for i in range(len(columns))}
            facts.append(fact_dict)
            
        return facts
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate vector embedding for text using OpenAI's embedding service
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            np.ndarray: Vector embedding
        """
        return self.embedding_service.get_embedding(text)
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate vector embeddings for multiple texts at once
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List[np.ndarray]: List of vector embeddings
        """
        return self.embedding_service.get_embeddings(texts)
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Ensure vectors are normalized for more accurate cosine similarity
        vec1_normalized = vec1 / np.linalg.norm(vec1)
        vec2_normalized = vec2 / np.linalg.norm(vec2)
        
        return cosine_similarity([vec1_normalized], [vec2_normalized])[0][0]
    
    def _generate_hash(self, text: str) -> str:
        """
        Generate a hash signature for text
        
        Args:
            text: Text to hash
            
        Returns:
            str: SHA-256 hash hexdigest
        """
        # Normalize text before hashing (lowercase, remove extra whitespace)
        normalized_text = ' '.join(text.lower().split())
        return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
    
    def _serialize_vector(self, vector: np.ndarray) -> str:
        """
        Serialize vector for storage
        
        Args:
            vector: Numpy array to serialize
            
        Returns:
            str: Comma-separated string of vector values
        """
        return ','.join(str(x) for x in vector)
    
    def _deserialize_vector(self, serialized_vector: str) -> np.ndarray:
        """
        Deserialize vector from storage
        
        Args:
            serialized_vector: Comma-separated string of vector values
            
        Returns:
            np.ndarray: Deserialized vector
        """
        return np.array([float(x) for x in serialized_vector.split(',')])
