
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

class Fact:
    """Representation of a fact about an entity"""
    
    def __init__(
        self,
        fact_id: Optional[int] = None,
        entity_id: int = None,
        fact_text: str = None,
        embedding: Optional[np.ndarray] = None,
        source_url: Optional[str] = None,
        source_name: Optional[str] = None,
        timestamp_captured: Optional[datetime] = None,
        timestamp_published: Optional[datetime] = None,
        confidence_score: float = 1.0,
        hash_signature: Optional[str] = None
    ):
        self.fact_id = fact_id
        self.entity_id = entity_id
        self.fact_text = fact_text
        self.embedding = embedding
        self.source_url = source_url
        self.source_name = source_name
        self.timestamp_captured = timestamp_captured or datetime.utcnow()
        self.timestamp_published = timestamp_published or datetime.utcnow()
        self.confidence_score = confidence_score
        self.hash_signature = hash_signature
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        """Create a Fact object from a dictionary"""
        return cls(
            fact_id=data.get('fact_id'),
            entity_id=data.get('entity_id'),
            fact_text=data.get('fact_text'),
            embedding=data.get('embedding'),
            source_url=data.get('source_url'),
            source_name=data.get('source_name'),
            timestamp_captured=data.get('timestamp_captured'),
            timestamp_published=data.get('timestamp_published'),
            confidence_score=data.get('confidence_score', 1.0),
            hash_signature=data.get('hash_signature')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'fact_id': self.fact_id,
            'entity_id': self.entity_id,
            'fact_text': self.fact_text,
            'source_url': self.source_url,
            'source_name': self.source_name,
            'timestamp_captured': self.timestamp_captured,
            'timestamp_published': self.timestamp_published,
            'confidence_score': self.confidence_score,
            'hash_signature': self.hash_signature
        }