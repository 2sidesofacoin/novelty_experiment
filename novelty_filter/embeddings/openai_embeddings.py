
import numpy as np
import openai
import time
import os
from typing import List

class OpenAIEmbeddings:
    def __init__(self, api_key=None, model="text-embedding-3-small"):
        """
        Initialize the OpenAI embeddings service
        
        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model to use
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        normalized_text = ' '.join(text.lower().split())
        return self._call_api_with_retry([normalized_text])[0]
    
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        normalized_texts = [' '.join(text.lower().split()) for text in texts]
        return self._call_api_with_retry(normalized_texts)
    
    def _call_api_with_retry(self, texts: List[str], max_retries=3, backoff_factor=2):
        """Call OpenAI API with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                
                # Extract embeddings in the same order
                embeddings = []
                for item in response.data:
                    embeddings.append(np.array(item.embedding))
                    
                return embeddings
            
            except (openai.RateLimitError, openai.APIError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise
                
                # Exponential backoff
                sleep_time = backoff_factor ** attempt
                print(f"OpenAI API error: {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)