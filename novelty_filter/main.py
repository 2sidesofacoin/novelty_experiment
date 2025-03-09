from novelty_filter.facts.fact_comparison import FactComparisonSystem
from novelty_filter.embeddings.openai_embeddings import OpenAIEmbeddings
import os
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    # Initialize services
    api_key = os.environ.get("OPENAI_API_KEY")
    
    embedding_service = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small"
    )
    
    fact_system = FactComparisonSystem(
        db_path="facts.duckdb",
        embedding_service=embedding_service
    )
    
    # Example usage
    # Add an entity
    entity_id = 1
    fact_system.conn.execute("""
        INSERT INTO entities (entity_id, entity_name, entity_type) 
        VALUES (?, ?, ?)
    """, (entity_id, "Acme Corp", "company"))
    
    # Add a fact
    fact_system.add_fact(
        entity_id=entity_id,
        fact_text="Acme Corp reported Q3 revenue of $1.2B, exceeding analyst expectations.",
        source_name="Financial Times",
        source_url="https://ft.com/example"
    )
    
    # Check if a new fact is novel
    is_novel, similar_facts = fact_system.is_novel_fact(
        entity_id=entity_id,
        fact_text="Acme Corporation announced third quarter revenues of $1.2 billion, which was above market expectations."
    )
    
    print(f"Is the fact novel? {is_novel}")
    if not is_novel:
        print("Similar existing facts:")
        for fact in similar_facts:
            print(f" - {fact['fact_text']}")

if __name__ == "__main__":
    main()