from novelty_filter.facts.fact_comparison import FactComparisonSystem
from novelty_filter.embeddings.openai_embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import time
import argparse

# Load environment variables from .env file
load_dotenv()

def process_facts_batch(batch_size=10, demo_mode=True):
    """
    Process facts in batch mode for improved performance
    
    Args:
        batch_size: Number of facts to process in each batch
        demo_mode: If True, runs in demonstration mode with sample data
    """
    print(f"Starting batch processing with batch size: {batch_size}")
    
    # Initialize services
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    embedding_service = OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small"  # Use smaller model for cost efficiency
    )
    
    fact_system = FactComparisonSystem(
        db_path="facts.duckdb",
        embedding_service=embedding_service
    )
    
    # Demo/sample data
    if demo_mode:
        # Create sample entities if they don't exist
        entities = [
            {"entity_id": 1, "entity_name": "Acme Corp", "entity_type": "company"},
            {"entity_id": 2, "entity_name": "TechGiant Inc", "entity_type": "company"},
            {"entity_id": 3, "entity_name": "GlobalBank", "entity_type": "financial"}
        ]
        
        # Check if entities exist, if not create them
        for entity in entities:
            check = fact_system.conn.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_id = ?", 
                (entity["entity_id"],)
            ).fetchone()[0]
            
            if check == 0:
                fact_system.conn.execute(
                    "INSERT INTO entities (entity_id, entity_name, entity_type) VALUES (?, ?, ?)",
                    (entity["entity_id"], entity["entity_name"], entity["entity_type"])
                )
                print(f"Created entity: {entity['entity_name']}")
        
        # Sample facts to process
        sample_facts = [
            {"entity_id": 1, "fact_text": "Acme Corp reported Q3 revenue of $1.2B.", "source_name": "Financial Times"},
            {"entity_id": 1, "fact_text": "Acme Corp announced a new CEO starting next month.", "source_name": "Business Wire"},
            {"entity_id": 2, "fact_text": "TechGiant Inc launched a new AI product today.", "source_name": "Tech News"},
            {"entity_id": 2, "fact_text": "TechGiant's stock price increased by 5% following product announcement.", "source_name": "Market Watch"},
            {"entity_id": 3, "fact_text": "GlobalBank announced a new sustainable finance initiative.", "source_name": "Banking Daily"},
            {"entity_id": 1, "fact_text": "Acme Corporation's quarterly revenue reached $1.2 billion in Q3.", "source_name": "Reuters"},
            {"entity_id": 2, "fact_text": "TechGiant released their cutting-edge artificial intelligence solution.", "source_name": "TechCrunch"},
            {"entity_id": 3, "fact_text": "GlobalBank expands green financing program with $500M commitment.", "source_name": "Financial News"},
            {"entity_id": 3, "fact_text": "GlobalBank reports 12% increase in sustainable lending.", "source_name": "ESG Today"},
            {"entity_id": 1, "fact_text": "Acme Corp plans to expand operations in Asia.", "source_name": "Asia Business Review"}
        ]
        
        # Process facts in batches
        print("\n--- BATCH NOVELTY CHECKING ---")
        start_time = time.time()
        
        # Check novelty for all facts in batch
        novelty_results = fact_system.check_facts_novelty_batch(sample_facts)
        
        # Process results and display
        print(f"Processed {len(sample_facts)} facts in {time.time() - start_time:.2f} seconds")
        
        for i, (is_novel, similar_facts) in enumerate(novelty_results):
            fact = sample_facts[i]
            print(f"\nFact: {fact['fact_text']}")
            print(f"Novel: {is_novel}")
            
            if not is_novel and similar_facts:
                print("Similar existing facts:")
                for similar in similar_facts:
                    print(f" - {similar['fact_text']} (similarity: {similar.get('similarity', 'exact match')})")
        
        # Now add all novel facts to the database in batch
        print("\n--- BATCH ADDING NOVEL FACTS ---")
        start_time = time.time()
        
        # Filter only novel facts
        novel_facts = [
            fact for i, fact in enumerate(sample_facts) 
            if novelty_results[i][0]  # is_novel is True
        ]
        
        if novel_facts:
            # Add timestamp and other optional fields
            for fact in novel_facts:
                fact['timestamp_published'] = datetime.utcnow()
                
            # Add novel facts in batch
            fact_ids = fact_system.add_facts_batch(novel_facts)
            
            print(f"Added {len(fact_ids)} novel facts to database in {time.time() - start_time:.2f} seconds")
            for i, fact_id in enumerate(fact_ids):
                print(f" - Fact ID {fact_id}: {novel_facts[i]['fact_text'][:50]}...")
        else:
            print("No novel facts to add.")
        
    else:
        # Real data processing from a file
        # This section would load from a JSON file or other data source
        try:
            with open("facts_to_process.json", "r") as f:
                facts_data = json.load(f)
                
            total_facts = len(facts_data)
            print(f"Loaded {total_facts} facts to process")
            
            # Process in batches
            for i in range(0, total_facts, batch_size):
                batch = facts_data[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(total_facts+batch_size-1)//batch_size}")
                
                # Check novelty
                novelty_results = fact_system.check_facts_novelty_batch(batch)
                
                # Filter novel facts
                novel_facts = [
                    fact for j, fact in enumerate(batch) 
                    if novelty_results[j][0]  # is_novel is True
                ]
                
                if novel_facts:
                    # Add novel facts to database
                    fact_ids = fact_system.add_facts_batch(novel_facts)
                    print(f"Added {len(fact_ids)} novel facts to database")
                
                # Optional: Save results to output file
                # with open(f"batch_results_{i//batch_size}.json", "w") as f:
                #     json.dump([{"fact": batch[j], "is_novel": result[0]} for j, result in enumerate(novelty_results)], f, indent=2)
                
        except FileNotFoundError:
            print("Error: facts_to_process.json not found. Run in demo mode instead.")
            return process_facts_batch(batch_size, demo_mode=True)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process facts in batch mode for novelty detection')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of facts to process in each batch')
    parser.add_argument('--real-data', action='store_true', help='Process real data instead of demo data')
    parser.add_argument('--input-file', type=str, default='facts_to_process.json', 
                        help='JSON file containing facts to process (only used with --real-data)')
    
    args = parser.parse_args()
    
    # Run with parsed arguments
    process_facts_batch(batch_size=args.batch_size, demo_mode=not args.real_data)
    
    # You can also customize the input file if using real data
    # if args.real_data:
    #     process_facts_with_file(args.input_file, args.batch_size)