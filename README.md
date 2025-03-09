# Novelty Filter

A system for detecting and filtering novel facts about entities using vector embeddings and semantic similarity.

## Overview

The Novelty Filter is designed to identify whether new information about an entity is truly novel or just a rephrasing of existing knowledge. The system uses semantic understanding through vector embeddings to compare incoming facts with those already stored in a database.

Key features:
- Semantic similarity detection using OpenAI embeddings
- Fast exact matching using hash signatures
- Batch processing for improved performance
- Comprehensive evaluation framework
- DuckDB for efficient local storage

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key for embeddings

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/novelty-filter.git
cd novelty-filter
```

2. Install the package:
```bash
pip install -e .
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from novelty_filter.facts.fact_comparison import FactComparisonSystem
from novelty_filter.embeddings.openai_embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize services
api_key = os.environ.get("OPENAI_API_KEY")
embedding_service = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
fact_system = FactComparisonSystem(db_path="facts.duckdb", embedding_service=embedding_service)

# Check if a fact is novel
is_novel, similar_facts = fact_system.is_novel_fact(
    entity_id=1,  # Entity ID to check against
    fact_text="Acme Corp reported Q3 revenue of $1.2B.",
    similarity_threshold=0.85  # Optional, default is 0.85
)

print(f"Is the fact novel? {is_novel}")
if not is_novel:
    print("Similar existing facts:")
    for fact in similar_facts:
        print(f" - {fact['fact_text']}")
```

### Batch Processing

For improved performance when processing multiple facts:

```python
# Prepare facts to check
facts_to_check = [
    {"entity_id": 1, "fact_text": "Acme Corp reported Q3 revenue of $1.2B."},
    {"entity_id": 2, "fact_text": "TechGiant Inc launched a new AI product."}
]

# Check novelty for all facts in batch
novelty_results = fact_system.check_facts_novelty_batch(facts_to_check)

# Process results
for i, (is_novel, similar_facts) in enumerate(novelty_results):
    fact = facts_to_check[i]
    print(f"\nFact: {fact['fact_text']}")
    print(f"Novel: {is_novel}")
    
    if not is_novel and similar_facts:
        print("Similar existing facts:")
        for similar in similar_facts:
            print(f" - {similar['fact_text']}")
```

### Running the Demonstration

The repository includes a demo script that shows the system in action:

```bash
python -m novelty_filter.batch_processing_script --batch-size 10
```

## Evaluation

The system includes a comprehensive evaluation framework to measure performance:

```bash
python -m novelty_filter.evaluation.evaluate_cli --create-dataset
```

This will:
1. Create a test dataset with facts of varying similarity
2. Test the system at different similarity thresholds (0.75-0.95)
3. Generate metrics (precision, recall, F1 score)
4. Save results and visualizations to the `evaluation_results` directory

## Repository Structure

```
novelty_filter/
├── __init__.py
├── main.py                      # Main entry point example
├── batch_processing_script.py   # Batch processing demonstration
├── config/                      # Configuration settings
├── db/                          # Database connectivity
│   ├── connection.py            # DuckDB connection handling
│   └── schema.py                # Database schema definition
├── embeddings/                  # Vector embedding services
│   └── openai_embeddings.py     # OpenAI embeddings implementation
├── entities/                    # Entity management
├── evaluation/                  # Evaluation framework
│   ├── create_test_set.py       # Test data generation
│   ├── evaluate_cli.py          # CLI for evaluation
│   └── run_evaluation.py        # Evaluation logic
└── facts/                       # Core fact handling
    ├── fact.py                  # Fact class definition
    └── fact_comparison.py       # Similarity comparison logic
```

## Performance Tuning

The system's performance can be tuned by adjusting the similarity threshold:

- Higher threshold (e.g., 0.90-0.95): More sensitive, may generate more false positives (identify facts as novel when they are actually similar to existing ones)
- Lower threshold (e.g., 0.75-0.80): More conservative, may generate more false negatives (miss truly novel facts)

Based on evaluation results, a threshold of 0.80-0.85 typically provides the best balance of precision and recall.

## Advanced Usage

### Custom Embedding Service

You can implement your own embedding service by creating a class that provides these methods:
- `get_embedding(text: str) -> np.ndarray`
- `get_embeddings(texts: List[str]) -> List[np.ndarray]`

```python
from novelty_filter.facts.fact_comparison import FactComparisonSystem

# Initialize with your custom embedding service
fact_system = FactComparisonSystem(
    db_path="facts.duckdb",
    embedding_service=your_custom_embedding_service
)
```

## License

[MIT License](LICENSE)

## Acknowledgments

This project uses:
- OpenAI for vector embeddings
- DuckDB for efficient local database storage
- scikit-learn for vector similarity computations
