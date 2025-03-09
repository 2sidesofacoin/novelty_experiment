
import json
import csv
from typing import List, Dict, Any
import random

def create_evaluation_dataset(output_path="evaluation_data.json"):
    """
    Creates a gold standard dataset for evaluating the novelty detection system.
    
    The dataset contains:
    - Original facts
    - Exact duplicates
    - Near duplicates (paraphrased)
    - Similar but novel facts (same topic but new info)
    - Completely novel facts
    
    Each fact is labeled with its expected novelty classification.
    """
    # Base facts that we'll use to create variants
    base_facts = [
        {
            "entity_id": 1,
            "entity_name": "Acme Corp",
            "fact_text": "Acme Corp reported annual revenue of $10.5B for 2023.",
            "expected_novel": True  # First occurrence should be novel
        },
        {
            "entity_id": 1, 
            "entity_name": "Acme Corp",
            "fact_text": "Acme Corp appointed Jane Smith as the new CEO.",
            "expected_novel": True
        },
        {
            "entity_id": 2,
            "entity_name": "TechGiant Inc",
            "fact_text": "TechGiant Inc launched its new AI platform yesterday.",
            "expected_novel": True
        },
        {
            "entity_id": 3,
            "entity_name": "GlobalBank",
            "fact_text": "GlobalBank announced a 15% increase in quarterly profits.",
            "expected_novel": True
        }
    ]
    
    # Create evaluation dataset
    evaluation_data = []
    
    # Add original facts first (these are novel)
    evaluation_data.extend(base_facts)
    
    # Add exact duplicates (not novel)
    for fact in base_facts:
        duplicate = fact.copy()
        duplicate["expected_novel"] = False  # Duplicate should not be novel
        duplicate["duplicate_type"] = "exact"
        duplicate["original_fact"] = fact["fact_text"]
        evaluation_data.append(duplicate)
    
    # Add near duplicates - paraphrases (not novel)
    paraphrases = [
        {
            "original": "Acme Corp reported annual revenue of $10.5B for 2023.",
            "paraphrase": "In 2023, Acme Corporation's yearly revenue totaled $10.5 billion."
        },
        {
            "original": "Acme Corp appointed Jane Smith as the new CEO.",
            "paraphrase": "Jane Smith has been named the new Chief Executive Officer of Acme Corp."
        },
        {
            "original": "TechGiant Inc launched its new AI platform yesterday.",
            "paraphrase": "Yesterday, TechGiant Incorporated released their newest artificial intelligence platform."
        },
        {
            "original": "GlobalBank announced a 15% increase in quarterly profits.",
            "paraphrase": "GlobalBank reported that their quarterly profits have risen by 15 percent."
        }
    ]
    
    for i, base_fact in enumerate(base_facts):
        paraphrase = base_fact.copy()
        paraphrase["fact_text"] = paraphrases[i]["paraphrase"]
        paraphrase["expected_novel"] = False
        paraphrase["duplicate_type"] = "paraphrase"
        paraphrase["original_fact"] = base_fact["fact_text"]
        evaluation_data.append(paraphrase)
    
    # Add similar but novel facts (same topic but new information)
    similar_but_novel = [
        {
            "entity_id": 1,
            "entity_name": "Acme Corp",
            "fact_text": "Acme Corp's annual revenue reached $11.2B for 2024, up 7% from last year.",
            "expected_novel": True,
            "related_to": "Acme Corp reported annual revenue of $10.5B for 2023."
        },
        {
            "entity_id": 1,
            "entity_name": "Acme Corp",
            "fact_text": "Jane Smith outlined a new 5-year growth strategy for Acme Corp during her first shareholder meeting.",
            "expected_novel": True,
            "related_to": "Acme Corp appointed Jane Smith as the new CEO."
        },
        {
            "entity_id": 2,
            "entity_name": "TechGiant Inc",
            "fact_text": "TechGiant Inc's AI platform has gained 1 million users in its first month.",
            "expected_novel": True,
            "related_to": "TechGiant Inc launched its new AI platform yesterday."
        },
        {
            "entity_id": 3,
            "entity_name": "GlobalBank",
            "fact_text": "GlobalBank has expanded its sustainable investment portfolio to $5B following strong quarterly results.",
            "expected_novel": True,
            "related_to": "GlobalBank announced a 15% increase in quarterly profits."
        }
    ]
    evaluation_data.extend(similar_but_novel)
    
    # Add completely different facts (novel)
    completely_novel = [
        {
            "entity_id": 1,
            "entity_name": "Acme Corp",
            "fact_text": "Acme Corp will open a new research center in Singapore next month.",
            "expected_novel": True
        },
        {
            "entity_id": 2,
            "entity_name": "TechGiant Inc", 
            "fact_text": "TechGiant Inc has acquired startup Quantum Computing Solutions for $500M.",
            "expected_novel": True
        },
        {
            "entity_id": 3,
            "entity_name": "GlobalBank",
            "fact_text": "GlobalBank has appointed Dr. Robert Chen as the new Chief Data Officer.",
            "expected_novel": True
        }
    ]
    evaluation_data.extend(completely_novel)
    
    # Shuffle the dataset
    random.shuffle(evaluation_data)
    
    # Save evaluation data
    with open(output_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"Created evaluation dataset with {len(evaluation_data)} examples at {output_path}")
    
    return evaluation_data

if __name__ == "__main__":
    create_evaluation_dataset()