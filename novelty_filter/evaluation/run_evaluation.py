
import json
import csv
import time
import os
from typing import Dict, List, Tuple, Any
import numpy as np
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from novelty_filter.facts.fact_comparison import FactComparisonSystem
from novelty_filter.embeddings.openai_embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class NoveltyEvaluator:
    def __init__(self, db_path="evaluation.duckdb"):
        """Initialize the evaluator with a clean test database"""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.embedding_service = OpenAIEmbeddings(
            api_key=self.api_key,
            model="text-embedding-3-small"
        )
        
        # Use a separate database for evaluation
        self.fact_system = FactComparisonSystem(
            db_path=db_path,
            embedding_service=self.embedding_service
        )
        
        # Initialize database if needed
        self._initialize_database()
    
    def _initialize_database(self):
        """Create necessary tables in the database"""
        # Check if entities table exists
        tables = self.fact_system.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entities'"
        ).fetchall()
        
        if not tables:
            # Create entities table
            self.fact_system.conn.execute("""
                CREATE TABLE entities (
                    entity_id INTEGER PRIMARY KEY,
                    entity_name TEXT NOT NULL,
                    entity_type TEXT
                )
            """)
            
            # Create facts table
            self.fact_system.conn.execute("""
                CREATE TABLE facts (
                    fact_id INTEGER PRIMARY KEY,
                    entity_id INTEGER,
                    fact_text TEXT NOT NULL,
                    fact_vector TEXT,
                    source_url TEXT,
                    source_name TEXT,
                    timestamp_captured TIMESTAMP,
                    timestamp_published TIMESTAMP,
                    hash_signature TEXT,
                    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
                )
            """)
            print("Initialized empty database for evaluation")
    
    def run_evaluation(self, dataset_path="evaluation_data.json", 
                       threshold_values=None):
        """
        Run the novelty detection evaluation on the dataset
        
        Args:
            dataset_path: Path to the evaluation dataset
            threshold_values: List of threshold values to test
        """
        # Default threshold values if none provided
        if threshold_values is None:
            threshold_values = [0.75, 0.80, 0.85, 0.90, 0.95]
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            evaluation_data = json.load(f)
        
        # Create entities
        self._create_entities(evaluation_data)
        
        # Results for all thresholds
        all_results = {}
        
        # Run evaluation at different threshold values
        for threshold in threshold_values:
            print(f"\n--- Evaluating with similarity threshold: {threshold} ---")
            
            # Reset database for each threshold evaluation
            self._reset_facts_table()
            
            results = self._evaluate_with_threshold(evaluation_data, threshold)
            all_results[threshold] = results
            
            # Print metrics for this threshold
            self._print_metrics(results)
        
        # Find the best threshold based on F1 score
        best_threshold = max(all_results.items(), key=lambda x: x[1]['metrics']['f1'])
        print(f"\n--- Best threshold: {best_threshold[0]} with F1: {best_threshold[1]['metrics']['f1']:.4f} ---")
        
        # Generate visualizations
        self._generate_visualizations(all_results)
        
        return all_results
    
    def _create_entities(self, evaluation_data):
        """Create entities in the database from the evaluation data"""
        # Extract unique entities
        entities = {}
        for item in evaluation_data:
            entity_id = item['entity_id']
            if entity_id not in entities:
                entities[entity_id] = {
                    'entity_id': entity_id,
                    'entity_name': item['entity_name']
                }
        
        # Add entities to database
        for entity in entities.values():
            # Check if entity already exists
            exists = self.fact_system.conn.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_id = ?",
                (entity['entity_id'],)
            ).fetchone()[0]
            
            if not exists:
                self.fact_system.conn.execute(
                    "INSERT INTO entities (entity_id, entity_name) VALUES (?, ?)",
                    (entity['entity_id'], entity['entity_name'])
                )
        
        print(f"Created {len(entities)} entities in the database")
    
    def _reset_facts_table(self):
        """Clear all facts from the database"""
        self.fact_system.conn.execute("DELETE FROM facts")
        print("Reset facts table")
    
    def _evaluate_with_threshold(self, evaluation_data, threshold):
        """
        Evaluate novelty detection with a specific similarity threshold
        
        Args:
            evaluation_data: List of facts with expected novelty labels
            threshold: Similarity threshold for novelty detection
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        results = {
            'threshold': threshold,
            'predictions': [],
            'metrics': {},
            'errors': {'false_positives': [], 'false_negatives': []},
            'processing_time': 0
        }
        
        true_labels = []  # 1 for novel, 0 for not novel
        predicted_labels = []  # 1 for predicted novel, 0 for predicted not novel
        
        # Process each fact in order
        for item in evaluation_data:
            # Get expected novelty
            expected_novel = item['expected_novel']
            true_labels.append(1 if expected_novel else 0)
            
            # Check if the system thinks it's novel
            is_novel, similar_facts = self.fact_system.is_novel_fact(
                entity_id=item['entity_id'],
                fact_text=item['fact_text'],
                similarity_threshold=threshold
            )
            predicted_labels.append(1 if is_novel else 0)
            
            # Record the result
            result = {
                'fact_text': item['fact_text'],
                'entity_id': item['entity_id'],
                'expected_novel': expected_novel,
                'predicted_novel': is_novel,
                'correct': expected_novel == is_novel,
                'similar_facts': [f['fact_text'] for f in similar_facts] if similar_facts else []
            }
            results['predictions'].append(result)
            
            # Track errors
            if expected_novel and not is_novel:
                # False negative - should be novel but predicted as not novel
                results['errors']['false_negatives'].append(result)
            elif not expected_novel and is_novel:
                # False positive - should not be novel but predicted as novel
                results['errors']['false_positives'].append(result)
            
            # If the fact was identified as novel, add it to the database
            if is_novel:
                self.fact_system.add_fact(
                    entity_id=item['entity_id'],
                    fact_text=item['fact_text']
                )
        
        # Calculate metrics
        results['metrics'] = {
            'precision': precision_score(true_labels, predicted_labels),
            'recall': recall_score(true_labels, predicted_labels),
            'f1': f1_score(true_labels, predicted_labels),
            'accuracy': sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels),
            'confusion_matrix': confusion_matrix(true_labels, predicted_labels).tolist()
        }
        
        # Calculate processing time
        results['processing_time'] = time.time() - start_time
        
        return results
    
    def _print_metrics(self, results):
        """Print evaluation metrics"""
        metrics = results['metrics']
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        
        # Print confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        print("\nConfusion Matrix:")
        print("                  Predicted Not Novel  Predicted Novel")
        print(f"Actually Not Novel      {cm[0][0]}                {cm[0][1]}")
        print(f"Actually Novel         {cm[1][0]}                {cm[1][1]}")
        
        # Print error counts
        print(f"\nFalse Positives: {len(results['errors']['false_positives'])}")
        print(f"False Negatives: {len(results['errors']['false_negatives'])}")
    
    def _generate_visualizations(self, all_results):
        """Generate visualizations of the evaluation results"""
        # Create results directory if it doesn't exist
        os.makedirs("evaluation_results", exist_ok=True)
        
        # Extract metrics for different thresholds
        thresholds = list(all_results.keys())
        precisions = [all_results[t]['metrics']['precision'] for t in thresholds]
        recalls = [all_results[t]['metrics']['recall'] for t in thresholds]
        f1_scores = [all_results[t]['metrics']['f1'] for t in thresholds]
        
        # Plot metrics vs threshold
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, 'o-', label='Precision')
        plt.plot(thresholds, recalls, 'o-', label='Recall')
        plt.plot(thresholds, f1_scores, 'o-', label='F1 Score')
        plt.xlabel('Similarity Threshold')
        plt.ylabel('Score')
        plt.title('Novelty Detection Performance vs Similarity Threshold')
        plt.grid(True)
        plt.legend()
        plt.savefig('evaluation_results/threshold_performance.png')
        
        # Plot confusion matrix for best threshold
        best_threshold = max(all_results.items(), key=lambda x: x[1]['metrics']['f1'])[0]
        cm = np.array(all_results[best_threshold]['metrics']['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Novel', 'Novel'],
                   yticklabels=['Not Novel', 'Novel'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Threshold = {best_threshold})')
        plt.savefig('evaluation_results/confusion_matrix.png')
        
        # Save detailed results to CSV
        self._save_results_to_csv(all_results)
        print("\nSaved visualizations and results to 'evaluation_results' directory")
    
    def _save_results_to_csv(self, all_results):
        """Save detailed results to CSV files"""
        # Save overall metrics
        with open('evaluation_results/metrics_by_threshold.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Threshold', 'Precision', 'Recall', 'F1', 'Accuracy', 'Processing Time'])
            
            for threshold, results in all_results.items():
                metrics = results['metrics']
                writer.writerow([
                    threshold,
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1'],
                    metrics['accuracy'],
                    results['processing_time']
                ])
        
        # Save errors for the best threshold
        best_threshold = max(all_results.items(), key=lambda x: x[1]['metrics']['f1'])[0]
        best_results = all_results[best_threshold]
        
        with open('evaluation_results/false_positives.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Entity ID', 'Fact Text', 'Similar Facts'])
            
            for error in best_results['errors']['false_positives']:
                writer.writerow([
                    error['entity_id'],
                    error['fact_text'],
                    '; '.join(error['similar_facts'])
                ])
        
        with open('evaluation_results/false_negatives.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Entity ID', 'Fact Text', 'Similar Facts'])
            
            for error in best_results['errors']['false_negatives']:
                writer.writerow([
                    error['entity_id'],
                    error['fact_text'],
                    '; '.join(error['similar_facts'])
                ])

if __name__ == "__main__":
    evaluator = NoveltyEvaluator()
    evaluator.run_evaluation()