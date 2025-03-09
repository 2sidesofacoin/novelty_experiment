
import argparse
import os
from novelty_filter.evaluation.create_test_set import create_evaluation_dataset
from novelty_filter.evaluation.run_evaluation import NoveltyEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate novelty detection system")
    parser.add_argument("--create-dataset", action="store_true", help="Create a new evaluation dataset")
    parser.add_argument("--dataset-path", default="evaluation_data.json", help="Path to the evaluation dataset")
    parser.add_argument("--db-path", default="evaluation.duckdb", help="Path to the evaluation database")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.75, 0.80, 0.85, 0.90, 0.95],
                        help="Similarity thresholds to evaluate")
    
    args = parser.parse_args()
    
    # Create dataset if requested
    if args.create_dataset:
        print("Creating evaluation dataset...")
        create_evaluation_dataset(args.dataset_path)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Dataset not found at {args.dataset_path}. Creating a new one...")
        create_evaluation_dataset(args.dataset_path)
    
    # Run evaluation
    print(f"Running evaluation with thresholds: {args.thresholds}")
    evaluator = NoveltyEvaluator(db_path=args.db_path)
    results = evaluator.run_evaluation(args.dataset_path, args.thresholds)
    
    # Print overall results summary
    best_threshold = max(results.items(), key=lambda x: x[1]['metrics']['f1'])[0]
    best_metrics = results[best_threshold]['metrics']
    
    print("\n--- OVERALL RESULTS SUMMARY ---")
    print(f"Best similarity threshold: {best_threshold}")
    print(f"Best F1 score: {best_metrics['f1']:.4f}")
    print(f"Precision at best threshold: {best_metrics['precision']:.4f}")
    print(f"Recall at best threshold: {best_metrics['recall']:.4f}")
    print(f"Accuracy at best threshold: {best_metrics['accuracy']:.4f}")
    print("\nDetailed results saved to 'evaluation_results' directory")

if __name__ == "__main__":
    main()