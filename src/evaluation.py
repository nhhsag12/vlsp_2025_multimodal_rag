import json
import os
import torch
import faiss
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any
from tqdm import tqdm


def extract_law_and_article_id(record_id: str) -> Tuple[str, str]:
    """Extract law_id and article_id from record_id"""
    try:
        parts = record_id.split('#')
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return None, None
    except:
        return None, None


def calculate_f2_score_per_sample(retrieved_record_ids: List[str], relevant_articles: List[Dict]) -> Dict:
    """Calculate F2 score for a single test sample"""
    # Convert relevant articles to set of (law_id, article_id) tuples
    relevant_set = set()
    for article in relevant_articles:
        relevant_set.add((article["law_id"], article["article_id"]))

    # Convert retrieved record IDs to set of (law_id, article_id) tuples
    retrieved_set = set()
    for record_id in retrieved_record_ids:
        law_id, article_id = extract_law_and_article_id(record_id)
        if law_id and article_id:
            retrieved_set.add((law_id, article_id))

    # Calculate metrics
    num_retrieved = len(retrieved_set)
    num_relevant = len(relevant_set)
    num_correct = len(retrieved_set.intersection(relevant_set))

    # Calculate precision and recall
    precision = num_correct / num_retrieved if num_retrieved > 0 else 0.0
    recall = num_correct / num_relevant if num_relevant > 0 else 0.0

    # Calculate F2 score
    if precision + recall > 0:
        f2_score = (5 * precision * recall) / (4 * precision + recall)
    else:
        f2_score = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f2_score": f2_score,
        "num_retrieved": num_retrieved,
        "num_relevant": num_relevant,
        "num_correct": num_correct
    }


def create_faiss_index(document_embeddings: Dict) -> Tuple[faiss.Index, List[str]]:
    """Create FAISS index from document embeddings"""
    record_id_to_embedding = {}
    for key, values in document_embeddings.items():
        if isinstance(values["embedding"], list):
            record_id_to_embedding[key] = torch.tensor(values["embedding"])
        else:
            record_id_to_embedding[key] = values["embedding"]

    # Get embedding dimension
    first_embedding = next(iter(record_id_to_embedding.values()))
    embedding_dim = first_embedding.shape[0]

    # Create FAISS index
    index = faiss.IndexFlatIP(embedding_dim)

    # Add all embeddings to the index
    embeddings_matrix = []
    record_ids = []
    for record_id, embedding in record_id_to_embedding.items():
        embeddings_matrix.append(embedding.detach().cpu().numpy())
        record_ids.append(record_id)

    embeddings_matrix = np.array(embeddings_matrix)
    index.add(embeddings_matrix.astype(np.float32))

    return index, record_ids


def evaluate_model(model, test_data: List[Dict], document_embeddings: Dict,
                   image_base_path: str, k: int = 5, similarity_threshold: float = 0.7,
                   device: torch.device = None) -> Dict[str, Any]:
    """
    Evaluate the retrieval model on test data

    Args:
        model: The trained retriever model
        test_data: List of test questions with ground truth
        document_embeddings: Dictionary mapping record_id to document info
        image_base_path: Base path to test images
        k: Number of top documents to retrieve
        similarity_threshold: Minimum similarity score threshold
        device: Device to run evaluation on

    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device

    # Create FAISS index
    index, record_ids = create_faiss_index(document_embeddings)

    # Set model to evaluation mode
    model.eval()

    results = []
    per_sample_metrics = []

    with torch.no_grad():
        for test_item in tqdm(test_data, desc="Evaluating"):
            test_id = test_item["id"]
            image_id = test_item["image_id"]
            question = test_item["question"]
            relevant_articles = test_item.get("relevant_articles", [])

            # Load and process image
            image_path = os.path.join(image_base_path, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                continue

            try:
                image = Image.open(image_path).convert('RGB')
                image = image.resize((224, 224))

                # Get multimodal embedding from retriever
                multimodal_embedding = model(image, question)

                if isinstance(multimodal_embedding, np.ndarray):
                    query_embedding = multimodal_embedding.reshape(1, -1)
                else:
                    query_embedding = multimodal_embedding.detach().cpu().numpy().reshape(1, -1)

                # Search in FAISS index
                scores, indices = index.search(query_embedding.astype(np.float32), k)

                # Filter by similarity threshold and get results
                top_results = []
                retrieved_record_ids = []

                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if score > similarity_threshold:
                        record_id = record_ids[idx]
                        top_results.append({
                            "rank": i + 1,
                            "record_id": record_id,
                            "score": float(score)
                        })
                        retrieved_record_ids.append(record_id)

                # Calculate metrics for this sample
                sample_metrics = calculate_f2_score_per_sample(retrieved_record_ids, relevant_articles)
                sample_metrics["test_id"] = test_id
                per_sample_metrics.append(sample_metrics)

                results.append({
                    "test_id": test_id,
                    "image_id": image_id,
                    "question": question,
                    "top_results": top_results,
                    "num_retrieved": len(retrieved_record_ids)
                })

            except Exception as e:
                print(f"Error processing {test_id}: {str(e)}")
                continue

    # Calculate overall metrics
    if per_sample_metrics:
        avg_precision = np.mean([m["precision"] for m in per_sample_metrics])
        avg_recall = np.mean([m["recall"] for m in per_sample_metrics])
        avg_f2 = np.mean([m["f2_score"] for m in per_sample_metrics])
    else:
        avg_precision = avg_recall = avg_f2 = 0.0

    # Calculate coverage statistics
    cases_with_results = sum(1 for r in results if r["num_retrieved"] > 0)
    total_retrieved = sum(r["num_retrieved"] for r in results)

    evaluation_results = {
        "overall_metrics": {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f2_score": avg_f2,
            "samples_processed": len(per_sample_metrics),
            "total_samples": len(test_data),
            "cases_with_results": cases_with_results,
            "coverage_rate": cases_with_results / len(results) if results else 0.0,
            "total_retrieved": total_retrieved,
            "avg_retrieved_per_case": total_retrieved / len(results) if results else 0.0
        },
        "per_sample_metrics": per_sample_metrics,
        "detailed_results": results
    }

    return evaluation_results


def load_evaluation_data():
    """Load evaluation data (test set and document embeddings)"""
    # Load public test data
    public_test_path = "data/VLSP 2025 - MLQA-TSR Data Release/public_test/vlsp_2025_public_test_task2.json"
    with open(public_test_path, "r") as f:
        public_test = json.load(f)

    # Load document embeddings
    document_embedding_path = "data/record_id_to_document_embedding.json"
    with open(document_embedding_path, "r") as f:
        document_embeddings = json.load(f)

    # Image base path
    image_base_path = "data/VLSP 2025 - MLQA-TSR Data Release/public_test/public_test_images/public_test_images"

    return public_test, document_embeddings, image_base_path


def print_evaluation_summary(eval_results: Dict[str, Any], epoch: int = None):
    """Print a summary of evaluation results"""
    metrics = eval_results["overall_metrics"]

    header = f"EVALUATION RESULTS" + (f" - EPOCH {epoch}" if epoch is not None else "")
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    print(f"Samples processed: {metrics['samples_processed']}/{metrics['total_samples']}")
    print(f"Cases with results: {metrics['cases_with_results']} ({metrics['coverage_rate'] * 100:.1f}%)")
    print(f"Average documents retrieved per case: {metrics['avg_retrieved_per_case']:.2f}")
    print(f"Average Precision: {metrics['avg_precision']:.4f}")
    print(f"Average Recall: {metrics['avg_recall']:.4f}")
    print(f"Average F2 Score: {metrics['avg_f2_score']:.4f}")
    print(f"{'=' * 60}")