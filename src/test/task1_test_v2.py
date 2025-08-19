import json
from copy import deepcopy

import torch
import faiss
import os
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple

# --- Import custom modules from your project ---
# Make sure the script is run from a location where these imports are valid
from src.multimodal_retriever.retriever import Retriever
from src.multimodal_retriever.retriever_v2 import RetrieverV2
from src.reranker.cross_encoder import CrossEncoder
from src.utils.utils import load_model


def setup_faiss_index(embedding_path: str) -> Tuple[faiss.Index, List[str], Dict]:
    """
    Loads document embeddings from a file and prepares a FAISS index for efficient search.

    Args:
        embedding_path: Path to the JSON file containing document embeddings.

    Returns:
        A tuple containing the FAISS index, a list of record IDs, and the embedding data as a dictionary.
    """
    print(f"Loading document embeddings from {embedding_path}...")
    with open(embedding_path, "r") as f:
        document_embeddings = json.load(f)

    # Convert embeddings from lists to torch tensors
    tensor_document_embeddings = {
        key: {"text": values["text"], "embedding": torch.tensor(values["embedding"])}
        for key, values in document_embeddings.items()
    }

    record_ids = list(tensor_document_embeddings.keys())
    embeddings_list = [v["embedding"].numpy() for v in tensor_document_embeddings.values()]

    if not embeddings_list:
        raise ValueError("No embeddings found to create a FAISS index.")

    embedding_dim = embeddings_list[0].shape[0]
    embeddings_matrix = np.vstack(embeddings_list)

    print(f"Creating FAISS index with {len(embeddings_list)} embeddings of dimension {embedding_dim}.")
    index = faiss.IndexFlatIP(embedding_dim)  # Using Inner Product for similarity
    index.add(embeddings_matrix)

    print(f"FAISS index created successfully. Total entries: {index.ntotal}")
    return index, record_ids, tensor_document_embeddings

def run_full_rag_pipeline(
        retriever: Retriever,
        test_data: List[Dict],
        faiss_index: faiss.Index,
        record_ids: List[str],
        doc_embeddings: Dict,
        config: Dict
) -> List[Dict]:
    """
    Executes the end-to-end RAG pipeline using only retrieval.

    Args:
        retriever: The trained retriever model.
        test_data: The list of test cases to process.
        faiss_index: The FAISS index for document search.
        record_ids: A list of record IDs corresponding to the FAISS index order.
        doc_embeddings: A dictionary mapping record IDs to their text and embeddings.
        config: A configuration dictionary with paths and parameters.

    Returns:
        A list of final results, with documents from the retrieval stage.
    """
    final_results = []
    retriever.eval()

    with torch.no_grad():
        for i, test_item in enumerate(test_data):
            test_id = test_item["id"]
            image_id = test_item["image_id"]
            question = test_item["question"]

            print(f"\nProcessing item {i + 1}/{len(test_data)} (ID: {test_id})...")

            # --- 1. Load Image ---
            image_path = os.path.join(config["image_base_path"], f"{image_id}.jpg")
            if not os.path.exists(image_path):
                print(f"  -> Warning: Image not found at {image_path}. Skipping.")
                continue

            query_image = Image.open(image_path)
            query_image = retriever.preprocess_val(query_image).unsqueeze(0)

            # --- 2. Retrieval Stage (This is now the final stage) ---
            print("  -> Stage 1: Retrieving initial documents.")
            query_embedding = retriever(texts=question, images=query_image).reshape(1, -1)
            scores, indices = faiss_index.search(query_embedding, config["retrieval_k"])

            retrieved_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if float(score) > 0.75: # Keep documents that meet a certain score threshold
                    record_id = record_ids[idx]
                    retrieved_docs.append({
                        "record_id": record_id,
                        "text": doc_embeddings[record_id]["text"],
                        "retriever_score": float(score)
                    })

            if not retrieved_docs:
                print("  -> No documents found in retrieval stage.")
                continue

            print(f"  -> Retrieved {len(retrieved_docs)} documents.")

            final_results.append({
                "id": test_id,
                "image_id": image_id,
                "question": question,
                "relevant_articles": [
                    {"law_id": doc["record_id"].split('#')[0], "article_id": doc["record_id"].split('#')[1]}
                    for doc in retrieved_docs
                ]
            })

    return final_results


def evaluate_f2_score(predictions: List[Dict], ground_truth: List[Dict]):
    """
    Calculates the average F2 score, precision, and recall for the given predictions against the ground truth.

    Args:
        predictions: A list of prediction dictionaries, each containing an 'id' and 'relevant_articles'.
        ground_truth: A list of ground truth dictionaries with the same structure as predictions.
    """
    ground_truth_map = {item['id']: item for item in ground_truth}

    total_f2 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    processed_samples = 0

    for pred_item in predictions:
        item_id = pred_item.get('id')
        if not item_id or item_id not in ground_truth_map:
            continue

        gt_item = ground_truth_map[item_id]

        predicted_articles = pred_item.get('relevant_articles', [])
        gt_articles = gt_item.get('relevant_articles', [])

        # Convert to sets of tuples for easy comparison
        pred_set = {(a['law_id'], a['article_id']) for a in predicted_articles}
        gt_set = {(a['law_id'], a['article_id']) for a in gt_articles}

        if not gt_set:  # if there are no ground truth articles, skip from evaluation
            continue

        processed_samples += 1

        num_correct = len(pred_set.intersection(gt_set))
        num_predicted = len(pred_set)

        precision = num_correct / num_predicted if num_predicted > 0 else 0.0
        recall = num_correct / len(gt_set) if len(gt_set) > 0 else 0.0

        if (4 * precision + recall) > 0:
            f2 = (5 * precision * recall) / (4 * precision + recall)
        else:
            f2 = 0.0

        total_f2 += f2
        total_precision += precision
        total_recall += recall

    if processed_samples > 0:
        avg_f2 = total_f2 / processed_samples
        avg_precision = total_precision / processed_samples
        avg_recall = total_recall / processed_samples
    else:
        avg_f2 = 0.0
        avg_precision = 0.0
        avg_recall = 0.0

    print("\n--- F2 Score Evaluation ---")
    print(f"Processed samples: {processed_samples}/{len(predictions)}")
    print(f"Average F2 Score: {avg_f2:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print("--------------------------")

    return {
        "avg_f2_score": avg_f2,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall
    }


def main():
    """
    Main function to orchestrate the RAG pipeline execution without a reranker.
    """
    # --- Configuration ---
    CONFIG = {
        "retriever_path": "trained_model/final_model.pt-2.pt",
        "doc_embedding_path": "data/record_id_to_document_embedding.json",
        "test_data_path": "data/VLSP 2025 - MLQA-TSR Data Release/public_test/vlsp_2025_public_test_task2.json",
        "image_base_path": "data/VLSP 2025 - MLQA-TSR Data Release/public_test/public_test_images/public_test_images",
        "output_path": "submission_result/submission_retrieval_only.json",
        "retrieval_k": 10,  # Number of documents to retrieve initially
    }

    # --- Initialization ---
    print("--- Initializing RAG Pipeline (Retrieval Only) ---")
    retriever_model = RetrieverV2()
    retriever_model = load_model(retriever_model, CONFIG["retriever_path"])

    faiss_index, record_ids, doc_embeddings = setup_faiss_index(CONFIG["doc_embedding_path"])

    print("\n--- Loading Test Data ---")
    with open(CONFIG["test_data_path"], "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test items.")

    # --- Run Pipeline ---
    print("\n--- Starting RAG Processing ---")
    final_submission_results = run_full_rag_pipeline(
        retriever=retriever_model,
        test_data=test_data,
        faiss_index=faiss_index,
        record_ids=record_ids,
        doc_embeddings=doc_embeddings,
        config=CONFIG
    )

    # --- Evaluate Results ---
    evaluate_f2_score(final_submission_results, test_data)

    # --- Save Results ---
    print(f"\n--- Saving Final Results ---")
    output_dir = os.path.dirname(CONFIG["output_path"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(CONFIG["output_path"], "w", encoding="utf-8") as f:
        json.dump(final_submission_results, f, indent=4, ensure_ascii=False)

    print(f"Processing complete. Submission file saved to: {CONFIG['output_path']}")


if __name__ == "__main__":
    main()