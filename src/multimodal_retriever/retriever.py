import time
from typing import Union, List

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer

from src.multimodal_retriever.attention.cross_attention import CrossAttention
from src.multimodal_retriever.encoders.text_encoder import TextEncoder
from src.multimodal_retriever.encoders.vision_encoder import VisionEncoder


class Retriever(nn.Module):
    """
    Enhanced multimodal retriever with additional neural network layers
    """

    def __init__(self, projection_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()  # Manages its own device

        # Get embedding dimensions from the models
        text_embed_dim = self.text_encoder.model.get_sentence_embedding_dimension()  # Should be 1024
        image_embed_dim = self.vision_encoder.vision_model.config.hidden_size  # Should be 768

        # The common dimension to project both embeddings into
        self.projection_dim = projection_dim

        # Enhanced fusion module with more layers
        self.fusion_module = CrossAttention(
            text_embed_dim=text_embed_dim,
            image_embed_dim=image_embed_dim,
            projection_dim=self.projection_dim,
            dropout=dropout
        )

        # Multi-layer output projection network
        self.output_projection = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim * 2),
            nn.LayerNorm(self.projection_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.projection_dim * 2, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.Dropout(dropout),
            nn.Linear(self.projection_dim, self.projection_dim)
        )

        # Additional contextual processing layer
        # self.contextual_processor = nn.TransformerEncoderLayer(
        #     d_model=self.projection_dim,
        #     nhead=8,
        #     dim_feedforward=self.projection_dim * 4,
        #     dropout=dropout,
        #     batch_first=True
        # )

    def forward(self, images: Union[Image.Image, List[Image.Image]], text_queries: Union[str, List[str]]) -> Union[
        np.ndarray, torch.Tensor]:
        """
        Generates final, fused embeddings for retrieval with enhanced processing
        :param images: Single image (Image.Image) or list of images (List[Image.Image])
        :param text_queries: Single text query (str) or list of text queries (List[str])
        :return: For single input: np.ndarray (1D array), For batch: torch.Tensor (2D tensor)
        """
        # Handle single input case (backward compatibility)
        if isinstance(images, Image.Image) and isinstance(text_queries, str):
            return self._forward_single(images, text_queries)

        # Handle batch input case
        if isinstance(images, list) and isinstance(text_queries, list):
            return self._forward_batch(images, text_queries)

        raise ValueError("Both images and text_queries must be either single items or lists of the same length")

    def _forward_single(self, image: Image.Image, text_query: str) -> np.ndarray:
        """
        Enhanced forward pass for single image and text query
        """
        # Get current device
        device = next(self.parameters()).device

        # 1. Encode the image to get enhanced patch embeddings (K, V)
        image_embedding = self.vision_encoder(image)
        image_embedding = torch.clone(image_embedding)

        # 2. Encode the text to get the query embedding (Q)
        text_embedding = self.text_encoder.encode(text_query)  # Shape: (1, 1024)
        text_embedding = text_embedding.unsqueeze(1).to(device)  # (batch, dim) -> (batch, seq_len, dim)
        text_embedding = torch.clone(text_embedding)

        # 3. Fuse them using the enhanced cross-attention module
        fused_embedding = self.fusion_module(text_embedding, image_embedding)

        # 4. Apply contextual processing
        # contextual_embedding = self.contextual_processor(fused_embedding)
        contextual_embedding = fused_embedding

        # 5. Project through the enhanced output projection network
        final_vector = self.output_projection(contextual_embedding.squeeze(1))

        # 6. Normalize the final vector (good practice for cosine similarity search)
        final_vector = nn.functional.normalize(final_vector, p=2, dim=1)

        # Detach from graph, move to CPU and convert to numpy
        return final_vector.detach().cpu().numpy()

    def _forward_batch(self, images: List[Image.Image], text_queries: List[str]) -> torch.Tensor:
        """
        Enhanced forward pass for batch of images and text queries
        """
        if len(images) != len(text_queries):
            raise ValueError(
                f"Number of images ({len(images)}) must match number of text queries ({len(text_queries)})")

        # Get current device
        device = next(self.parameters()).device

        # 1. Encode the images to get enhanced patch embeddings (K, V)
        batched_image_embeddings = self.vision_encoder(images)
        batched_image_embeddings = torch.clone(batched_image_embeddings)

        # 2. Encode the text to get the query embeddings (Q)
        batched_text_embeddings = self.text_encoder.encode(text_queries)
        batched_text_embeddings = batched_text_embeddings.unsqueeze(1).to(
            device)  # (batch, dim) -> (batch, num_patches, dim)
        batched_text_embeddings = torch.clone(batched_text_embeddings)

        # 3. Fuse them using the enhanced cross-attention module
        fused_embeddings = self.fusion_module(batched_text_embeddings, batched_image_embeddings)

        # 4. Apply contextual processing
        # contextual_embeddings = self.contextual_processor(fused_embeddings)
        contextual_embeddings = fused_embeddings

        # 5. Project through the enhanced output projection network
        # Shape: (batch_size, 1, projection_dim) -> (batch_size, projection_dim)
        final_vectors = self.output_projection(contextual_embeddings.squeeze(1))

        # 6. Normalize the final vectors (good practice for cosine similarity search)
        final_vectors = nn.functional.normalize(final_vectors, p=2, dim=1)

        return final_vectors


if __name__ == "__main__":
    print("\n--- Initializing Enhanced Multimodal Retriever ---")
    # We can define the projection dimension here. Let's use the text model's dimension.
    retriever = Retriever(projection_dim=1024, dropout=0.1)
    retriever.eval()  # Set to evaluation mode

    # --- Setup a dummy vector database (your "raw_db") ---
    print("\n--- Setting up a dummy FAISS vector database ---")

    # Let's imagine our database contains documents about different topics.
    # We will just embed their text descriptions for this example.
    # In a real scenario, these could be embeddings of documents, images, or multimodal chunks.
    dummy_documents = [
        "A detailed report on the financial performance of tech companies in 2023.",
        "A recipe for a classic Italian lasagna with a rich bolognese sauce.",
        "User manual for the 'VisionPro 9000' camera, explaining its advanced features.",
        "A travel guide to the scenic mountains of Switzerland, highlighting hiking trails.",
        "Architectural blueprints for a modern, sustainable skyscraper with a glass facade.",
    ]

    # We use a text encoder to create embeddings for the documents.
    # The database embeddings should match the final output dimension of our retriever.
    # The retriever's output is `projection_dim` which is 1024.
    doc_encoder = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    print("Encoding dummy documents for the database...")
    document_embeddings = doc_encoder.encode(dummy_documents, normalize_embeddings=True)
    # document_embeddings = [doc_encoder.encode(dummy_document) for dummy_document in dummy_documents]

    # Create a FAISS index. The dimension must match the embeddings.
    embedding_dim = document_embeddings.shape[1]
    index = faiss.IndexFlatIP(
        embedding_dim)  # Using Inner Product (IP) which is equivalent to cosine for normalized vectors.
    index.add(document_embeddings.astype(np.float32))
    print(f"FAISS index created with {index.ntotal} vectors of dimension {embedding_dim}.")

    # --- Perform a multimodal query ---
    print("\n--- Performing a multimodal query ---")

    # Example query: an image of a building and a question about it.
    # Let's get a sample image from the web.
    image_url = "https://upload.wikimedia.org/wikipedia/commons/4/45/WilderBuildingSummerSolstice.jpg"
    try:
        query_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        print("Successfully downloaded a sample image of a skyscraper.")
    except Exception as e:
        print(f"Could not download image, creating a dummy one. Error: {e}")
        query_image = Image.new('RGB', (224, 224), color='red')

    query_text = "Show me the plans for this building."

    print(f"Query Text: '{query_text}'")

    # Generate the multimodal query vector
    print("Generating multimodal query vector...")
    with torch.inference_mode():
        query_vector = retriever(query_image, query_text)

    # Search the FAISS index
    k = 3  # Retrieve top 3 results
    print(f"\nSearching for top {k} results in the database...")
    distances, indices = index.search(query_vector.astype(np.float32), k)

    # Print results
    print("\n--- Retrieval Results ---")
    for i in range(k):
        doc_index = indices[0][i]
        similarity_score = distances[0][i]
        print(f"Rank {i + 1}: Document {doc_index} (Similarity: {similarity_score:.4f})")
        print(f"   Content: '{dummy_documents[doc_index]}'")

    # Test batch processing
    print("\n--- Testing batch processing ---")
    test_images = [query_image, query_image]
    test_texts = ["Show me the plans for this building.", "What is this building?"]

    retriever.train()  # Set to training mode for batch processing
    batch_embeddings = retriever(test_images, test_texts)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")