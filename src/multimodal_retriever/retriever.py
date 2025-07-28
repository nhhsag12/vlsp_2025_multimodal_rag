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
    The main class that combines all components to generate a multimodal query embedding
    """
    def __init__(self, projection_dim: int = 1024):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_encoder = VisionEncoder().to(self.device)
        self.text_encoder = TextEncoder() # Manages its own device

        # Get embedding dimensions from the models
        text_embed_dim = self.text_encoder.model.get_sentence_embedding_dimension() # Should be 1024
        image_embed_dim = self.vision_encoder.vision_model.config.hidden_size # Should be 768

        # The common dimension to project both embeddings into
        self.projection_dim = projection_dim

        # As per the diagram, the text embedding (Q) and image features (K, V) are fused
        # The fusion module now handles the projection internally
        self.fusion_module = CrossAttention(
            text_embed_dim=text_embed_dim,
            image_embed_dim=image_embed_dim,
            projection_dim=self.projection_dim
        ).to(self.device)

        # The final output projection takes projection_dim as input
        self.output_projection = nn.Linear(self.projection_dim, self.projection_dim).to(self.device)

    def forward(self, image: Image.Image, text_query: str) -> np.ndarray:
        """
        Generates a final, fused embedding for retrieval
        :param image: the input image (Image.Image)
        :param text_query: The text query (str)
        :return: np.ndarray: A 1D numpy array representing the final query vector
        """
        # 1. Encode the image to get patch embeddings (K, V)
        image_embedding = self.vision_encoder(image)

        # 2. Encode the text to get the query embedding (Q)
        text_embedding = self.text_encoder.encode(text_query) # Shape: (1, 1024)
        text_embedding = text_embedding.unsqueeze(1) # (batch, dim) -> (batch, seq_len, dim)

        # 3. Fuse them using the cross-attention module
        # The fusion module will handle projecting to the common dimension
        fused_embedding = self.fusion_module(text_embedding, image_embedding)

        # 4. Project the fused embedding to get the final query vector
        # Squeeze to remove the sequence dimension from (1, 1, 1024) -> (1, 1024)
        final_vector = self.output_projection(fused_embedding.squeeze(1))

        # 5. normalize the final vector (good practice for cosine similarity search)
        final_vector = nn.functional.normalize(final_vector, p=2, dim=1)

        # Detach from graph, move to CPU and convert to numpy
        return final_vector.detach().cpu().numpy()

if __name__ == "__main__":
    print("\n--- Initializing Multimodal Retriever ---")
    # We can define the projection dimension here. Let's use the text model's dimension.
    retriever = Retriever(projection_dim=1024)
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
    doc_encoder = SentenceTransformer("BAAI/bge-m3",device="cuda" if torch.cuda.is_available() else "cpu")
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




