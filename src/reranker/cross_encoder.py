import torch
import torch.nn as nn
import json

from src.multimodal_retriever.attention.cross_attention import CrossAttention
from src.multimodal_retriever.encoders.text_encoder import TextEncoder
from src.multimodal_retriever.encoders.vision_encoder import VisionEncoder
from typing import Union, List, Tuple
from PIL import Image


class CrossEncoder(nn.Module):
    def __init__(self, text_model: TextEncoder, vision_model: VisionEncoder, dropout=0.1):
        super().__init__()

        # The file of the document embedding
        self.document_to_embedding_path = "data/document_text_to_embedding.json"
        self.document_to_embedding = None
        if self.document_to_embedding_path:
            with open(self.document_to_embedding_path, "r") as f:
                self.document_embeddings = json.load(f)

        # --- Initialize the models ---
        self.text_model = text_model
        self.vision_model = vision_model

        # --- Fusion and Classification Head ---
        text_hidden_size = self.text_model.model.get_sentence_embedding_dimension()
        image_hidden_size = self.vision_model.vision_model.config.hidden_size

        # Fusion layer to combine multimodal query with document text
        fusion_input_size = text_hidden_size + image_hidden_size + text_hidden_size  # query_text + image + doc_text
        self.projection_dim = int(text_hidden_size + image_hidden_size)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_input_size // 2),
            nn.LayerNorm(fusion_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_size // 2, fusion_input_size // 4),
            nn.LayerNorm(fusion_input_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_size // 4, 1)  # Binary relevance score
        )

        # Alternative: You could use the attention mechanism for better fusion
        self.use_attention = True
        if self.use_attention:
            self.attention_layer = CrossAttention(
                image_embed_dim=image_hidden_size,
                text_embed_dim=text_hidden_size,
                projection_dim=self.projection_dim,
            )
            # self.attention_layer = nn.MultiheadAttention(
            #     embed_dim=text_hidden_size,
            #     num_heads=8,
            #     dropout=dropout,
            #     batch_first=True
            # )

    def forward(self,
                query_image: Union[Image.Image, List[Image.Image]],
                query_text: Union[str, List[str]],
                document_texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Forward pass for multimodal cross encoder

        :param query_image: PIL Image or list of PIL Images
        :param query_text: Query text string or list of strings
        :param document_texts: Document text string or list of strings
        :return: Relevance scores tensor of shape (batch_size, 1)
        """
        # Get current device
        device = next(self.parameters()).device

        # Ensure all inputs are lists for batch processing
        if isinstance(query_image, Image.Image):
            query_image = [query_image]
        if isinstance(query_text, str):
            query_text = [query_text]
        if isinstance(document_texts, str):
            document_texts = [document_texts]

        batch_size = len(query_image)

        # Encode query text
        query_text_embeddings = self.text_model.encode(query_text)  # (batch_size, text_hidden_size)
        query_text_embeddings = query_text_embeddings.unsqueeze(1) # (batch_size, 1, text_hidden_size)

        # Encode query image
        image_embeddings = self.vision_model(query_image)  # (batch_size, seq_len, image_hidden_size)
        # Use CLS token (first token) or average pooling for image representation
        # image_embedding = image_embedding[:, 0, :]  # (batch_size, image_hidden_size) - CLS token

        # Encode document texts
        if self.document_embeddings:
            doc_text_embeddings = []
            for text in document_texts:
                doc_embedding = self.document_embeddings.get(text, None)
                if doc_embedding:
                    doc_text_embeddings.append(doc_embedding)
            doc_text_embeddings = torch.tensor(doc_text_embeddings) # (batch_size, text_hidden_size)
        else:
            doc_text_embeddings = self.text_model.encode(document_texts)  # (batch_size, text_hidden_size)

        # Move embeddings to correct device
        query_text_embeddings = query_text_embeddings.clone().to(device)
        image_embeddings = image_embeddings.to(device)
        doc_text_embeddings = doc_text_embeddings.to(device)

        if self.use_attention:
            attended_output = self.attention_layer(
                text_embedding=query_text_embeddings,
                image_embedding=image_embeddings
            )
            attended_output = attended_output.squeeze(1)

            # Combine with original embeddings
            # print(doc_text_embeddings.shape)
            # print(attended_output.shape)
            fused_features = torch.cat([
                attended_output,
                doc_text_embeddings
            ], dim=-1)
        else:
            # Simple concatenation fusion
            fused_features = torch.cat([
                query_text_embeddings,
                image_embeddings,
                doc_text_embeddings
            ], dim=-1)  # (batch_size, fusion_input_size)

        # Apply fusion layer to get relevance scores
        relevance_scores = self.fusion_layer(fused_features)  # (batch_size, 1)

        return relevance_scores

    def predict_relevance(self,
                          query_image: Union[Image.Image, List[Image.Image]],
                          query_text: Union[str, List[str]],
                          document_texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Predict relevance scores and apply sigmoid for probability interpretation

        :param query_image: PIL Image or list of PIL Images
        :param query_text: Query text string or list of strings
        :param document_texts: Document text string or list of strings
        :return: Relevance probabilities tensor of shape (batch_size, 1)
        """
        with torch.no_grad():
            scores = self.forward(query_image, query_text, document_texts)
            probabilities = torch.sigmoid(scores)
            return probabilities

    def rank_documents(self,
                       query_image: Image.Image,
                       query_text: str,
                       document_texts: List[str]) -> List[Tuple[int, float]]:
        """
        Rank documents by relevance to the multimodal query

        :param query_image: Single PIL Image
        :param query_text: Single query text string
        :param document_texts: List of document text strings
        :return: List of tuples (document_index, relevance_score) sorted by relevance
        """
        # Prepare batch inputs
        batch_images = [query_image] * len(document_texts)
        batch_query_texts = [query_text] * len(document_texts)

        # Get relevance scores
        scores = self.predict_relevance(batch_images, batch_query_texts, document_texts)

        # Create ranked list
        ranked_docs = [(i, score.item()) for i, score in enumerate(scores)]
        ranked_docs.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending

        return ranked_docs