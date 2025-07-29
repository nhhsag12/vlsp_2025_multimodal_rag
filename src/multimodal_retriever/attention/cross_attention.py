import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """
    Fuses a text query embedding with image patch embeddings using multi-head cross-attention
    """
    def __init__(self, text_embed_dim: int, image_embed_dim: int, projection_dim: int, num_heads: int = 8):
        super().__init__()
        # Projection layers to map both text and image embeddings to a common dimenstion
        self.text_projection = nn.Linear(text_embed_dim, projection_dim)
        self.image_projection = nn.Linear(image_embed_dim, projection_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            batch_first=True # Expects (batch, seq, feature)
        )

    def forward(self, text_embedding: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Performs projection and then cross-attention

        :param text_embedding: The query from the text (torch.Tensor). Shape: (batch_size, 1, text_embed_dim)
        :param image_embeddings: The key/value from the image (torch.Tensor). Shape: (batch_size, num_patches, image_embed_dim)
        :return: torch.Tensor: The fused embedding. Shape: (batch_size, 1, projection_dim)
        """
        # Clone the input tensors to ensure they're proper tensors for autograd

        # Project the text and image embeddings to the common dimension
        projected_text = self.text_projection(text_embedding)
        projected_image = self.image_projection(image_embeddings)
        # print(projected_text.shape)
        # print(projected_image.shape)

        # In cross-attention, the query comes from one modality (text) and the keys/values from another (image)
        # Query: projected_text
        # Key: projected_image
        # Value: projected_image
        attn_output, _ = self.cross_attention(projected_text, projected_image, projected_image)
        return attn_output
