import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """
    Enhanced fuses a text query embedding with image patch embeddings using multi-head cross-attention
    with additional processing layers
    """
    def __init__(self, text_embed_dim: int, image_embed_dim: int, projection_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Projection layers to map both text and image embeddings to a common dimension
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.image_projection = nn.Sequential(
            nn.Linear(image_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )

        # Multi-layer cross-attention
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=projection_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)  # Two cross-attention layers
        ])
        
        # Layer normalization for each attention layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(projection_dim) for _ in range(2)
        ])
        
        # Feed-forward networks after each attention layer
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(projection_dim, projection_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim * 4, projection_dim),
                nn.Dropout(dropout)
            ) for _ in range(2)
        ])

    def forward(self, text_embedding: torch.Tensor, image_embedding: torch.Tensor) -> torch.Tensor:
        """
        Performs projection and then multi-layer cross-attention

        :param text_embedding: The query from the text (torch.Tensor). Shape: (batch_size, 1, text_embed_dim)
        :param image_embedding: The key/value from the image (torch.Tensor). Shape: (batch_size, num_patches, image_embed_dim)
        :return: torch.Tensor: The fused embedding. Shape: (batch_size, 1, projection_dim)
        """
        # Project the text and image embeddings to the common dimension
        projected_text = self.text_projection(text_embedding)
        projected_image = self.image_projection(image_embedding)

        # Multi-layer cross-attention with residual connections
        output = projected_text
        for i, (attn_layer, layer_norm, ff_layer) in enumerate(zip(
            self.cross_attention_layers, self.layer_norms, self.feed_forwards
        )):
            # Cross-attention
            attn_output, _ = attn_layer(output, projected_image, projected_image)
            
            # Residual connection + layer norm
            output = layer_norm(output + attn_output)
            
            # Feed-forward network with residual connection
            ff_output = ff_layer(output)
            output = layer_norm(output + ff_output)
        
        return output
