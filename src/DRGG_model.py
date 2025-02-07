import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Backbone(nn.Module):
    """Feature extractor backbone, typically a CNN like ResNet."""
    def __init__(self, output_dim=256):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove classification head
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        features = self.cnn(x)
        return self.fc(features)

class Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, input_dim=256, num_layers=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    """Transformer Decoder with multi-layer output tracking."""
    def __init__(self, output_dim=256, num_layers=6):
        super().__init__()
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=output_dim, nhead=8)
            for _ in range(num_layers)
        ])

    def forward(self, x, memory):
        layer_outputs = []
        for layer in self.decoder_layers:
            x = layer(x, memory)
            layer_outputs.append(x)  # Store output from each layer

        return layer_outputs  # List of outputs, one per layer

class RelationHeads(nn.Module):
    """Processes relation tokens through two parallel branches."""
    def __init__(self, input_dim=256, hidden_dim=128, upsample_factor=2, output_dim=5):
        super().__init__()

        # First branch
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.upsample1 = nn.Upsample(scale_factor=upsample_factor, mode='nearest')
        self.mlp1_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Second branch
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.mlp2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.upsample2 = nn.Upsample(scale_factor=upsample_factor, mode='nearest')
        self.mlp2_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Concatenation layer
        self.concat_layer = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        x_pooled1 = self.pool1(x.transpose(1, 2)).squeeze(-1)
        d1 = self.mlp1(x_pooled1)
        d1 = self.upsample1(d1.unsqueeze(-1)).squeeze(-1)
        d1 = self.mlp1_out(d1)

        x_pooled2 = self.pool2(x.transpose(1, 2)).squeeze(-1)
        d2 = self.mlp2(x_pooled2)
        d2 = self.upsample2(d2.unsqueeze(-1)).squeeze(-1)
        d2 = self.mlp2_out(d2)

        # Concatenation and final prediction
        d_concat = torch.cat([d1, d2], dim=-1)
        return self.concat_layer(d_concat)

class RelationAggregator(nn.Module):
    """Weighted aggregation of Relation Heads' outputs."""
    def __init__(self, input_dim=5, num_layers=6):
        super().__init__()
        self.num_layers = num_layers
        self.alphas = nn.Parameter(torch.randn(num_layers))  # Learnable weights
        self.mlp_g = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, relation_outputs):
        """
        relation_outputs: List of tensors [(batch, seq_len, output_dim)] from each relation head
        """
        # Compute softmax over the alphas
        alpha_weights = F.softmax(self.alphas, dim=0)  # (num_layers,)

        # Weighted sum of relation head outputs
        weighted_sum = sum(alpha * output for alpha, output in zip(alpha_weights, relation_outputs))

        # Final transformation
        return self.mlp_g(weighted_sum)

class DRGGModel(nn.Module):
    """End-to-End Model integrating all components"""
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.relation_heads = RelationHeads()
        self.aggregator = RelationAggregator()
        self.object_heads = nn.Linear(256, 10)  # Example output classes for layout objects

    def forward(self, image, object_queries):
        features = self.backbone(image)
        encoded_features = self.encoder(features.unsqueeze(0))  # Add sequence dimension
        decoder_outputs = self.decoder(object_queries, encoded_features)

        # Apply Relation Heads to each decoder output
        relation_outputs = [self.relation_heads(output) for output in decoder_outputs]

        # Weighted Aggregation of relation head outputs
        aggregated_relations = self.aggregator(relation_outputs)

        object_predictions = self.object_heads(decoder_outputs[-1])  # Use last decoder output for objects

        return object_predictions, aggregated_relations

#checking if this actually runs lol
if __name__ == '__main__':
    image = torch.randn(1, 3, 224, 224)  # Dummy image
    object_queries = torch.randn(5, 1, 256)  # Dummy object queries

    model = DRGGModel()
    obj_pred, rel_pred = model(image, object_queries)
    print(obj_pred.shape, rel_pred.shape)
