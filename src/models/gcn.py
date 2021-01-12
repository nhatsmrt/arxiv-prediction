from torch import nn
from dgl.nn import GraphConv
from typing import List, Callable
from pytorch_lightning import LightningModule


__all__ = ['MultilayerGCN']


class MultilayerGCN(LightningModule):
    def __init__(
            self, in_features: int, hidden_features: List[int], out_features: int, activation: Callable[[], nn.Module]=nn.ReLU,

    ):
        super().__init__()

        layers = [GraphConv(in_feats=in_features, out_feats=hidden_features[0])]
        dropouts = [nn.Dropout(0.5) for _ in range(len(hidden_features))]
        batch_norms = [nn.BatchNorm1d(hidden_feature) for hidden_feature in hidden_features]
        activations = [activation() for _ in range(len(hidden_features))]

        for i in range(1, len(hidden_features)):
            layers.append(GraphConv(hidden_features[i - 1], hidden_features[i]))

        layers.append(GraphConv(hidden_features[-1], out_features))
        self.layers = nn.ModuleList(layers)
        self.dropouts = nn.ModuleList(dropouts)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.activations = nn.ModuleList(activations)

    def forward(self, blocks, input_features):
        for i, layer in enumerate(self.layers):
              input_features = layer(blocks[i], input_features)

              if i < len(self.layers) - 1:
                input_features = self.dropouts[i](self.activations[i](self.batch_norms[i](input_features)))

        return input_features
