from torch import nn, relu
from dgl.nn import GraphConv
from typing import List, Callable


__all__ = ['MultilayerGCN']


class MultilayerGCN(nn.Module):
    def __init__(self, in_features: int, hidden_features: List[int], out_features: int, activation: Callable[[], nn.Module]=nn.ReLU):
        super().__init__()

        layers = [GraphConv(in_feats=in_features, out_feats=hidden_features[0]), activation()]

        for i in range(1, len(hidden_features)):
            layers.extend([GraphConv(hidden_features[i - 1], hidden_features[i]), activation()])

        layers.append(GraphConv(hidden_features[-1], out_features))
        self.layers = nn.ModuleList(layers)

    def forward(self, blocks, input_features):
        for i, layer in enumerate(self.layers):
            input_features = layer(blocks[i], input_features)

        return input_features

