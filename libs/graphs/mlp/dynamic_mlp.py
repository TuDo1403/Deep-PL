from torch import nn

class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, drop_prob):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(p=drop_prob)
        )

    def forward(self, x):
        return self.layer(x)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, 
                 out_features,
                 backbones,
                 **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([
            FullyConnectedLayer(in_features, out_features, drop_prob)\
                for (in_features, _), (out_features, drop_prob) in zip(backbones, backbones[1:])
        ])

        self.last_layer = nn.Linear(backbones[-1][0], out_features)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out.view(out.size(0), -1), self.last_layer(out)