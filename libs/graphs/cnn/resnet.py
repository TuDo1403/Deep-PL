from torch import nn

from .operations import ResidualBLock, Conv3x3BNReLU

class ResNet(nn.Module):
    def __init__(self, 
                 num_classes, 
                 phases, 
                 reductions,
                 expansion_rate=2,
                 stem_out_channels=64,
                 use_stem=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.stem_conv = Conv3x3BNReLU(3, stem_out_channels) if use_stem else nn.Identity()
        in_channels = stem_out_channels if use_stem else 3
        out_channels = in_channels

        for i, (blocks_per_phase, reduction) in enumerate(zip(phases, reductions)):
            if reduction:
                reduction_block = ResidualBLock(in_channels, out_channels, stride=2)
                self.layers += [reduction_block]
            out_channels *= expansion_rate
            for j in range(blocks_per_phase):
                block = ResidualBLock(in_channels, out_channels)
                self.layers += [block]
                in_channels = out_channels
            
            
            

        self.gap = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        out = self.stem_conv(x)
        for _, layer in enumerate(self.layers):
            out = layer(out)
        
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits





