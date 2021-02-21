from torch import nn
# from wrapper import layer_wrapper
from .wrapper import layer_wrapper
class DiabetesClassifier(nn.Module):
    def __init__(self, input_size, n1, n2, n3, output_size, dropout=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            layer_wrapper(input_size, n1, dropout=dropout),
            layer_wrapper(n1, n2, dropout=dropout),
            layer_wrapper(n2, n3, dropout=dropout),
            layer_wrapper(n3, output_size, activation="lsoftmax")
        )

    def forward(self, x):
        return self.fc(x)