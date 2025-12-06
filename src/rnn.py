import torch
from torch import nn

class AMCRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cell: str = "gru",
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        input_dim = 2  # I/Q as features

        if cell.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 2, 128)
        x = x.transpose(1, 2)       # -> (B, 128, 2)
        out, _ = self.rnn(x)        # (B, T, H*dirs)
        last = out[:, -1, :]        # last timestep
        logits = self.fc(last)
        return logits


RNN_ARCH_CONFIGS = {
    "GRU_base_h128_L1": dict(cell="gru", hidden_size=128, num_layers=1, bidirectional=False, dropout=0.0),
    "GRU_big_h256_L2_do0.3": dict(cell="gru", hidden_size=256, num_layers=2, bidirectional=False, dropout=0.3),
    "GRU_bidir_h128_L1": dict(cell="gru", hidden_size=128, num_layers=1, bidirectional=True, dropout=0.0),
    "LSTM_base_h128_L1": dict(cell="lstm", hidden_size=128, num_layers=1, bidirectional=False, dropout=0.0),
}

DEFAULT_ARCH_KEY = "GRU_base_h128_L1"


def build_model_for_name(name: str, num_classes: int, device=None) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_cfg = RNN_ARCH_CONFIGS.get(name, RNN_ARCH_CONFIGS[DEFAULT_ARCH_KEY])
    model = AMCRNN(num_classes=num_classes, **arch_cfg).to(device)
    return model
