"""Expert Chess Model Architecture and Activation Extraction.

SE-ResNet-20 Dual-Head Chess Neural Network (Ahmed Darwish).
Extracts 768-dimensional latent representations combining:
  - 256-dim penultimate value representation (positional & strategic evaluation)
  - 512-dim slice of penultimate policy representation (tactical move dynamics)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

PIECE_IDX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def fen_to_tensor(fen: str) -> np.ndarray:
    """Converts a FEN string to an (18, 8, 8) float tensor for the SE-ResNet."""
    board = chess.Board(fen)
    t = np.zeros((18, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            r = 7 - (sq >> 3)
            c = sq & 7
            ch = PIECE_IDX[piece.piece_type]
            t[ch if piece.color else ch + 6, r, c] = 1.0
    if board.turn == chess.WHITE:
        t[12] = 1.0
    t[13] = float(board.has_kingside_castling_rights(chess.WHITE))
    t[14] = float(board.has_queenside_castling_rights(chess.WHITE))
    t[15] = float(board.has_kingside_castling_rights(chess.BLACK))
    t[16] = float(board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        t[17, 7 - (board.ep_square >> 3), board.ep_square & 7] = 1.0
    return t


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + r, inplace=True)


class ExpertChessEncoder(nn.Module):
    """SE-ResNet-20 Dual-Head Chess Expert.
    
    Produces 768-dimensional latent vectors encoding the full board state.
    """
    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        super().__init__()
        in_channels = 18
        num_filters = 256
        num_res_blocks = 20
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[SEResBlock(num_filters) for _ in range(num_res_blocks)])
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 64 * 64 * 5),
        )
        if weights_path:
            raw = torch.load(weights_path, map_location=device)
            renamed = {}
            for k, v in raw.items():
                new_k = k.replace("stem.", "input_block.").replace("v_head.", "value_head.").replace("p_head.", "policy_head.")
                renamed[new_k] = v
            self.load_state_dict(renamed, strict=False)
        self.to(device)
        self.eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 768-dimensional latent vector (256 value + 512 policy)."""
        with torch.no_grad():
            feat = self.input_block(x)
            feat = self.tower(feat)
            val_dense = self.value_head[:6](feat)  # [B, 256]
            pol_dense = self.policy_head[:6](feat)[:, :512]  # [B, 512]
            emb = torch.cat([val_dense, pol_dense], dim=-1)  # [B, 768]
        return emb
