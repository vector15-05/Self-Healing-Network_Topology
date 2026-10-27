from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EdgeMessagePassingNetwork(nn.Module):
    """Edge-aware message-passing network with link-risk and impact heads."""

    def __init__(
        self,
        node_features: int = 6,
        edge_features: int = 6,
        hidden_dim: int = 32,
        message_passing_steps: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or message_passing_steps <= 0:
            raise ValueError("hidden_dim and message_passing_steps must be positive")
        self.node_projection = nn.Linear(node_features, hidden_dim)
        self.edge_projection = nn.Linear(edge_features, hidden_dim)
        self.message_networks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(message_passing_steps)
        )
        self.update_cells = nn.ModuleList(
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(message_passing_steps)
        )
        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.temporal_link_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.impact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if node_features.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Expected node_features [N,F] and edge_index [2,E]")
        if edge_features.ndim != 2 or edge_features.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_features must have one row per edge")

        node_state = torch.relu(self.node_projection(node_features))
        edge_state = torch.relu(self.edge_projection(edge_features))
        source, target = edge_index.long()
        all_source = torch.cat((source, target), dim=0)
        all_target = torch.cat((target, source), dim=0)
        all_edge = torch.cat((edge_state, edge_state), dim=0)
        for message_network, update_cell in zip(self.message_networks, self.update_cells):
            messages = message_network(
                torch.cat(
                    (node_state[all_source], node_state[all_target], all_edge),
                    dim=-1,
                )
            )
            aggregate = torch.zeros_like(node_state)
            aggregate.index_add_(0, all_target, messages)
            counts = torch.bincount(all_target, minlength=node_state.shape[0]).to(node_state.dtype)
            aggregate = aggregate / counts.clamp_min(1.0).unsqueeze(-1)
            node_state = update_cell(aggregate, node_state)

        edge_state_for_head = torch.cat(
            (node_state[source], node_state[target], edge_state),
            dim=-1,
        )
        return {
            "link_failure_logits": self.link_head(edge_state_for_head).squeeze(-1),
            "impact": self.impact_head(node_state.mean(dim=0)).squeeze(-1),
            "node_embeddings": node_state,
            "edge_embeddings": edge_state_for_head,
        }

    def forward_window(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode a fixed topology over K ticks and predict next-horizon link risk."""
        if node_features.ndim != 3 or edge_features.ndim != 3:
            raise ValueError("Temporal inputs must be node_features [K,N,F], edge_features [K,E,F]")
        if node_features.shape[0] != edge_features.shape[0]:
            raise ValueError("Temporal node and edge windows must have the same K")
        window_outputs = [
            self.forward(node_features[tick], edge_index, edge_features[tick])
            for tick in range(node_features.shape[0])
        ]
        edge_sequence = torch.stack(
            [output["edge_embeddings"] for output in window_outputs],
            dim=0,
        ).transpose(0, 1)
        temporal_state, _ = self.temporal_encoder(edge_sequence)
        final_state = temporal_state[:, -1]
        return {
            "link_failure_logits": self.temporal_link_head(final_state).squeeze(-1),
            "impact": window_outputs[-1]["impact"],
            "node_embeddings": window_outputs[-1]["node_embeddings"],
            "edge_embeddings": final_state,
        }


def train_edge_gnn(
    model: EdgeMessagePassingNetwork,
    graphs: list[dict[str, Any]],
    *,
    epochs: int = 50,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Train the link-risk head on tensor dictionaries produced by data conversion."""
    if not graphs:
        raise ValueError("At least one training graph is required")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        losses: list[torch.Tensor] = []
        for graph in graphs:
            if "window" in graph:
                window = graph["window"]
                output = model.forward_window(
                    window["node_features"],
                    window["edge_index"],
                    window["edge_features"],
                )
            else:
                output = model(
                    graph["node_features"],
                    graph["edge_index"],
                    graph["edge_features"],
                )
            labels = graph["labels"].to(dtype=torch.float32)
            positives = labels.sum().clamp_min(1.0)
            negatives = (labels.numel() - labels.sum()).clamp_min(1.0)
            loss = nn.functional.binary_cross_entropy_with_logits(
                output["link_failure_logits"],
                labels,
                pos_weight=(negatives / positives),
            )
            if "impact_label" in graph:
                loss = loss + nn.functional.mse_loss(
                    output["impact"],
                    graph["impact_label"].to(dtype=torch.float32),
                )
            losses.append(loss)
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        history.append(float(total_loss.detach().cpu()))
    return history
