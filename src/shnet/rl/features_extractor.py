from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GraphPolicyFeatureExtractor(BaseFeaturesExtractor):
    """Pool edge-aware message-passing embeddings into an SB3 policy vector."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 32,
        message_passing_steps: int = 2,
    ) -> None:
        node_slots = observation_space["node_features"].shape[0]
        edge_slots = observation_space["edge_features"].shape[0]
        node_features = observation_space["node_features"].shape[-1]
        edge_features = observation_space["edge_features"].shape[-1]
        flow_features = observation_space["flow_features"].shape[-1]
        calculated_features_dim = (
            node_slots * hidden_dim
            + edge_slots * hidden_dim * 3
            + hidden_dim * 3
            + flow_features
            + observation_space["metrics"].shape[-1]
        )
        super().__init__(observation_space, calculated_features_dim)
        self.node_projection = nn.Linear(node_features, hidden_dim)
        self.edge_projection = nn.Linear(edge_features, hidden_dim)
        self.messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(message_passing_steps)
        )
        self.updates = nn.ModuleList(
            nn.GRUCell(hidden_dim, hidden_dim)
            for _ in range(message_passing_steps)
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        nodes = torch.relu(self.node_projection(observations["node_features"].float()))
        edges = torch.relu(self.edge_projection(observations["edge_features"].float()))
        edge_index = observations["edge_index"].long()
        batch_size = nodes.shape[0]
        embeddings: list[torch.Tensor] = []
        for batch in range(batch_size):
            node_state = nodes[batch]
            node_mask = observations["node_mask"][batch].bool()
            edge_mask = observations["edge_mask"][batch].bool()
            flow_mask = observations["flow_mask"][batch].bool()
            node_state = node_state * node_mask.unsqueeze(-1)
            edge_state = edges[batch, edge_mask]
            source = edge_index[batch, 0, edge_mask]
            target = edge_index[batch, 1, edge_mask]
            all_source = torch.cat((source, target), dim=0)
            all_target = torch.cat((target, source), dim=0)
            all_edge = torch.cat((edge_state, edge_state), dim=0)
            for message_network, update_cell in zip(self.messages, self.updates):
                message = message_network(
                    torch.cat(
                        (
                            node_state[all_source],
                            node_state[all_target],
                            all_edge,
                        ),
                        dim=-1,
                    )
                )
                aggregate = torch.zeros_like(node_state)
                aggregate.index_add_(0, all_target, message)
                count = torch.bincount(all_target, minlength=node_state.shape[0]).to(node_state.dtype)
                aggregate = aggregate / count.clamp_min(1.0).unsqueeze(-1)
                node_state = update_cell(aggregate, node_state)
                node_state = node_state * node_mask.unsqueeze(-1)
            valid_nodes = node_state[node_mask]
            pooled_nodes = valid_nodes.mean(dim=0)
            pooled_max_nodes = valid_nodes.max(dim=0).values
            pooled_edges = edge_state.mean(dim=0) if edge_state.numel() else node_state.new_zeros(node_state.shape[-1])
            contextual_edges = node_state.new_zeros(
                (edge_mask.shape[0], node_state.shape[-1] * 3)
            )
            if source.numel():
                contextual_edges[edge_mask] = torch.cat(
                    (node_state[source], node_state[target], edge_state),
                    dim=-1,
                )
            valid_flows = observations["flow_features"][batch].float()[flow_mask]
            pooled_flows = (
                valid_flows.mean(dim=0)
                if valid_flows.numel()
                else node_state.new_zeros(observations["flow_features"].shape[-1])
            )
            metrics = observations["metrics"][batch].float()
            embeddings.append(
                torch.cat(
                    (
                        node_state.reshape(-1),
                        contextual_edges.reshape(-1),
                        pooled_nodes,
                        pooled_max_nodes,
                        pooled_edges,
                        pooled_flows,
                        metrics,
                    ),
                    dim=0,
                )
            )
        return torch.stack(embeddings, dim=0)
