import torch
import torch.nn as nn


class WideAndDeepModel(nn.Module):
    def __init__(self, wide_dim, deep_dim, hidden_units, output_dim):
        super(WideAndDeepModel, self).__init__()
        self.wide = nn.Linear(wide_dim, output_dim)
        self.deep = nn.Sequential(
            nn.Linear(deep_dim, hidden_units[0]),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_units[i], hidden_units[i+1]), nn.ReLU())
              for i in range(len(hidden_units) - 1)],
            nn.Linear(hidden_units[-1], output_dim)
        )

    def forward(self, wide_input, deep_input):
        wide_out = self.wide(wide_input)
        deep_out = self.deep(deep_input)
        combined_out = wide_out + deep_out  # assuming output_dim=1 for a regression or binary classification task
        return torch.sigmoid(combined_out)  # use sigmoid for binary classification

# Example initialization:
# model = WideAndDeepModel(wide_dim=10, deep_dim=20, hidden_units=[64, 32], output_dim=1)


class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, general_dim, hidden_units):
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, general_dim)
        self.item_embedding = nn.Embedding(num_items, general_dim)
        self.mlp = nn.Sequential(
            nn.Linear(general_dim * 2, hidden_units[0]),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_units[i], hidden_units[i + 1]), nn.ReLU())
              for i in range(len(hidden_units) - 1)],
            nn.Linear(hidden_units[-1], 1)
        )

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        output = self.mlp(vector)
        return torch.sigmoid(output)  # use sigmoid for binary classification

# Example initialization:
# model = NCFModel(num_users=1000, num_items=1000, general_dim=8, hidden_units=[64, 32])
