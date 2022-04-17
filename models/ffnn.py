import torch
import torch.nn as nn


class FFNNModel(nn.Module):
    def __init__(self, config):
        super(FFNNModel, self).__init__()

        self.investment_embedding = nn.Embedding(config.investment_embed_size, config.embed_size, padding_idx=0)

        self.embedding_proj = nn.Sequential(
            nn.Linear(config.embed_size, config.embed_size),
            nn.LayerNorm(config.embed_size),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        # encoder
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(config.feature_size + config.embed_size, momentum=0.1, affine=False),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_size + config.embed_size, config.hidden_size * 4),
            nn.BatchNorm1d(config.hidden_size * 4, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size * 2),
            nn.BatchNorm1d(config.hidden_size * 2, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        self.reg_layer = nn.Sequential(
            nn.Linear(config.hidden_size, len(config.target_cols)),
        )

        self.class_layer = nn.Sequential(
            nn.Linear(config.hidden_size, len(config.target_cols)),
        )

        self.apply(self.init_weights)

    def init_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0., std=0.02)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        if isinstance(module, nn.LayerNorm) and module.weight is not None:
            module.weight.data.fill_(1.)

        if isinstance(module, nn.LayerNorm) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, investment_embed, feature):

        embed = self.investment_embedding(investment_embed)
        embed = self.embedding_proj(embed).squeeze(dim=1)
        feature = torch.cat([feature, embed], dim=1)

        # encoded embedding
        seq_emb = self.encoder(feature)

        # prediction
        reg_pred = self.reg_layer(seq_emb)
        class_pred = self.class_layer(seq_emb)

        return reg_pred, class_pred
