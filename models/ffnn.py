import torch.nn as nn


class FFNNModel(nn.Module):
    def __init__(self, config):
        super(FFNNModel, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(config.feature_size, momentum=0.1, affine=False),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_size, config.hidden_size * 4),
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

    def forward(self, feature):

        # encoded embedding
        seq_emb = self.encoder(feature)

        # prediction
        reg_pred = self.reg_layer(seq_emb)
        class_pred = self.class_layer(seq_emb)

        return reg_pred, class_pred
