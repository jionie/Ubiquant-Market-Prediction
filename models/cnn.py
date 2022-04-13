import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, keras_init=True):

        super(CNNModel, self).__init__()

        self.stem_cnn = nn.Sequential(
            nn.Linear(300, 256),
            nn.BatchNorm1d(256, momentum=0.1, affine=False),
            nn.SiLU(),
        )
        
        cnn_dim = 16
        dropout = 0.5

        self.conv = nn.Sequential(
            nn.Conv1d(1, cnn_dim, 4, stride=1),
            nn.BatchNorm1d(cnn_dim, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_dim, cnn_dim * 2, 4, stride=4),
            nn.BatchNorm1d(cnn_dim * 2, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_dim * 2, cnn_dim * 4, 4, stride=4),
            nn.BatchNorm1d(cnn_dim * 4, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_dim * 4, cnn_dim * 4, 4, stride=2),
            nn.BatchNorm1d(cnn_dim * 4, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_dim * 4, cnn_dim * 8, 4, stride=2),
            nn.BatchNorm1d(cnn_dim * 8, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.flatten = nn.Flatten(1)

        self.SE_net = nn.Sequential(
            nn.Linear(256, 32),
            nn.SiLU(),
            nn.Linear(32, 256),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64, momentum=0.1, affine=False),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 2)
        )
        
        if keras_init:
            self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if ('fc' in name) or ('stem' in name) or ('head' in name) or ('mlp' in name):
                if 'weight' in name and len(p.shape) > 1:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)
            elif 'conv' in name:
                if len(p.shape) > 1:
                    nn.init.xavier_uniform_(p.data)

    def forward(self, x):
        
        cnn_out = self.stem_cnn(x)
        cnn_out = cnn_out.reshape(cnn_out.shape[0], 1, cnn_out.shape[1])
        
        cnn_out = self.conv(cnn_out)
        cnn_out = self.flatten(cnn_out)
        
        se_weight = self.SE_net(cnn_out)
        cnn_out = cnn_out * se_weight
        
        cnn_out = self.head(cnn_out).squeeze(1)
        
        return cnn_out[:, [0]], cnn_out[:, [1]]
