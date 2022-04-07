import os


class Config:
    # config settings
    def __init__(self,
                 model_type="ffnn",
                 seed=2020,
                 fold=0,
                 batch_size=64,
                 accumulation_steps=1,
                 hidden_size=64,
                 dropout=0.15,
                 lr_scheduler_name="CyclicLR",
                 weight_decay=2e-2,
                 l1_weight=2e-4,
                 ):

        # data configs
        self.data_dir = "../inputs/ubiquant-market-prediction"

        self.train_data_full = os.path.join(self.data_dir, "train_normalized.pkl")
        self.test_data = os.path.join(self.data_dir, "train_normalized.pkl")

        self.target_cols = [
            "target_demean_normalized",
            "target_normalized",
            "avg_target_demean_normalized",
            "avg_target_normalized",
        ]
        self.target_cols_orig = [
            "target",
        ]

        # cross validation configs
        self.split = "GroupKFold"
        self.fold = fold
        self.seed = seed
        self.num_workers = 4
        self.batch_size = batch_size
        self.val_batch_size = 2048

        # setting
        self.reuse_model = False
        self.freeze_encoder = False
        self.load_from_load_from_data_parallel = False
        self.load_pretrain = False
        self.data_parallel = False  # enable data parallel training
        self.apex = True  # enable mix precision training
        self.load_optimizer = False
        self.skip_layers = []

        # model common config
        self.model_type = model_type
        self.model_name = "UMPModel"
        self.feature_size = 300
        self.hidden_size = hidden_size
        self.dropout = dropout

        # optimizer
        self.optimizer_name = "AdamW"
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 2

        # negtive sample weights
        self.neg_weight = 1

        # online hard example mining weights
        self.ohem_weight = 1

        # direction weight
        self.direction_weight = 1

        # lr scheduler, can choose to use proportion or steps
        # self.lr_scheduler_name = "CyclicLR"
        self.lr_scheduler_name = lr_scheduler_name
        self.warmup_epoches = 1
        self.warmup_proportion = 0
        self.warmup_steps = 40

        # lr
        self.lr = 8e-3
        self.warmup_lr = 2e-5
        self.min_lr = 2e-5
        self.weight_decay = weight_decay

        # L1 Norm
        self.with_l1 = True
        self.l1_weight = l1_weight

        # gradient accumulation
        self.accumulation_steps = accumulation_steps
        # epochs
        self.num_epoch = 40
        self.num_adjust_bn_epoch = 1
        # saving rate
        self.saving_rate = 1
        # early stopping
        self.early_stopping = 5
        # progress rate
        self.progress_rate = 1

        # path, specify the path for saving model
        self.checkpoint_pretrain = os.path.join("../ckpts/Pretrained", self.model_type, "pytorch_model_masked.pth")
        self.model_folder = os.path.join("../ckpts/", self.model_type + "_seed_" + str(seed))
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)

        self.checkpoint_folder = os.path.join(self.model_folder, "fold-{}".format(str(fold))
                                              + "-ohem-{}".format(self.ohem_weight)
                                              + "-" + self.lr_scheduler_name
                                              + "-lr-{}".format(str(self.lr))
                                              + "-hidden-{}".format(str(self.hidden_size))
                                              + "-dropout-{}".format(str(self.dropout))
                                              + "-l2-{}".format(str(self.weight_decay))
                                              + "-l1-{}".format(str(self.l1_weight))
                                              + "-epoch-{}".format(self.num_epoch)
                                              + "-direction-{}".format(self.direction_weight)
                                              )

        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)

        self.save_point = os.path.join(self.checkpoint_folder, "{}_step_{}_epoch.pth")
        self.load_points = [p for p in os.listdir(self.checkpoint_folder) if p.endswith(".pth")]
        if len(self.load_points) != 0:
            self.load_point = sorted(self.load_points, key=lambda x: int(x.split("_")[0]))[-1]
            self.load_point = os.path.join(self.checkpoint_folder, self.load_point)
        else:
            self.reuse_model = False
