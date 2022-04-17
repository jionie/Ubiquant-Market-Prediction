import os
import numpy as np
import pandas as pd
from joblib import load
import torch
from torch.utils.data import DataLoader


class UMPDataset:
    def __init__(self,
                 config,
                 sample_indices,
                 investment_embed,
                 features,
                 targets,
                 target_orig,
                 mode="train",
                 ):

        self.config = config
        self.sample_indices = sample_indices
        self.num_targets = len(self.config.target_cols)

        self.investment_embed = investment_embed
        self.features = features
        self.targets = targets
        self.target_orig = target_orig
        self.mode = mode

    def __getitem__(self, idx):

        # get investment embed
        if self.mode == "train":
            if np.random.random(1) < self.config.investment_embed_rate:
                investment_embed = np.ones_like(self.investment_embed[idx], dtype=np.int16) * 3578
            else:
                investment_embed = self.investment_embed[idx]
        else:
            investment_embed = self.investment_embed[idx]

        # get feature
        feature = np.nan_to_num(self.features[idx], nan=0, posinf=0, neginf=0)

        # get target
        if self.targets is not None:
            target = self.targets[idx]
        else:
            target = None

        if self.target_orig is not None:
            target_orig = self.target_orig[idx]
        else:
            target_orig = None

        return investment_embed, feature, target, target_orig

    def __len__(self):
        return len(self.sample_indices)


def collate(batch):
    investment_embeds = []
    features = []
    targets = []
    targets_orig = []

    for (investment_embed, feature, target, target_orig) in batch:
        investment_embeds.append(investment_embed)
        features.append(feature)
        targets.append(target)
        targets_orig.append(target_orig)

    investment_embeds = torch.from_numpy(np.stack(investment_embeds)).contiguous().int()

    features = torch.from_numpy(np.stack(features)).contiguous().float()

    if targets[0] is None:
        targets = None
    else:
        targets = torch.from_numpy(np.stack(targets)).contiguous().float()

    if targets_orig[0] is None:
        targets_orig = None
    else:
        targets_orig = torch.from_numpy(np.stack(targets_orig)).contiguous().float()

    return investment_embeds, features, targets, targets_orig


def get_train_val_loader(config):
    if config.split == "full":
        print("full dataset mode")

        train_df = pd.read_pickle(config.train_data_full)

        feature_cols = ["f_{}".format(feature_idx) for feature_idx in range(300)]
        train_investment_embed = train_df[config.investment_embed_col].values
        train_features = train_df[feature_cols].values
        train_targets = train_df[config.target_cols].values
        train_target_orig = train_df[config.target_cols_orig].values

        train_dataset = UMPDataset(
            config=config,
            sample_indices=range(train_targets.shape[0]),
            features=train_features,
            targets=train_targets,
            target_orig=train_target_orig,
            investment_embed=train_investment_embed,
            mode="train"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate,
        )

        val_dataset = UMPDataset(
            config=config,
            sample_indices=range(train_targets.shape[0]),
            features=train_features,
            targets=train_targets,
            target_orig=train_target_orig,
            investment_embed=train_investment_embed,
            mode="val"
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate,
        )

    elif config.split == "GroupKFold":
        # train_df = pd.read_pickle(os.path.join(config.data_dir,
        #                                        "train_new_GroupKFold_{}_train.pkl".format(config.fold)))
        train_df = pd.read_pickle(os.path.join(config.data_dir,
                                               "train_normalized_GroupKFold_{}_train.pkl".format(config.fold)))
        # train_df = pd.read_pickle(os.path.join(config.data_dir,
        #                                        "train_demodel_GroupKFold_{}_train.pkl".format(config.fold)))

        feature_cols = ["f_{}".format(feature_idx) for feature_idx in range(300)]
        train_investment_embed = train_df[config.investment_embed_col].values
        train_features = train_df[feature_cols].values
        train_targets = train_df[config.target_cols].values
        train_target_orig = train_df[config.target_cols_orig].values

        train_dataset = UMPDataset(
            config=config,
            sample_indices=range(train_targets.shape[0]),
            features=train_features,
            targets=train_targets,
            target_orig=train_target_orig,
            investment_embed=train_investment_embed,
            mode="train"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate,
        )

        # val_df = pd.read_pickle(os.path.join(config.data_dir,
        #                                      "train_new_GroupKFold_{}_val.pkl".format(config.fold)))
        val_df = pd.read_pickle(os.path.join(config.data_dir,
                                             "train_normalized_GroupKFold_{}_val.pkl".format(config.fold)))
        # val_df = pd.read_pickle(os.path.join(config.data_dir,
        #                                      "train_demodel_GroupKFold_{}_val.pkl".format(config.fold)))

        val_features = val_df[feature_cols].values
        val_investment_embed = val_df[config.investment_embed_col].values
        val_targets = val_df[config.target_cols].values
        val_target_orig = val_df[config.target_cols_orig].values

        val_dataset = UMPDataset(
            config=config,
            sample_indices=range(val_targets.shape[0]),
            features=val_features,
            targets=val_targets,
            target_orig=val_target_orig,
            investment_embed=val_investment_embed,
            mode="val"
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate,
        )

    else:

        raise NotImplementedError

    return train_loader, val_loader


def get_test_loader(config, online=False):
    test_df = load(config.test_data)

    feature_cols = ["f_{}".format(feature_idx) for feature_idx in range(300)]
    test_investment_embed = test_df[config.investment_embed_col].values
    test_features = test_df[feature_cols].values

    if not online:
        test_targets = test_df[config.target_cols].values
        test_target_orig = test_df[config.target_cols_orig].values
    else:
        test_targets = None
        test_target_orig = None

    test_dataset = UMPDataset(
        config,
        sample_indices=range(test_targets.shape[0]),
        features=test_features,
        targets=test_targets,
        target_orig=test_target_orig,
        investment_embed=test_investment_embed,
        mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.val_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate,
    )

    return test_loader
