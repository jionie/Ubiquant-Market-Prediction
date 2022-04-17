import os

os.environ["OMP_NUM_THREADS"] = "1"

# import common libraries
import random
import argparse
import numpy as np
from joblib import dump
import time

# import pytorch related libraries
import torch
from torch.nn import MSELoss, BCEWithLogitsLoss
from tensorboardX import SummaryWriter
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup

# import dataset class
from dataset.dataset import get_train_val_loader, get_test_loader

# import utils
from utils.ranger import Ranger
from utils.lrs_scheduler import WarmRestart
from utils.metric import pearson_correlation
from utils.file import Logger

# import model
from models.ffnn import FFNNModel
from models.cnn import CNNModel

# import config
from train_config import Config as TrainConfig

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--model_type", type=str, default="cnn", required=False, help="specify the model type")
parser.add_argument("--grid_search", type=bool, default=False, required=False, help="specify the grid search mode")
parser.add_argument("--seed", type=int, default=2022, required=False, help="specify the seed")
parser.add_argument("--batch_size", type=int, default=20480, required=False, help="specify the batch size")
parser.add_argument("--accumulation_steps", type=int, default=1, required=False, help="specify the accumulation_steps")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)
    os.environ["PYHTONHASHseed"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class UMP:
    def __init__(self, config):
        super(UMP).__init__()
        self.config = config
        self.setup_logger()
        self.setup_gpu()
        self.load_data()
        self.prepare_train()
        self.setup_model()

    def setup_logger(self):
        self.log = Logger()
        self.log.open((os.path.join(self.config.checkpoint_folder, "train_log.txt")), mode="a+")

    def setup_gpu(self):
        # confirm the device which can be either cpu or gpu
        self.config.use_gpu = torch.cuda.is_available()
        self.num_device = torch.cuda.device_count()
        if self.config.use_gpu:
            self.config.device = "cuda"
            if self.num_device <= 1:
                self.config.data_parallel = False
            elif self.config.data_parallel:
                torch.multiprocessing.set_start_method("spawn", force=True)
        else:
            self.config.device = "cpu"
            self.config.data_parallel = False

    def load_data(self):

        self.start = time.time()
        self.log.write("\nLoading data...\n")

        self.train_data_loader, self.val_data_loader = get_train_val_loader(
            self.config
        )

        self.test_data_loader = get_test_loader(
            self.config
        )

        self.train_data_loader_len = len(self.train_data_loader)

        self.log.write("\nLoading data finished cost time {}s\n".format(time.time() - self.start))
        self.start = time.time()

    def prepare_train(self):
        # preparation for training
        self.step = 0
        self.epoch = 0
        self.finished = False
        self.valid_epoch = 0
        self.train_loss, self.valid_loss, self.valid_metric_optimal = float("inf"), float("inf"), float("-inf")
        self.writer = SummaryWriter()

        # eval setting
        self.eval_step = int(self.train_data_loader_len * self.config.saving_rate)
        self.log_step = int(self.train_data_loader_len * self.config.progress_rate)
        self.eval_count = 0
        self.count = 0

    def pick_model(self):
        # for switching model
        if self.config.model_type == "ffnn":
            self.model = FFNNModel(self.config).to(self.config.device)
        elif self.config.model_type == "cnn":
            self.model = CNNModel(self.config).to(self.config.device)
        else:
            raise NotImplementedError

        if self.config.load_pretrain:

            self.log.write("\nLoad pretrained weights from: {}\n".format(self.config.checkpoint_pretrain))
            checkpoint_to_load = torch.load(self.config.checkpoint_pretrain, map_location=self.config.device)["model"]
            model_state_dict = checkpoint_to_load

            if self.config.data_parallel:
                state_dict = self.model.model.state_dict()
            else:
                state_dict = self.model.state_dict()

            keys = list(state_dict.keys())

            for key in keys:
                if any(s in key for s in self.config.skip_layers):
                    continue
                try:
                    state_dict[key] = model_state_dict[key]
                except:
                    print("Missing key:", key)

            if self.config.data_parallel:
                self.model.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

    def differential_lr(self, warmup=True, freeze_encoder=False):

        param_optimizer = list(self.model.named_parameters())
        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight"
        ]
        no_backbone = [
            "predefined_shared_fore.0.weight",
            "hidden_sector_shared_fore.0.weight"
            "individual_shared_fore.0.weight", 
            "reg_layer.0.weight", 
            "class_layer.0.weight"
        ]

        self.optimizer_grouped_parameters = []

        if warmup:
            if freeze_encoder:
                self.optimizer_grouped_parameters.append(
                    {"params": [p for n, p in param_optimizer if n in no_backbone
                                ],
                     "lr": self.config.warmup_lr,
                     "weight_decay": self.config.weight_decay,
                     }
                )
            else:
                self.optimizer_grouped_parameters.append(
                    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                                ],
                     "lr": self.config.warmup_lr,
                     "weight_decay": self.config.weight_decay,
                     }
                )
                self.optimizer_grouped_parameters.append(
                    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                                ],
                     "lr": self.config.warmup_lr,
                     "weight_decay": 0,
                     }
                )

        else:
            if freeze_encoder:
                self.optimizer_grouped_parameters.append(
                    {"params": [p for n, p in param_optimizer if n in no_backbone
                                ],
                     "lr": self.config.lr,
                     "weight_decay": self.config.weight_decay,
                     }
                )
            else:
                self.optimizer_grouped_parameters.append(
                    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                                ],
                     "lr": self.config.lr,
                     "weight_decay": self.config.weight_decay,
                     }
                )
                self.optimizer_grouped_parameters.append(
                    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                                ],
                     "lr": self.config.lr,
                     "weight_decay": 0,
                     }
                )

    def prepare_optimizer(self):

        # optimizer
        if self.config.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.optimizer_grouped_parameters, eps=self.config.adam_epsilon)
        elif self.config.optimizer_name == "Ranger":
            self.optimizer = Ranger(self.optimizer_grouped_parameters)
        elif self.config.optimizer_name == "AdamW":
            self.optimizer = torch.optim.AdamW(self.optimizer_grouped_parameters,
                                               eps=self.config.adam_epsilon,
                                               betas=(0.9, 0.999),
                                               )
        elif self.config.optimizer_name == "Nesterov":
            self.optimizer = torch.optim.SGD(self.optimizer_grouped_parameters,
                                             momentum=0.8,
                                             dampening=0,
                                             nesterov=True
                                             )
        else:
            raise NotImplementedError

        # lr scheduler
        if self.config.lr_scheduler_name == "WarmupCosineAnealing":
            num_train_optimization_steps = self.config.num_epoch * self.train_data_loader_len \
                                           // self.config.accumulation_steps
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.config.warmup_steps,
                                                             num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "WarmRestart":
            self.scheduler = WarmRestart(self.optimizer, T_max=5, T_mult=1, eta_min=1e-6)
            self.lr_scheduler_each_iter = False
        elif self.config.lr_scheduler_name == "WarmupLinear":
            num_train_optimization_steps = self.config.num_epoch * self.train_data_loader_len \
                                           // self.config.accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps,
                                                             num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5,
                                                                        patience=3, min_lr=self.config.min_lr,
                                                                        verbose=True)
            self.lr_scheduler_each_iter = False
        elif self.config.lr_scheduler_name == "WarmupConstant":
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
                                                               num_warmup_steps=self.config.warmup_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=[7, 28, 42],
                                                                  gamma=0.1)
            self.lr_scheduler_each_iter = False
        elif self.config.lr_scheduler_name == "CyclicLR":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                               self.config.min_lr,
                                                               self.config.lr,
                                                               step_size_up=int(self.train_data_loader_len),
                                                               step_size_down=None,
                                                               mode="triangular2",
                                                               gamma=1.0,
                                                               scale_fn=None,
                                                               scale_mode="cycle",
                                                               cycle_momentum=False,
                                                               base_momentum=0.8,
                                                               max_momentum=0.9,
                                                               last_epoch=- 1,
                                                               verbose=False
                                                               )
            self.lr_scheduler_each_iter = True
        else:
            raise NotImplementedError

        # lr scheduler step for checkpoints
        if self.lr_scheduler_each_iter:
            self.scheduler.step(self.step)
        else:
            if self.config.lr_scheduler_name != "ReduceLROnPlateau":
                self.scheduler.step(self.epoch)

    def prepare_apex(self):
        self.scaler = torch.cuda.amp.GradScaler()

    def load_check_point(self):
        self.log.write("Model loaded as {}.".format(self.config.load_point))
        checkpoint_to_load = torch.load(self.config.load_point, map_location=self.config.device)
        self.step = checkpoint_to_load["step"]
        self.epoch = checkpoint_to_load["epoch"]

        model_state_dict = checkpoint_to_load["model"]
        if self.config.load_from_load_from_data_parallel:
            model_state_dict = {k[13:]: v for k, v in model_state_dict.items()}

        if self.config.data_parallel:
            state_dict = self.model.model.state_dict()
        else:
            state_dict = self.model.state_dict()

        keys = list(state_dict.keys())

        for key in keys:
            if any(s in key for s in self.config.skip_layers):
                continue
            try:
                state_dict[key] = model_state_dict[key]
            except Exception as e:
                print("Missing key:", key)

        if self.config.data_parallel:
            self.model.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        if self.config.load_optimizer:
            self.optimizer.load_state_dict(checkpoint_to_load["optimizer"])

    def save_check_point(self):
        # save model, optimizer, and everything required to keep
        checkpoint_to_save = {
            "step": self.step,
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            # "optimizer": self.optimizer.state_dict()
        }

        save_path = self.config.save_point.format(self.step, self.epoch)
        torch.save(checkpoint_to_save, save_path)
        self.log.write("Model saved as {}.\n".format(save_path))

    def setup_model(self):

        self.log.write("\nSetting up model...\n")

        self.pick_model()

        if self.config.data_parallel:

            self.differential_lr(warmup=True, freeze_encoder=self.config.freeze_encoder)

            self.prepare_optimizer()

            if self.config.apex:
                self.prepare_apex()

            if self.config.reuse_model:
                self.load_check_point()

            self.model = torch.nn.DataParallel(self.model)

        else:
            if self.config.reuse_model:
                self.load_check_point()

            self.differential_lr(warmup=True, freeze_encoder=self.config.freeze_encoder)

            self.prepare_optimizer()

            if self.config.apex:
                self.prepare_apex()

        self.log.write("\nSetting up model finished cost time {}s\n".format(time.time() - self.start))
        self.start = time.time()

    def count_parameters(self):
        # get total size of trainable parameters
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def count_nonzero_parameters(self):
        # get total size of trainable parameters
        return sum(p.data.count_nonzero() for p in self.model.parameters() if p.requires_grad)

    def show_info(self):
        # show general information before training
        self.log.write("\n*General Setting*")
        self.log.write("\nseed: {}".format(self.config.seed))
        self.log.write("\nmodel: {}".format(self.config.model_name))
        self.log.write("\ntrainable parameters:{:,.0f}".format(self.count_parameters()))
        self.log.write("\ndevice: {}".format(self.config.device))
        self.log.write("\nuse gpu: {}".format(self.config.use_gpu))
        self.log.write("\ndevice num: {}".format(self.num_device))
        self.log.write("\noptimizer: {}".format(self.optimizer))
        self.log.write("\nlr_scheduler: {}".format(self.config.lr_scheduler_name))
        self.log.write("\nlr: {}".format(self.config.lr))
        self.log.write("\nl1 regularization: {}".format(self.config.with_l1))
        self.log.write("\nl1 regularization weight: {}".format(self.config.l1_weight))
        self.log.write("\nreuse model: {}".format(self.config.reuse_model))
        if self.config.reuse_model:
            self.log.write("\nModel restored from {}.".format(self.config.load_point))
        self.log.write("\n")

    def train_batch(
            self,
            investment_embed,
            feature,
            target,
            target_orig,
            weight
    ):
        # set input to cuda mode
        investment_embed = investment_embed.to(self.config.device).int()
        feature = feature.to(self.config.device).float()

        if target is not None:
            target = target.to(self.config.device).float()

        if target_orig is not None:
            target_orig = target_orig.to(self.config.device).float()

        with torch.autograd.set_detect_anomaly(True):

            outputs, class_outputs = self.model(investment_embed, feature)

        outputs_direction = torch.where(outputs >= 0, 1, -1)
        target_direction = torch.where(target >= 0, 1, -1)

        wrong_direction = outputs_direction * target_direction
        all_direction = target_direction * target_direction

        if self.config.apex:
            with torch.autograd.set_detect_anomaly(True):
                with torch.cuda.amp.autocast():
                    loss = self.reg_criterion(outputs, target)

                    if self.config.with_l1:
                        for param in self.model.parameters():
                            loss += torch.sum(torch.abs(param)) * self.config.l1_weight

                # weighted pos and neg loss
                if self.config.neg_weight != 1:
                    loss[target < 0] *= self.config.neg_weight

                # classification loss
                if self.config.direction_weight > 0:
                    loss += self.class_criterion(class_outputs, torch.where(target > 0, 1, 0).float()) \
                            * self.config.direction_weight

                # online hard example mining
                if self.config.ohem_weight != 1:
                    loss, _ = torch.topk(loss, int(self.config.ohem_weight * loss.size()[0]), dim=0)

                loss = torch.mean(loss) * weight
                self.scaler.scale(loss).backward()

        else:

            loss = self.reg_criterion(outputs, target)

            if self.config.with_l1:
                for param in self.model.parameters():
                    loss += torch.sum(torch.abs(param)) * self.config.l1_weight

            # weighted pos and neg loss
            if self.config.neg_weight != 1:
                loss[target < 0] *= self.config.neg_weight

            # classification loss
            if self.config.direction_weight > 0:
                loss += self.class_criterion(class_outputs, torch.where(target > 0, 1, 0).float()) \
                        * self.config.direction_weight

            # online hard example mining
            if self.config.ohem_weight != 1:
                loss, _ = torch.topk(loss, int(self.config.ohem_weight * loss.size()[0]), dim=0)

            loss = torch.mean(loss) * weight
            loss.backward()

        return loss, outputs, target, target_orig, wrong_direction, all_direction

    def train_op(self):
        self.show_info()
        self.log.write("** start training here! **\n")
        self.log.write("   batch_size=%d,  accumulation_steps=%d\n" % (self.config.batch_size,
                                                                       self.config.accumulation_steps))
        self.log.write("   experiment  = %s\n" % str(__file__.split("/")[-2:]))

        self.reg_criterion = MSELoss(reduction="none")
        self.class_criterion = BCEWithLogitsLoss(reduction="none")

        while self.epoch <= self.config.num_epoch:

            self.train_outputs = []
            self.train_targets = []
            self.train_targets_orig = []
            self.train_wrong_direction = []
            self.train_all_direction = []

            # warmup lr for parameter warmup_epoch
            if self.epoch == self.config.warmup_epoches:
                self.differential_lr(warmup=False, freeze_encoder=self.config.freeze_encoder)
                self.prepare_optimizer()

            # update lr and start from start_epoch
            if (self.epoch >= 1) and (not self.lr_scheduler_each_iter) \
                    and (self.config.lr_scheduler_name != "ReduceLROnPlateau"):
                self.scheduler.step()

            self.log.write("\n")
            self.log.write("Epoch%s\n" % self.epoch)
            self.log.write("\n")

            sum_train_loss = np.zeros_like(self.train_loss)
            sum_train = np.zeros_like(self.train_loss)

            # set model training mode
            self.model.train()

            # init optimizer
            self.model.zero_grad()

            for tr_batch_i, (investment_embed, feature, target, target_orig,) in enumerate(self.train_data_loader):

                batch_size = feature.shape[0]

                if tr_batch_i >= self.train_data_loader_len:
                    break

                rate = 0
                for param_group in self.optimizer.param_groups:
                    rate += param_group["lr"] / len(self.optimizer.param_groups)

                loss, outputs, target, target_orig, wrong_direction, all_direction = self.train_batch(
                    investment_embed,
                    feature,
                    target,
                    target_orig,
                    weight=1
                )

                if (tr_batch_i + 1) % self.config.accumulation_steps == 0:
                    # use apex
                    if self.config.apex:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm,
                                                       norm_type=2)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm,
                                                       norm_type=2)
                        self.optimizer.step()

                    self.model.zero_grad()

                    # adjust lr
                    if self.lr_scheduler_each_iter:
                        self.scheduler.step()

                    self.writer.add_scalar("train_loss", loss.item(),
                                           (self.epoch - 1) * self.train_data_loader_len * batch_size
                                           + tr_batch_i * batch_size)
                    self.step += 1

                # translate to predictions
                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                outputs = to_numpy(outputs)
                target = to_numpy(target)
                target_orig = to_numpy(target_orig)

                wrong_direction = to_numpy(torch.where(wrong_direction < 0, 1, 0)).sum()
                all_direction = to_numpy(all_direction).sum()

                self.train_outputs.append(outputs)
                self.train_targets.append(target)
                self.train_targets_orig.append(target_orig)
                self.train_wrong_direction.append(wrong_direction)
                self.train_all_direction.append(all_direction)

                sum_train_loss = sum_train_loss + np.array([loss.item() * batch_size])
                sum_train = sum_train + np.array([batch_size])

                # log for training
                if (tr_batch_i + 1) % self.log_step == 0:
                    train_loss = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train[...] = 0

                    train_outputs = np.concatenate(self.train_outputs, axis=0)[:, 0]
                    train_targets = np.concatenate(self.train_targets, axis=0)[:, 0]
                    train_targets_orig = np.concatenate(self.train_targets_orig, axis=0)
                    train_wrong_direction = np.sum(self.train_wrong_direction)
                    train_all_direction = np.sum(self.train_all_direction)

                    train_pearson = np.round(pearson_correlation(
                        train_outputs,
                        train_targets_orig
                    ), 6)

                    if train_outputs[train_outputs > 0].shape[0] > 2:
                        train_pearson_pos = np.round(pearson_correlation(
                            train_outputs[train_outputs > 0],
                            train_targets_orig[train_outputs > 0]
                        ), 6)
                    else:
                        train_pearson_pos = 0

                    if train_outputs[train_outputs < 0].shape[0] > 2:
                        train_pearson_neg = np.round(pearson_correlation(
                            train_outputs[train_outputs < 0],
                            train_targets_orig[train_outputs < 0]
                        ), 6)
                    else:
                        train_pearson_neg = 0

                    zero_param_ratio = 1 - self.count_nonzero_parameters() / self.count_parameters()
                    target_bias = np.mean(train_targets) / np.std(train_targets)
                    pred_bias = np.mean(train_outputs) / np.std(train_outputs)
                    self.log.write(
                        "lr: {} loss: {} target bias: {} pred bias: {} train_pearson: {} train_pearson_pos: {} train_pearson_neg: {} zero param ratio: {} wrong direction ratio (-1 for no valid): {}\n"
                            .format(np.around(rate, 6),
                                    np.around(train_loss[0], 4),
                                    np.around(target_bias.astype(np.float32), 4),
                                    np.around(pred_bias.astype(np.float32), 4),
                                    np.around(train_pearson, 4),
                                    np.around(train_pearson_pos, 4),
                                    np.around(train_pearson_neg, 4),
                                    np.around(zero_param_ratio.cpu().numpy(), 4),
                                    train_wrong_direction / train_all_direction,
                                    )
                    )

                if (tr_batch_i + 1) % self.eval_step == 0:
                    self.log.write("Training for one eval step finished cost time {}s\n\n".format(time.time() -
                                                                                                  self.start))
                    self.start = time.time()

                    self.evaluate_op()

                    self.log.write("Evaluating for one eval step finished cost time {}s\n\n".format(time.time() -
                                                                                                    self.start))
                    self.start = time.time()

                    self.model.train()

            if self.count >= self.config.early_stopping and self.epoch >= 30:
                break

            self.epoch += 1

        # for _ in range(self.config.num_adjust_bn_epoch):
        #
        #     self.test_op(online=False, adjust_bn=True)
        #
        #     self.log.write("Updating BN for one test step finished cost time {}s\n\n".format(time.time() -
        #                                                                                      self.start))
        #     self.start = time.time()
        #
        #     self.evaluate_op()
        #
        #     self.log.write("Evaluating for one eval step finished cost time {}s\n\n".format(time.time() -
        #                                                                                     self.start))
        #     self.start = time.time()
        #
        #     self.epoch += 1
        #     self.step += 1

    def evaluate_op(self):

        self.eval_count += 1

        # eval set metrics
        valid_loss = np.zeros(1, np.float32)
        valid_num = np.zeros_like(valid_loss)

        self.eval_outputs = []
        self.eval_targets = []
        self.eval_targets_orig = []

        self.reg_criterion = MSELoss(reduction="none")

        with torch.no_grad():
            for val_batch_i, (investment_embed, feature, target, target_orig) in enumerate(self.val_data_loader):

                # set model to eval mode
                self.model.eval()

                batch_size = feature.shape[0]

                # set input to cuda mode
                investment_embed = investment_embed.to(self.config.device).int()
                feature = feature.to(self.config.device).float()
                target = target.to(self.config.device).float()
                target_orig = target_orig.to(self.config.device).float()

                outputs, class_outputs = self.model(investment_embed, feature)

                loss = self.reg_criterion(outputs, target)

                if self.config.with_l1:
                    l1_loss = 0
                    for param in self.model.parameters():
                        l1_loss += torch.sum(torch.abs(param))

                    loss += self.config.l1_weight * l1_loss

                # classification loss
                if self.config.direction_weight > 0:
                    loss += self.class_criterion(class_outputs, torch.where(target > 0, 1, 0).float()) \
                            * self.config.direction_weight

                loss = torch.mean(loss)

                self.writer.add_scalar("val_loss", loss.item(),
                                       (self.eval_count - 1) * len(self.val_data_loader) * batch_size
                                       + val_batch_i * batch_size)

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                outputs = to_numpy(outputs)
                target_orig = to_numpy(target_orig)
                target = to_numpy(target)

                self.eval_outputs.append(outputs)
                self.eval_targets.append(target)
                self.eval_targets_orig.append(target_orig)

                valid_loss = valid_loss + np.array([loss.item() * batch_size])
                valid_num = valid_num + np.array([batch_size])

        valid_loss = valid_loss / valid_num

        eval_outputs = np.concatenate(self.eval_outputs, axis=0)[:, 0]
        eval_targets = np.concatenate(self.eval_targets, axis=0)[:, 0]
        eval_targets_orig = np.concatenate(self.eval_targets_orig, axis=0)

        eval_pearson = np.round(pearson_correlation(
            eval_outputs,
            eval_targets_orig), 6)

        if eval_outputs[eval_outputs > 0].shape[0] > 2:
            eval_pearson_pos = np.round(pearson_correlation(
                eval_outputs[eval_outputs > 0],
                eval_targets_orig[eval_outputs > 0]), 6)
        else:
            eval_pearson_pos = 0

        if eval_outputs[eval_outputs < 0].shape[0] > 2:
            eval_pearson_neg = np.round(pearson_correlation(
                eval_outputs[eval_outputs < 0],
                eval_targets_orig[eval_outputs < 0]), 6)
        else:
            eval_pearson_neg = 0

        target_bias = np.mean(eval_targets) / np.std(eval_targets)
        pred_bias = np.mean(eval_outputs) / np.std(eval_outputs)
        self.log.write(
            "eval     loss:  {} target bias: {} pred bias: {} eval_pearson:  {} eval_pearson_pos:  {} eval_pearson_neg:  {}\n"
                .format(
                np.around(valid_loss[0], 4),
                np.around(target_bias.astype(np.float32), 4),
                np.around(pred_bias.astype(np.float32), 4),
                np.around(eval_pearson, 4),
                np.around(eval_pearson_pos, 4),
                np.around(eval_pearson_neg, 4)
            )
        )
        self.log.write(
            "eval     target max:  {} target min: {} pred max: {} pred_min :{}\n"
                .format(
                eval_targets.max(),
                eval_targets.min(),
                eval_outputs.max(),
                eval_outputs.min()
            )
        )

        dump(eval_outputs,
             os.path.join(self.config.checkpoint_folder, "val_preds_fold_{}.pkl".format(self.config.fold)))
        dump(eval_targets_orig,
             os.path.join(self.config.checkpoint_folder, "val_targets_fold_{}.pkl".format(self.config.fold)))

        if self.config.lr_scheduler_name == "ReduceLROnPlateau":
            self.scheduler.step(eval_pearson)

        if abs(pred_bias) < 0.2 or self.epoch >= self.config.num_epoch:
            self.save_check_point()
            self.count = 0

        else:
            self.count += 1

    def test_op(self, online=False, adjust_bn=False):

        self.eval_count += 1
        test_loss = np.zeros(1, np.float32)
        test_num = np.zeros_like(test_loss)

        self.test_outputs = []
        self.test_targets = []
        self.test_targets_orig = []
        predictions = []

        self.reg_criterion = MSELoss(reduction="none")

        with torch.no_grad():

            # init cache
            torch.cuda.empty_cache()

            for test_batch_i, (investment_embed, feature, target, target_orig) in enumerate(self.test_data_loader):

                # set model to eval mode
                self.model.eval()

                batch_size = feature.shape[0]

                if adjust_bn:
                    for m in self.model.modules():
                        if m.__class__.__name__.startswith("BatchNorm") or m.__class__.__name__.startswith("LayerNorm"):
                            m.train()

                # set input to cuda mode
                investment_embed = investment_embed.to(self.config.device).int()
                feature = feature.to(self.config.device).float()

                if target is not None:
                    target = target.to(self.config.device).float()

                if target_orig is not None:
                    target_orig = target_orig.to(self.config.device).float()

                outputs, class_outputs = self.model(investment_embed, feature)

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                if not online and target is not None:

                    loss = self.reg_criterion(outputs, target)

                    if self.config.with_l1:
                        l1_loss = 0
                        for param in self.model.parameters():
                            l1_loss += torch.sum(torch.abs(param))

                        loss += self.config.l1_weight * l1_loss

                    # classification loss
                    if self.config.direction_weight > 0:
                        loss += self.class_criterion(class_outputs,
                                                     torch.where(target > 0, 1, 0).float()) \
                                * self.config.direction_weight

                    loss = torch.mean(loss)

                    self.writer.add_scalar("test_loss", loss.item(),
                                           (self.eval_count - 1) * len(
                                               self.test_data_loader) * batch_size + test_batch_i * batch_size)

                    outputs = to_numpy(outputs)
                    target_orig = to_numpy(target_orig)
                    target = to_numpy(target)

                    self.test_outputs.append(outputs)
                    self.test_targets.append(target)
                    self.test_targets_orig.append(target_orig)

                    test_loss = test_loss + np.array([loss.item() * batch_size])
                    test_num = test_num + np.array([batch_size])

                else:
                    predictions.append(to_numpy(outputs))

            if not online and target is not None:

                test_loss = test_loss / test_num

                test_outputs = np.concatenate(self.test_outputs, axis=0)[:, 0]
                test_targets = np.concatenate(self.test_targets, axis=0)[:, 0]
                test_targets_orig = np.concatenate(self.test_targets_orig, axis=0)

                test_pearson = np.round(pearson_correlation(test_outputs, test_targets_orig), 6)
                test_pearson_pos = np.round(pearson_correlation(test_outputs[test_outputs > 0],
                                                                test_targets_orig[test_outputs > 0]), 6)
                test_pearson_neg = np.round(pearson_correlation(test_outputs[test_outputs < 0],
                                                                test_targets_orig[test_outputs < 0]), 6)

                dump(test_outputs, os.path.join(self.config.checkpoint_folder, "test_preds.pkl"))
                dump(test_targets, os.path.join(self.config.checkpoint_folder, "test_targets.pkl"))

                if self.epoch % 100 == 0:
                    self.save_check_point()

                target_bias = np.mean(test_targets) / np.std(test_targets)
                pred_bias = np.mean(test_outputs) / np.std(test_outputs)
                self.log.write(
                    "test       loss:  {} target bias: {} pred bias: {} test_pearson:  {} test_pearson_pos:  {} test_pearson_neg:  {}\n"
                        .format(
                        np.around(test_loss[0], 4),
                        np.around(target_bias.astype(np.float32), 4),
                        np.around(pred_bias.astype(np.float32), 4),
                        np.around(test_pearson, 4),
                        np.around(test_pearson_pos, 4),
                        np.around(test_pearson_neg, 4)
                    )
                )
                self.log.write(
                    "test       target min:  {} target min: {} pred max: {} pred min: {}\n"
                        .format(
                        test_targets.max(),
                        test_targets.min(),
                        test_outputs.max(),
                        test_outputs.min()
                    )
                )
                return None

            else:
                return np.concatenate(predictions, axis=0)


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.grid_search:

        for fold in range(1):
            # train with random reformat first
            train_config = TrainConfig(
                model_type=args.model_type,
                fold=fold,
                seed=args.seed,
                batch_size=args.batch_size,
                accumulation_steps=args.accumulation_steps
            )

            # seed
            seed_everything(train_config.seed)

            # init class instance
            train_qa = UMP(train_config)

            # trainig
            train_qa.train_op()

    else:

        for fold in range(1):

            for hidden_size in [
                64,
                # 96,
                # 128,
            ]:
                for (dropout, l1_weight) in [
                    (0.1, 2e-4),
                    # (0.15, 1.5e-4),
                    # (0.15, 3e-4),
                    # (0.15, 4e-4),
                    # (0.15, 1e-3),
                    # (0.15, 2e-4),
                    # (0.15, 5e-4),
                    # (0.15, 1e-3),
                    # (0.2, 5e-5),
                    # (0.2, 1e-4),
                    # (0.2, 2e-4),
                ]:
                    for weight_decay in [
                        # 1.5e-2,
                        2e-2,
                        # 2.5e-2,
                        # 3e-2,
                        # 3.5e-2,
                        # 4e-2,
                        # 4.5e-2,
                        # 5e-2,
                        # 1e-1
                    ]:
                        train_config = TrainConfig(
                            model_type=args.model_type,
                            seed=args.seed,
                            fold=fold,
                            batch_size=args.batch_size,
                            accumulation_steps=args.accumulation_steps,
                            hidden_size=hidden_size,
                            dropout=dropout,
                            weight_decay=weight_decay,
                            l1_weight=l1_weight,
                        )

                        # seed
                        seed_everything(train_config.seed)

                        # init class instance
                        train_qa = UMP(train_config)

                        # trainig
                        train_qa.train_op()
