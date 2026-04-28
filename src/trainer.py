# from math import e
from loaders.syn_prim_dataset import *
from models.model import Net, init_model
from utils.utils import *
import torch
import numpy as np
from tqdm import tqdm

# import random
# import cv2
# from scipy.spatial import distance
# from os import path
from configs import *
from torch_geometric.data import DataLoader, Batch

# from torch.utils.data import DataLoader as UtilsDataLoader
# from torch.utils.data import Sampler, BatchSampler
from torch import nn

# import visdom
from utils.log import log
from torch.utils.tensorboard import SummaryWriter
from utils.checks import check_grads, check_valid


# from evaluation import test_rt, test_match
from torch_scatter import scatter_std
from loaders.modelnet_dataset import *
from collections import Counter
from evaluation.eval_rigid import EvaluatorRigid
from evaluation.eval_deform import EvaluatorDeform


class trainer:
    def __init__(
        self,
        settings,
        loader_train,
        loader_val,
        surface_type,
        dataset_test=None,
        pretrain=True,
        model_path=None,
    ):
        self.settings = settings
        self.pretrain = pretrain
        

        self.report_k = 1

        self.criterion_cos = torch.nn.CosineSimilarity()
        self.criterion1 = torch.nn.L1Loss(reduction="mean")
        self.criterion2 = torch.nn.MSELoss(reduction="mean")

        self.model, optimizer_dict, epoch = init_model(model_path, self.settings)
        self.e = epoch
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.settings.learning_rate
        )

        if len(optimizer_dict):
            try:
                self.optimizer.load_state_dict(optimizer_dict)
                print("adam reloaded")
            except:
                pass

        for group in self.optimizer.param_groups:
            if "initial_lr" not in group:
                for group in self.optimizer.param_groups:
                    group.setdefault("initial_lr", self.settings.learning_rate)

        self.optimizer.lr = 1e-3
        for g in self.optimizer.param_groups:
            group.setdefault("initial_lr", self.settings.learning_rate)
            g["lr"] = 0.001
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, [], 1.0, last_epoch=self.e - 1
        )

        print("last epo:", self.e)
        print("current lr", self.scheduler.get_last_lr())

        self.triplet_loss = nn.TripletMarginLoss(
            margin=self.settings.m, p=2, reduction="mean"
        )

        # self.plotter = VisdomLinePlotter(
        #     env_name=self.tag + '-' + ('init' if pretrain else 'main'))
        self.writer = SummaryWriter(log_dir=get_log_path(self.settings))

        self.dataset_test = dataset_test
        self.loader_train = loader_train
        self.loader_val = loader_val

        if surface_type == "rigid":
            self.evaluator = EvaluatorRigid(self.writer, visualize=False)
        elif surface_type == "deform":
            self.evaluator = EvaluatorDeform(self.writer, visualize=False)
        else:
            self.evaluator = None

        # self.l_train_loss, self.d_train_loss, self.acc_train, self.avg_pos_t, self.avg_neg_t, self.c_train_loss, self.sg_train_loss = 0, 0, 0, 0, 0, 0, 0
        # self.l_valid_loss, self.d_valid_loss, self.acc_loss, self.avg_pos_v, self.avg_neg_v, self.c_valid_loss, self.sg_valid_loss = 0, 0, 0, 0, 0, 0, 0
        self.metrics_train = Counter({})
        self.metrics_val = Counter({})
        self.metrics_train_ctr = Counter({})
        self.metrics_val_ctr = Counter({})
        self.metrics_train_win = Counter({})
        self.metrics_val_win = Counter({})
        self.metrics_train_ctr_win = Counter({})
        self.metrics_val_ctr_win = Counter({})

    def add_metric(self, name, value, train=True):
        if train:
            self.metrics_train = self.metrics_train + (Counter({name: value}))
            self.metrics_train_ctr = self.metrics_train_ctr + (Counter({name: 1}))
            self.metrics_train_win = self.metrics_train_win + (Counter({name: value}))
            self.metrics_train_ctr_win = self.metrics_train_ctr_win + (
                Counter({name: 1})
            )
        else:
            self.metrics_val = self.metrics_val + (Counter({name: value}))
            self.metrics_val_ctr = self.metrics_val_ctr + (Counter({name: 1}))
            self.metrics_val_win = self.metrics_val_win + (Counter({name: value}))
            self.metrics_val_ctr_win = self.metrics_val_ctr_win + (Counter({name: 1}))

    def plot_win(self):
        for loss_k, loss_v in sorted(dict(self.metrics_train_win).items()):
            # self.plotter.plot(loss_k, 'train', loss_k, self.e,
            #                   loss_v/self.metrics_train_ctr_win[loss_k])
            self.writer.add_scalar(
                loss_k, loss_v / self.metrics_train_ctr_win[loss_k], self.e
            )

        for loss_k, loss_v in sorted(dict(self.metrics_val_win).items()):
            # self.plotter.plot(loss_k, 'val', loss_k, self.e,
            #                   loss_v/self.metrics_val_ctr_win[loss_k])
            self.writer.add_scalar(
                loss_k, loss_v / self.metrics_val_ctr_win[loss_k], self.e
            )

        self.metrics_train_win = Counter({})
        self.metrics_val_win = Counter({})
        self.metrics_train_ctr_win = Counter({})
        self.metrics_val_ctr_win = Counter({})

    def plot(self):
        s = self.settings
        for loss_k, loss_v in sorted(dict(self.metrics_train).items()):
            # self.plotter.plot(loss_k, 'train', loss_k, self.e,
            #                   loss_v/self.metrics_train_ctr[loss_k])
            self.writer.add_scalar(
                loss_k + "/train", loss_v / self.metrics_train_ctr[loss_k], self.e
            )

        for loss_k, loss_v in sorted(dict(self.metrics_val).items()):
            # self.plotter.plot(loss_k, 'val', loss_k, self.e,
            #                   loss_v/self.metrics_val_ctr[loss_k])
            self.writer.add_scalar(
                loss_k + "/val", loss_v / self.metrics_val_ctr[loss_k], self.e
            )

        self.metrics_train = Counter({})
        self.metrics_val = Counter({})
        self.metrics_train_ctr = Counter({})
        self.metrics_val_ctr = Counter({})

    def train(self):

        s = self.settings

        for self.e in range(self.e, s.epoch_size):
            """"""
            log(
                "-------Starting epoch {} (lr {})--------".format(
                    self.e, self.scheduler.get_last_lr()
                ),
                str(s.to_string()),
            )
            train_loader_iter = iter(self.loader_train)
            self.run_epoch(train_loader_iter, train_val=True)
            self.scheduler.step()
            if len(self.loader_val):
                with torch.no_grad():
                    validation_loader_iter = iter(self.loader_val)
                    self.run_epoch(validation_loader_iter, train_val=False)

            self.plot()

            result_dict = {}
            if (
                self.e % 1 == 0
                and self.pretrain is not True
            ):
                if self.dataset_test is not None and self.evaluator is not None:
                    result_dict = self.evaluator.eval(
                        test_data=self.dataset_test,
                        net=self.model,
                        epoch=self.e,
                    )
                    self.model.train()
            for k, v in result_dict.items():
                self.add_metric(k, v, False)
            try:
                from wandb_logger import log as _wb_log
                wandb_payload = {"epoch": self.e,
                                 "lr": float(self.scheduler.get_last_lr()[0])}
                for k, v in result_dict.items():
                    wandb_payload[f"eval/{k}"] = v
                _wb_log(wandb_payload, step=self.e)
            except Exception:
                pass
            torch.save(
                {
                    "epoch": self.e,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config_dict": self.model.config_dict,
                },
                get_model_path(self.pretrain, s),
            )
            torch.save(
                {
                    "epoch": self.e,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config_dict": self.model.config_dict,
                },
                get_model_path(self.pretrain, s, epoch=str(self.e)),
            )

        log("pretraining done.", str(s.to_string()))
        self.writer.close()

    def run_epoch(self, loader_iter, train_val):

        s = self.settings

        samples_length = (int)(len(loader_iter))

        shape_start_epoch = 10

        matching_loss_weight = 0 if self.e < shape_start_epoch else 1
        graphite_loss_weight = 1 if self.e < shape_start_epoch else 1

        if (
            # VERSION == 2
            self.e >= shape_start_epoch
            and not self.model.config_dict["graph_match"]
        ):
            self.model.config_dict["graph_match"] = True
            self.optimizer.lr = 1e-4  # 1e-4
            for g in self.optimizer.param_groups:
                g.setdefault("initial_lr", self.settings.learning_rate)
                g["lr"] = 1e-4
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, [], 1.0, last_epoch=self.e - 1
            )
            print("Activated graph matching")

        for sam_id in tqdm(range(samples_length)):
            # if VERSION == 1:
            #     data = next(loader_iter).to_data_list()
            #     input_data = Batch.from_data_list(data).to(
            #         torch.cuda.current_device()) if s.cuda else Batch.from_data_list(data)
            # if VERSION == 2:
            data = next(loader_iter)
            next_trip = data[0]
            meta = data[1]
            inputs = {
                key: Batch.from_data_list(next_trip[key], exclude_keys=["batch", "ptr"])
                for key in next_trip.keys()
            }
            if torch.cuda.is_available():
                inputs = {
                    key: inputs[key].to(torch.cuda.current_device())
                    for key in inputs.keys()
                }
            self.optimizer.zero_grad()

            frame_keys = inputs.keys()

            outputs = self.model(inputs, meta)
            # if pretrain:
            #     visualize_graph(input_data)

            # if FIXED_GRAPH:
            #     gt_labels = input_data.y.view(
            #         [len(input_data.y)//GRAPH_SIZE, GRAPH_SIZE])
            #     # gt_labels = {key: inputs[key].y.view([len(inputs[key].y)//GRAPH_SIZE, GRAPH_SIZE]) for key in inputs.keys()}
            #     gt_conf = input_data.confidence.view(
            #         [len(input_data.y)//GRAPH_SIZE, 1])
            # else:
            # gt_labels = input_data.y
            gt_labels = {key: inputs[key].y for key in frame_keys}
            # gt_conf = input_data.confidence
            gt_conf = {key: inputs[key].confidence for key in frame_keys}

            # descriptors_len=(len(output['descriptors'])//3)*3
            # for j in range(0,descriptors_len-2, 3):

            descs = {key: outputs[key]["descriptors"] for key in frame_keys}

            mutual_size = min(outputs["p"]["descriptors"].shape[0],outputs["n"]["descriptors"].shape[0])

            self.add_metric(
                "neg_dist",
                np.asscalar(
                    torch.mean(torch.sum((descs["a"] - descs["n"]) ** 2, 1) ** 0.5)
                    .cpu()
                    .detach()
                    .numpy()
                ),
                train_val,
            )
            self.add_metric(
                "pos_dist",
                np.asscalar(
                    torch.mean(torch.sum((descs["a"][:mutual_size,:] - descs["p"][:mutual_size,:]) ** 2, 1) ** 0.5)
                    .cpu()
                    .detach()
                    .numpy()
                ),
                train_val,
            )

            if self.pretrain:
                c = sum(
                    [
                        (
                            torch.argmax(gt_labels[key], dim=1)
                            == torch.argmax(outputs[key]["probabilities"], dim=1)
                        )
                        .sum()
                        .item()
                        for key in outputs.keys()
                    ]
                )
            else:
                if (
                    outputs["a"]["probabilities"].shape
                    == outputs["p"]["probabilities"].shape
                ):
                    c = (
                        (
                            torch.argmax(outputs["a"]["probabilities"], dim=1)
                            == torch.argmax(outputs["p"]["probabilities"], dim=1)
                        )
                        .sum()
                        .item()
                    )
                else:
                    c = 0

            dist_mat = distance_matrix(
                outputs["a"]["descriptors"].detach().cpu().numpy(),
                outputs["p"]["descriptors"].detach().cpu().numpy(),
                2,
            )
            indices_a = np.arange(outputs["a"]["descriptors"].shape[0])
            indices_p = np.argmin(dist_mat, 1)
            c = sum(indices_a == indices_p) / len(indices_p)

            self.add_metric("corrects", c, train_val)

            if 'partial_matches' in meta.keys():
                a_partial = outputs["a"]["descriptors"][meta['partial_matches'].squeeze(),:]
                n_partial = outputs["n"]["descriptors"][meta['partial_matches'].squeeze(),:]
                desc_unsuper = self.triplet_loss(
                    a_partial,
                    outputs["p"]["descriptors"],
                    n_partial,
                )
            else:

                desc_unsuper = self.triplet_loss(
                    outputs["a"]["descriptors"],
                    outputs["p"]["descriptors"],
                    outputs["n"]["descriptors"],
                )
            # pos=criterion2(output['descriptors'][range(0, s.mb_size, 3)],output['descriptors'][range(1, s.mb_size, 3)])
            # neg=criterion2(output['descriptors'][range(0, s.mb_size, 3)],output['descriptors'][range(2, s.mb_size, 3)])
            # a1=pos / ( neg + pos* s.m )
            desc_loss = desc_unsuper
            self.add_metric("desc_loss", desc_loss.item(), train_val)

            if "losses" in outputs["matching_results"].keys():

                matching_loss = outputs["matching_results"]["losses"]["loss_match"]
                reg_loss = outputs["matching_results"]["losses"]["loss_grad"]
                self.add_metric("matching_loss", matching_loss.item(), train_val)
                self.add_metric("reg_loss", reg_loss.item(), train_val)
                matching_losses = reg_loss + matching_loss
            else:
                matching_losses = torch.tensor(0)

            conf_super = sum(
                [
                    self.criterion2(gt_conf[key], outputs[key]["confidence"])
                    for key in frame_keys
                ]
            )
            conf_unsuper = sum(
                [
                    self.criterion2(
                        scatter_std(
                            outputs[key]["probabilities"], inputs[key].batch, dim=0
                        ),
                        outputs[key]["confidence"],
                    )
                    for key in frame_keys
                ]
            )
            conf_loss = conf_super if self.pretrain else conf_unsuper
            self.add_metric("conf_loss", conf_loss.item(), train_val)
            value_super = sum(
                [
                    self.criterion1(gt_labels[key], outputs[key]["probabilities"])
                    for key in frame_keys
                ]
            )
            if (
                outputs["a"]["probabilities"].shape
                == outputs["p"]["probabilities"].shape
            ):
                value_unsuper = self.criterion1(
                    outputs["a"]["probabilities"], outputs["p"]["probabilities"]
                )
            else:
                value_unsuper = torch.zeros(1).to(torch.cuda.current_device())
            value_loss = value_super if self.pretrain else value_unsuper
            self.add_metric("value_loss", value_loss.item(), train_val)

            graphite_loss = s.alpha * desc_loss  # + c_loss
            self.add_metric("graphite_loss", graphite_loss.item(), train_val)
            # t_loss=d_loss

            total_loss = (
                graphite_loss * graphite_loss_weight
                + matching_losses * matching_loss_weight
            )
            self.add_metric("total_loss", total_loss.item(), train_val)

            if train_val:
                if check_valid(total_loss):
                    break
                total_loss.backward()
                self.optimizer.step()
                if check_valid(self.model.state_dict()):
                    check_grads(self.model.state_dict())

            if sam_id % s.report_size == 0 and train_val and sam_id > 0:
                self.plot_win

        # length_loss = s.mb_size*samples_length
        # length_desc = s.mb_size*samples_length
