from datetime import datetime
import numpy as np
import time
import torch
from torch.nn import DataParallel
import tqdm
import yaml

from ..tools import AttrDict, load_csv_dict


__all__ = [
    "SupervisedLearner",
]


def load_config(
        config_path: str,
) -> AttrDict:
    from ..tools import make_attrdict

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return make_attrdict(config)


class Trainer:
    def __init__(
            self,
            train_dataset,
            val_dataset,
            model,
            config_path: str,
            use_cuda: bool = False,
    ) -> None:
        from .criterion import build_criterion
        from .dataloader import build_dataloader
        from .lr_scheduler import build_scheduler
        from .optimizer import build_optimizer

        self.model = model

        self.model.model = DataParallel(
            module=self.model.model,
            device_ids=list(np.arange(torch.cuda.device_count())),
        )

        self.config = load_config(config_path)

        self.epochs = self.config.EPOCH

        self.train_dataloader = build_dataloader(train_dataset, self.config)
        self.val_dataloader = build_dataloader(val_dataset, self.config) if val_dataset is not None else None

        self.criterion = build_criterion(self.config)

        self.optimizer = build_optimizer(model.model, self.config)

        self.scheduler = build_scheduler(self.optimizer, self.config)

        self.use_cuda = use_cuda

        self.datetime = datetime.now().strftime("%y%m%d%H%M%S")

    def run(
            self,
            weights_save_root: str,
            log_save_root: str,
            weights_save_period: int = 1,
            start_epoch: int = 1,
    ) -> None:
        from ..tools import makedir, save_dictionary_in_csv

        best_eval_acc = 0.0

        log_save_dir = \
            f"{log_save_root}/" \
            f"{self.train_dataloader.dataset.name}/" \
            f"{self.model.name}-{self.datetime}"

        weights_save_dir = \
            f"{weights_save_root}/" \
            f"{self.train_dataloader.dataset.name}/" \
            f"{self.model.name}_{self.datetime}"
        
        if start_epoch == 1:
            makedir(log_save_dir)
            makedir(weights_save_dir)
            
            # save train configuration
            with open(f"{log_save_dir}/config.yaml", "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)

            # save initial model
            print(f"Saving initial weights to {weights_save_dir}...\n")
            torch.save(
                self.model.model.state_dict(),
                f"{weights_save_dir}/{self.model.name}-init.pth"
            )

            log = {
                "epoch": [],
                "lr": [],

                "train_time": [],
                "train_loss": [],
                "train_acc": [],
            }

            if self.val_dataloader is not None:
                log["eval_time"] = []
                log["eval_loss"] = []
                log["eval_acc"] = []

        else:
            log = load_csv_dict(
                csv_path=f"{log_save_dir}/log.csv",
                index_col=None,
            )

        for epoch in range(start_epoch, self.epochs + 1):
            # train
            lr, train_time, train_loss, train_acc = \
                self.train(self.model.model, self.train_dataloader, epoch)

            log["epoch"].append(epoch)
            log["lr"].append(lr)

            log["train_time"].append(train_time)
            log["train_loss"].append(train_loss)
            log["train_acc"].append(train_acc)

            self.scheduler.step()

            # eval
            if self.val_dataloader is not None:
                eval_time, eval_loss, eval_acc = \
                    self.eval(self.model.model, self.val_dataloader, epoch)

                log["eval_time"].append(eval_time)
                log["eval_loss"].append(eval_loss)
                log["eval_acc"].append(eval_acc)

            else:
                eval_acc = train_acc

            # save best model weights
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc

                print(f"Saving best weights to {weights_save_dir}... (Epoch: {epoch})\n")
                torch.save(
                    self.model.model.state_dict(),
                    f"{weights_save_dir}/{self.model.name}-best.pth"
                )

            # save model weights per weights_save_period
            if not epoch % weights_save_period:
                print(f"Saving weights to {weights_save_dir}...\n")
                torch.save(
                    self.model.model.state_dict(),
                    f"{weights_save_dir}/{self.model.name}-epoch_{epoch}.pth"
                )

            # save train log
            save_dictionary_in_csv(
                dictionary=log,
                save_dir=log_save_dir,
                save_name="log",
                index_col="epoch",
            )

            # save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": self.model.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                f"{log_save_dir}/checkpoint.pt",
            )

    def train(self, model, dataloader, epoch):
        start = time.time()

        model.train()

        train_loss = 0.0
        train_acc = 0.0

        lr = self.optimizer.param_groups[0]["lr"]

        for (data, targets) in tqdm.tqdm(
                dataloader,
                desc=f"[EPOCH {epoch}/{self.epochs}] TRAIN (LR: {lr:0.8f})"
        ):
            if self.use_cuda:
                data = data.to(torch.device("cuda"))
                targets = targets.to(torch.device("cuda"))

            self.optimizer.zero_grad()

            outputs = model(data)

            loss = self.criterion(outputs, targets)
            loss.requires_grad_(True)
            loss.backward()

            self.optimizer.step()

            train_loss += loss.item()

            _, preds = outputs.max(1)
            train_acc += float(preds.eq(targets).sum().detach().cpu())

        finish = time.time()

        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader.dataset)

        print(f"TRAIN LOSS: {train_loss:.8f}\tTRAIN ACC: {(train_acc * 100):.4f}%\n")

        return self.optimizer.param_groups[0]["lr"], finish - start, train_loss, train_acc

    @torch.no_grad()
    def eval(self, model, dataloader, epoch):
        start = time.time()

        model.eval()

        eval_loss = 0.0
        eval_acc = 0.0

        for (data, targets) in tqdm.tqdm(
                dataloader,
                desc=f"[EPOCH {epoch}/{self.epochs}] EVAL"
        ):
            if self.use_cuda:
                data = data.to(torch.device("cuda"))
                targets = targets.to(torch.device("cuda"))

            outputs = model(data)

            loss = self.criterion(outputs, targets)
            eval_loss += loss.item()

            _, preds = outputs.max(1)
            eval_acc += float(preds.eq(targets).sum().detach().cpu())

        finish = time.time()

        eval_loss = eval_loss / len(dataloader)
        eval_acc = eval_acc / len(dataloader.dataset)

        print(f"EVAL LOSS: {eval_loss:.8f}\tEVAL ACC: {(eval_acc * 100):.4f}%\n")

        return finish - start, eval_loss, eval_acc

    def resume(
            self,
            weights_save_root: str,
            log_save_root: str,
            prev_datetime: str,
            weights_save_period: int = 1,
    ) -> None:
        self.datetime = prev_datetime

        log_save_dir = \
            f"{log_save_root}/" \
            f"{self.train_dataloader.dataset.name}/" \
            f"{self.model.name}-{self.datetime}"

        checkpoint = torch.load(f"{log_save_dir}/checkpoint.pt")

        stopped_epoch = checkpoint["epoch"]
        self.model.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.run(
            weights_save_root,
            log_save_root,
            weights_save_period,
            start_epoch=stopped_epoch + 1,
        )


class SupervisedLearner(Trainer):
    def __init__(
            self,
            train_dataset,
            val_dataset,
            model,
            config_path: str,
            use_cuda: bool = False,
    ) -> None:
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            config_path=config_path,
            use_cuda=use_cuda,
        )
