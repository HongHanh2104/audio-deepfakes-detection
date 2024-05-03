"""A generic training wrapper."""
import functools
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from kornia.losses import binary_focal_loss_with_logits

from src.aa import utils
from src.aa.aa_types import AttackEnum


LOGGER = logging.getLogger(__name__)


# def save_model(
#     model: torch.nn.Module,
#     model_dir: Union[Path, str],
#     name: str,
#     epoch: Optional[int] = None
# ) -> None:
#     full_model_dir = Path(f"{model_dir}/{name}")
#     full_model_dir.mkdir(parents=True, exist_ok=True)
#     if epoch is not None:
#         epoch_str = f"_{epoch:02d}"
#     else:
#         epoch_str = ""
#     torch.save(model.state_dict(), f"{full_model_dir}/ckpt{epoch_str}.pth")
#     LOGGER.info(f"Training model saved under: {full_model_dir}/ckpt{epoch}.pth")


class Trainer():
    """This is a lightweight wrapper for training models with gradient descent.

    Its main function is to store information about the training process.

    Args:
        epochs (int): The amount of training epochs.
        batch_size (int): Amount of audio files to use in one batch.
        device (str): The device to train on (Default 'cpu').
        batch_size (int): The amount of audio files to consider in one batch (Default: 32).
        optimizer_fn (Callable): Function for constructing the optimzer.
        optimizer_kwargs (dict): Kwargs for the optimzer.
    """

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
        use_scheduler: bool = False,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []
        self.use_scheduler = use_scheduler


def forward_and_loss(model, criterion, batch_x, batch_y, **kwargs):
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return batch_out, batch_loss


class GDTrainer(Trainer):

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        test_len: Optional[float] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
    ):
        if test_dataset is not None:
            train = dataset
            test = test_dataset
        else:
            test_len = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            lengths = [train_len, test_len]
            train, test = torch.utils.data.random_split(dataset, lengths)
            
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(2104)
        
        
        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=48,
            worker_init_fn=seed_worker,
            generator=g
        )
        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=48,
            worker_init_fn=seed_worker,
            generator=g
        )
        

        # criterion = torch.nn.BCEWithLogitsLoss()
        criterion = binary_focal_loss_with_logits
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        best_model = None
        best_acc = 0

        print(f"Starting training for {self.epochs} epochs!")

        forward_and_loss_fn = forward_and_loss

        if self.use_scheduler:
            batches_per_epoch = len(train_loader) * 2  # every 2nd epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optim,
                T_0=batches_per_epoch,
                T_mult=1,
                eta_min=5e-6,
                # verbose=True,
            )
        use_cuda = self.device != "cpu"

        for epoch in range(self.epochs):
            print(f"Epoch num: {epoch}")

            running_loss = 0
            num_correct = 0.0
            num_total = 0.0
            model.train()
            
            for i, (batch_x, _, _, batch_y) in enumerate(train_loader):
                # import pdb; pdb.set_trace()
                if i % 50 == 0:
                    lr = scheduler.get_last_lr()[0] if self.use_scheduler else self.optimizer_kwargs["lr"]

                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                batch_out, batch_loss = forward_and_loss_fn(model, criterion, batch_x, batch_y, use_cuda=use_cuda)
                batch_loss = batch_loss.mean()
                batch_pred = (torch.sigmoid(batch_out) + .5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()
                running_loss += (batch_loss.item() * batch_size)

                if i % 100 == 0:
                    print(
                         f"[{epoch:04d}][{i:05d}]: {running_loss / num_total} {num_correct/num_total*100}")

                optim.zero_grad()
                batch_loss.backward()
                optim.step()
                if self.use_scheduler:
                    scheduler.step()

            running_loss /= num_total
            train_accuracy = (num_correct / num_total) * 100

            print(f"Epoch [{epoch+1}/{self.epochs}]: train/loss: {running_loss}, train/accuracy: {train_accuracy}")

            test_running_loss = 0.0
            num_correct = 0.0
            num_total = 0.0
            model.eval()
            eer_val = 0

            for batch_x, _, _, batch_y in test_loader:
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)

                with torch.no_grad():
                    batch_pred = model(batch_x)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                batch_loss = criterion(batch_pred, batch_y).mean()

                test_running_loss += (batch_loss.item() * batch_size)

                batch_pred = torch.sigmoid(batch_pred)
                batch_pred_label = (batch_pred + .5).int()
                num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()

            if num_total == 0:
                num_total = 1

            test_running_loss /= num_total
            test_acc = 100 * (num_correct / num_total)
            print(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"test/loss: {test_running_loss}, "
                f"test/accuracy: {test_acc}, "
                f"test/eer: {eer_val}"
            )

            if best_model is None or test_acc > best_acc:
                best_acc = test_acc
                best_model = deepcopy(model.state_dict())

            print(
                f"Epoch [{epoch:04d}]: loss: {running_loss}, train acc: {train_accuracy}, test_acc: {test_acc}")

        model.load_state_dict(best_model)
        return model

