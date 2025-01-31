from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """ Trainer object for the TomoSAR2Height network.

    Args:
        model (nn.Module): Network tomosar2height
        optimizer (optimizer): PyTorch optimizer object
        device (device): PyTorch device
        optimize_every (int): Gradient accumulation steps
        use_cloud (bool): Flag to use point cloud as input
        use_image (bool): Flag to use image as input
        use_footprint (bool): Flag to use footprint as input
        weight_ce (float): Weight for cross-entropy loss
    """

    def __init__(self, model: nn.Module, optimizer, device=None, optimize_every=1,
                 use_cloud=False, use_image=False, use_footprint=False, weight_ce=10.):
        self.model: nn.Module = model
        self.optimizer = optimizer
        self.device = device

        self.loss_ce = nn.BCEWithLogitsLoss(reduction='mean')
        self.loss_l1 = nn.L1Loss(reduction='mean')
        self.weight_ce = weight_ce

        self.optimizer.zero_grad()

        # Gradient accumulation
        self.optimize_every = optimize_every
        self.accumulated_steps = 0

        self.accumulated_loss = 0.0
        self.accumulated_loss_dict = {'loss_ce': 0.0, 'loss_l1': 0.0}
        self.last_avg_loss = 0.0
        self.last_avg_loss_dict = {'loss_ce': 0.0, 'loss_l1': 0.0}

        self.use_cloud = use_cloud
        self.use_image = use_image
        self.use_footprint = use_footprint

    def train_step(self, data):
        """ Performs a training step.

        Args:
            data (dict): Data dictionary
        """
        device = self.device

        input_cloud = data.get('inputs').to(device) if self.use_cloud else None
        input_image = data.get('image').to(device) if self.use_image else None
        dsm_gt = data.get('dsm').to(device)[None, ...]

        self.model.train()

        pa, pb = self.model(input_cloud=input_cloud, input_image=input_image)

        loss_l1 = self.loss_l1(pa.squeeze(), dsm_gt.squeeze().float())
        if self.use_footprint:
            loss_ce = self.weight_ce * self.loss_ce(pb.squeeze(), (dsm_gt.squeeze() > 0.0001).float())
        else:
            loss_ce = torch.tensor(0.0, device=device)

        loss = loss_l1 + loss_ce
        loss.backward()

        self.accumulated_steps += 1
        self.accumulated_loss += loss.detach()
        self.accumulated_loss_dict['loss_ce'] += loss_ce.detach()
        self.accumulated_loss_dict['loss_l1'] += loss_l1.detach()

        # Gradient accumulation
        if self.accumulated_steps == self.optimize_every:
            self.optimizer.step()
            with torch.no_grad():
                self.last_avg_loss = self.accumulated_loss / self.optimize_every
                self.last_avg_loss_dict = {
                    key: value / self.optimize_every
                    for key, value in self.accumulated_loss_dict.items()
                }
            self.accumulated_loss = 0.0
            self.accumulated_steps = 0
            self.accumulated_loss_dict = {key: 0.0 for key in self.accumulated_loss_dict.keys()}
            self.optimizer.zero_grad()

    def evaluate(self, val_loader):
        """
        Performs an evaluation.

        Args:
            val_loader (DataLoader): PyTorch DataLoader

        Returns:
            dict: Dictionary of evaluation metrics
        """
        metric_ls_dict = defaultdict(list)

        for data in tqdm(val_loader, desc="Validation"):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                metric_ls_dict[k].append(v)

        metric_dict = {
            k: torch.tensor(v).mean().item()
            for k, v in metric_ls_dict.items()
        }

        return metric_dict

    def eval_step(self, data):
        """ Performs an evaluation step.

        Args:
            data (dict): Data dictionary

        Returns:
            dict: Dictionary of evaluation metrics for the step
        """
        self.model.eval()
        device = self.device

        input_cloud = data.get('inputs').to(device) if self.use_cloud else None
        input_image = data.get('image').to(device) if self.use_image else None
        dsm_gt = data.get('dsm').to(device)[None, ...]

        with torch.no_grad():
            pa, pb = self.model(input_cloud=input_cloud, input_image=input_image)
            loss_l1 = self.loss_l1(pa.squeeze(), dsm_gt.squeeze().float())
            if self.use_footprint:
                loss_ce = self.weight_ce * self.loss_ce(pb.squeeze(), (dsm_gt.squeeze() > 0.00001).float())
            else:
                loss_ce = torch.tensor(0.0, device=device)

            loss = loss_l1 + loss_ce

        return {
            'loss': loss.item(),
            'loss_l1': loss_l1.item(),
            'loss_ce': loss_ce.item()
        }
