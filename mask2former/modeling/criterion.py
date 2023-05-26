# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.utils.comm import get_world_size

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        if len(targets) == 0:
            loss = torch.zeros(
                1, dtype=outputs["pred_logits"].dtype, device=outputs["pred_logits"].device
            )
            return {"loss_ce": loss}
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        if len(targets) == 0:
            loss = torch.zeros(
                1, dtype=outputs["pred_masks"].dtype, device=outputs["pred_masks"].device
            )
            return {"loss_mask": loss, "loss_dice": loss}

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


def get_hull_ratio(mask):
    from skimage.morphology import convex_hull_image # avoid import during inference

    if mask.sum() == 0:
        return 0
    hull = convex_hull_image(mask.cpu().numpy())
    return mask.sum() / hull.sum()


@torch.no_grad()
def labels_to_image(labels, mask):
    label_img = -torch.ones(mask.shape, dtype=labels.dtype, device=labels.device)
    label_img[mask] = labels
    return label_img


@torch.no_grad()
def get_kmeans_labels(mask_feature, alpha, n_clusters=5, niter=20):
    from kmeans_pytorch import kmeans # avoid import during inference

    X = mask_feature[alpha]
    n_clusters = min(n_clusters, alpha.sum().item())
    if n_clusters <= 1:
        return None, None
    kmeans_labels, _ = kmeans(
        X, num_clusters=n_clusters, device=mask_feature.device, tqdm_flag=False, iter_limit=niter
    )
    labels = -torch.ones(alpha.shape, dtype=int, device=mask_feature.device)
    labels[alpha] = kmeans_labels.to(mask_feature.device)
    return labels, n_clusters


class SetCriterionContrastive(nn.Module):
    def __init__(
        self, weight_dict, n_clusters=10, n_sample_style="choice", tau1=10.0, tau2=10.0
    ) -> None:
        super().__init__()
        assert n_sample_style in ["range", "choice"], n_sample_style

        self.weight_dict = weight_dict
        self.is_range = n_sample_style == "range"
        if isinstance(n_clusters, int):
            n_clusters = (n_clusters, n_clusters)
        if self.is_range:
            assert len(n_clusters) == 2, (
                "n_clusters must be two values using 'range' sample style." f" Got {n_clusters}!"
            )
        self.n_clusters = n_clusters
        self.tau1 = tau1
        self.tau2 = tau2

    def forward(self, outputs, targets):
        return self.loss_contrastive(outputs, targets)

    def loss_contrastive(self, outputs, targets):
        if ("mask_features" not in outputs) or (len(targets) == 0):
            return {}
        mask_features = outputs["mask_features"]
        alphas = [t["alpha"] for t in targets]
        assert len(alphas) == len(mask_features)
        if self.is_range:
            n_clusters = np.random.randint(self.n_clusters[0], self.n_clusters[1] + 1)
        else:
            n_clusters = np.random.choice(self.n_clusters)

        n_valid = 0
        loss_centroid = torch.tensor(0, dtype=mask_features.dtype, device=mask_features.device)
        loss_affinity = torch.tensor(0, dtype=mask_features.dtype, device=mask_features.device)
        for mask_feature, alpha in zip(mask_features, alphas):
            alpha = F.interpolate(
                alpha[None, None, ...].float(), size=mask_feature.shape[-2:], mode="nearest-exact"
            )[0, 0].bool()
            mask_feature = mask_feature.permute(1, 2, 0)  # CHW to HWC
            mask_feature_norm = torch.zeros_like(mask_feature)
            mask_feature_norm[alpha] = F.normalize(mask_feature[alpha])
            labels, n_clusters_actual = get_kmeans_labels(
                mask_feature_norm, alpha, n_clusters=n_clusters
            )
            if labels is None:
                continue
            centroids = []
            for lbl in range(n_clusters_actual):
                mask = labels == lbl
                if get_hull_ratio(mask) > 0.1:
                    centroid = mask_feature_norm[mask].mean(dim=0, keepdim=True)
                    dist = torch.cdist(mask_feature_norm[mask], centroid)
                    loss_affinity += torch.exp(dist / self.tau1).mean()
                    centroids.append(centroid)

            if len(centroids) > 1:
                centroids = torch.cat(centroids, dim=0)
                dist = torch.cdist(centroids, centroids)
                loss_centroid += torch.exp(-dist / self.tau2).mean()
                n_valid += 1

        loss_centroid = loss_centroid / n_valid if n_valid > 1 else loss_centroid
        loss_affinity = loss_affinity / n_valid if n_valid > 1 else loss_affinity
        return {"loss_centroid": loss_centroid, "loss_affinity": loss_affinity}
