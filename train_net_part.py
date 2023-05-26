# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    RGBADatasetMapper,
    RGBACOCOInstanceNewBaselineDatasetMapper,
    AlphaAugCOCOInstanceNewBaselineDatasetMapper,
    AlphaAugStrongCOCOInstanceNewBaselineDatasetMapper,
    RGBAMaskFormerInstanceDatasetMapper,
    AlphaAugMaskFormerInstanceDatasetMapper,
    SemanticSegmentorWithTTA,
    build_contrastive_train_loader,
    concat_loaders,
    add_maskformer2_config,
    add_part_segmentation_config,
    _get_partimagenet_instances_meta,
    COCOEvaluatorScoreT,
)

from detectron2.data.datasets import register_coco_instances


datasets_root = "." # more flexible to the root of datasets

# ### PartImageNet ###
# dt alpha
for setting in ["", "_classless", "_classless_noannos"]:
    register_coco_instances(
        f"partimagenet_rgba_alpha0.5{setting}_indomain_train",
        {},
        f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/train_modified2_train{setting}_alphaiou0.5.json",
        f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/train",
    )
    register_coco_instances(
        f"partimagenet_rgba_alpha0.5{setting}_indomain_val",
        {},
        f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/train_modified2_val{setting}_alphaiou0.5.json",
        f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/train",
    )
    register_coco_instances(
        f"partimagenet_rgba_alpha0.5{setting}_val",
        {},
        f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/val_modified2{setting}_alphaiou0.5.json",
        f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/val",
    )
    register_coco_instances(
        f"partimagenet_rgba_alpha0.5{setting}_test",
        {},
        f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/test_modified2{setting}_alphaiou0.5.json",
        f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/test",
    )
register_coco_instances(
    "partimagenet_rgba_alpha0.5_classless_val_selftrain_pred_score0.1",
    {},
    f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/selftrain_score0.1_val_modified2_classless_alphaiou0.5.json",
    f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/val",
)
register_coco_instances(
    "partimagenet_rgba_alpha0.5_classless_indomain_train_selftrain_pred_score0.5",
    {},
    f"{datasets_root}/datasets//PartImageNet_RGBA_mask2former/selftrain/selftrain_score0.5_train_modified2_train_classless_alphaiou0.5.json",
    f"{datasets_root}/datasets/PartImageNet_RGBA_mask2former/train",
)
# ### PartImageNet ###

# ### PascalPart ###
for setting in [58, 108]:
    # dt alpha
    register_coco_instances(
        f"partpascal{setting}_rgba_alpha0.5_classless_noannos_train",
        {},
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/annotations_coco_alpha_mask2former/part{setting}/train_part{setting}_classless_noannos_alphaiou0.5.json",
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/RGBA_mask2former_crop",
    )
    register_coco_instances(
        f"partpascal{setting}_rgba_alpha0.5_classless_val",
        {},
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/annotations_coco_alpha_mask2former/part{setting}/val_part{setting}_classless_alphaiou0.5.json",
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/RGBA_mask2former_crop",
    )
    register_coco_instances(
        f"partpascal{setting}_rgba_alpha0.5_noannos_train",
        {},
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/annotations_coco_alpha_mask2former/part{setting}/train_part{setting}_noannos_alphaiou0.5.json",
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/RGBA_mask2former_crop",
    )
    register_coco_instances(
        f"partpascal{setting}_rgba_alpha0.5_noannos_val",
        {},
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/annotations_coco_alpha_mask2former/part{setting}/val_part{setting}_noannos_alphaiou0.5.json",
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/RGBA_mask2former_crop",
    )
    register_coco_instances(
        f"partpascal{setting}_rgba_alpha0.5_train",
        {},
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/annotations_coco_alpha_mask2former/part{setting}/train_part{setting}_polygon_alphaiou0.5.json",
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/RGBA_mask2former_crop",
    )
    register_coco_instances(
        f"partpascal{setting}_rgba_alpha0.5_classless_train",
        {},
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/annotations_coco_alpha_mask2former/part{setting}/train_part{setting}_classless_polygon_alphaiou0.5.json",
        f"{datasets_root}/datasets/Pascal_VOC_2010/part/RGBA_mask2former_crop",
    )
register_coco_instances(
    "partpascal_rgba_selftrain_pred_classless_score0.1",
    {},
    f"{datasets_root}/datasets/Pascal_VOC_2010/part/annotations_coco/selftrain_pred/pred_classless_v3_score0.1.json",
    f"{datasets_root}/datasets/Pascal_VOC_2010/part/RGBA_crop",
)
register_coco_instances(
    "partpascal_rgba_alpha0.5_selftrain_pred_classless_score0.1",
    {},
    f"{datasets_root}/datasets/Pascal_VOC_2010/part/annotations_coco_alpha_mask2former/selftrain_pred/pred_classless_score0.1.json",
    f"{datasets_root}/datasets/Pascal_VOC_2010/part/RGBA_mask2former_crop",
)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco" and dataset_name.startswith("part"):
            score_threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
            evaluator_list.append(
                COCOEvaluatorScoreT(
                    dataset_name,
                    output_dir=output_folder,
                    tasks=("segm",),
                    max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
                    score_threshold=score_threshold if score_threshold >= 0 else None,
                )
            )
            # evaluator_list.append(
            #     PartEvaluator(
            #         dataset_name,
            #         output_dir=output_folder,
            #         max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
            #         score_threshold=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            #     )
            # )
        elif evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
        # rgba
        elif cfg.INPUT.DATASET_MAPPER_NAME == "rgba_mask_former_instance":
            mapper = RGBAMaskFormerInstanceDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "rgba_alpha_aug_mask_former_instance":
            mapper = AlphaAugMaskFormerInstanceDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "rgba_coco_instance_lsj":
            mapper = RGBACOCOInstanceNewBaselineDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "rgba_alpha_aug_coco_instance_lsj":
            mapper = AlphaAugCOCOInstanceNewBaselineDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "rgba_alpha_aug_strong_coco_instance_lsj":
            mapper = AlphaAugStrongCOCOInstanceNewBaselineDatasetMapper(cfg, True)
        else:
            mapper = None

        loader = build_detection_train_loader(cfg, mapper=mapper)
        if (cfg.MODEL.META_ARCHITECTURE == "MaskFormerPartContrastive") and (len(cfg.DATASETS.CONTRASTIVE_TRAIN) > 0):
            loader = concat_loaders(loader, build_contrastive_train_loader(cfg, mapper=mapper))
        return loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if "rgba" in dataset_name:
            mapper = RGBADatasetMapper(cfg, False)
        else:
            mapper = None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_part_segmentation_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if args.machine_rank == -1:
        args.machine_rank = int(os.environ["SLURM_NODEID"])
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
