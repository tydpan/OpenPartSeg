# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config, add_part_segmentation_config

# dataset loading
from .data import build_contrastive_train_loader, concat_loaders
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.rgba_dataset_mapper import RGBADatasetMapper
from .data.dataset_mappers.rgba_coco_instance_new_baseline_dataset_mapper import (
    RGBACOCOInstanceNewBaselineDatasetMapper
)
from .data.dataset_mappers.rgba_alpha_aug_coco_instance_new_baseline_dataset_mapper import (
    AlphaAugCOCOInstanceNewBaselineDatasetMapper
)
from .data.dataset_mappers.rgba_alpha_aug_strong_coco_instance_new_baseline_dataset_mapper import (
    AlphaAugStrongCOCOInstanceNewBaselineDatasetMapper
)
from .data.dataset_mappers.rgba_mask_former_instance_dataset_mapper import (
    RGBAMaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.rgba_alpha_aug_mask_former_instance_dataset_mapper import (
    AlphaAugMaskFormerInstanceDatasetMapper,
)
from .data.datasets.register_part import _get_partimagenet_instances_meta

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.coco_evaluation_score_threshold import COCOEvaluatorScoreT
