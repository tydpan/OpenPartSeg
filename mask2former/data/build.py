import logging
import torch.utils.data as torchdata

from detectron2.config import configurable
from detectron2.data import DatasetMapper, build_detection_train_loader, get_detection_dataset_dicts
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler, RandomSubsetTrainingSampler
from detectron2.utils.logger import _log_api_usage


__all__ = [
    "build_contrastive_train_loader",
    "concat_loaders",
]


def _contrastive_train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=False,
            min_keypoints=0,
            proposal_files=None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        if isinstance(dataset, torchdata.IterableDataset):
            logger.info("Not using any sampler since the dataset is IterableDataset.")
            sampler = None
        else:
            logger.info("Using training sampler {}".format(sampler_name))
            if sampler_name == "TrainingSampler":
                sampler = TrainingSampler(len(dataset))
            elif sampler_name == "RepeatFactorTrainingSampler":
                repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset, cfg.DATALOADER.REPEAT_THRESHOLD
                )
                sampler = RepeatFactorTrainingSampler(repeat_factors)
            elif sampler_name == "RandomSubsetTrainingSampler":
                sampler = RandomSubsetTrainingSampler(
                    len(dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO
                )
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.CONTRASTIVE_IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }

@configurable(from_config=_contrastive_train_loader_from_config)
def build_contrastive_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
):
    return build_detection_train_loader(
        dataset, 
        mapper=mapper, 
        sampler=sampler, 
        total_batch_size=total_batch_size, 
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers, 
        collate_fn=collate_fn,
    )

def concat_loaders(loader1, loader2):
    for x1, x2, in zip(loader1, loader2):
        yield (x1, x2)