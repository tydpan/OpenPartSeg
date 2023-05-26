import copy
import argparse
from PIL import Image
import torch

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer
from mask2former import add_maskformer2_config
import mask2former.data.transforms as T


class DefaultPredictorRGBA:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdgeRGBA(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format == "RGBA", self.input_format


    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in RGBA order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class InstancesWrap():
    def __init__(self, instances, T=0.0, topk=100):
        instances = copy.deepcopy(instances)
        idx = instances.scores.argsort(descending=True)[:topk]
        self.pred_scores = instances.scores[idx]
        self.pred_masks = instances.pred_masks[idx, :, :]
        idx = self.pred_scores > T
        self.pred_scores = self.pred_scores[idx]
        self.pred_masks = self.pred_masks[idx]
        
    def has(self, name):
        return hasattr(self, name)
    
    def __len__(self):
        return len(self.pred_scores)


def get_predictor(cfg_path, model_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = model_path
    return DefaultPredictorRGBA(cfg)

def get_parser():
    parser = argparse.ArgumentParser(description="predictor for Open Part Seg")
    parser.add_argument(
        "ckpt",
        help="A path to checkpoint",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; " 
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo/outputs",
        help="A directory to save output visualizations",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Maximum number for instance predictions to be shown",
    )
    return parser


if __name__ == "__main__":
    import os
    import numpy as np
    import warnings

    args = get_parser().parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"loading model from {args.ckpt}...")
    predictor = get_predictor(
        "configs/part_segmentation/dt_alpha/clsag.yaml", 
        args.ckpt,
    )
    for input in args.input:
        im = np.asarray(Image.open(input))
        assert im.shape[2] == 4, f"{input} needs to be RGBA image"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = predictor(im)
        instances = InstancesWrap(
            outputs["instances"].to("cpu"), args.score_threshold, args.topk)
        v = Visualizer(im[:, :, :3], scale=1.2)
        out = v.draw_instance_predictions(instances)
        out = Image.fromarray(out.get_image())
        save_path = os.path.join(args.output, f"{os.path.basename(input)}")
        print(f"saved to {save_path}")
        out.save(save_path)