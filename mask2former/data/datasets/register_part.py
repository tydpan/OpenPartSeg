# fmt: off
PARTIMAGENET_CATEGORIES = [
    {'id': 1, 'name': 'Quadruped Head', 'color': [220, 20, 60], 'isthing': 1},
    {'id': 2, 'name': 'Quadruped Body', 'color': [119, 11, 32], 'isthing': 1},
    {'id': 3, 'name': 'Quadruped Foot', 'color': [0, 0, 142], 'isthing': 1},
    {'id': 4, 'name': 'Quadruped Tail', 'color': [0, 0, 230], 'isthing': 1},
    {'id': 5, 'name': 'Biped Head', 'color': [106, 0, 228], 'isthing': 1},
    {'id': 6, 'name': 'Biped Body', 'color': [0, 60, 100], 'isthing': 1},
    {'id': 7, 'name': 'Biped Hand', 'color': [0, 80, 100], 'isthing': 1},
    {'id': 8, 'name': 'Biped Foot', 'color': [0, 0, 70], 'isthing': 1},
    {'id': 9, 'name': 'Biped Tail', 'color': [0, 0, 192], 'isthing': 1},
    {'id': 10, 'name': 'Fish Head', 'color': [250, 170, 30], 'isthing': 1},
    {'id': 11, 'name': 'Fish Body', 'color': [100, 170, 30], 'isthing': 1},
    {'id': 12, 'name': 'Fish Fin', 'color': [220, 220, 0], 'isthing': 1},
    {'id': 13, 'name': 'Fish Tail', 'color': [175, 116, 175], 'isthing': 1},
    {'id': 14, 'name': 'Bird Head', 'color': [250, 0, 30], 'isthing': 1},
    {'id': 15, 'name': 'Bird Body', 'color': [165, 42, 42], 'isthing': 1},
    {'id': 16, 'name': 'Bird Wing', 'color': [255, 77, 255], 'isthing': 1},
    {'id': 17, 'name': 'Bird Foot', 'color': [0, 226, 252], 'isthing': 1},
    {'id': 18, 'name': 'Bird Tail', 'color': [182, 182, 255], 'isthing': 1},
    {'id': 19, 'name': 'Snake Head', 'color': [0, 82, 0], 'isthing': 1},
    {'id': 20, 'name': 'Snake Body', 'color': [120, 166, 157], 'isthing': 1},
    {'id': 21, 'name': 'Reptile Head', 'color': [110, 76, 0], 'isthing': 1},
    {'id': 22, 'name': 'Reptile Body', 'color': [174, 57, 255], 'isthing': 1},
    {'id': 23, 'name': 'Reptile Foot', 'color': [199, 100, 0], 'isthing': 1},
    {'id': 24, 'name': 'Reptile Tail', 'color': [72, 0, 118], 'isthing': 1},
    {'id': 25, 'name': 'Car Body', 'color': [255, 179, 240], 'isthing': 1},
    {'id': 26, 'name': 'Car Tier', 'color': [0, 125, 92], 'isthing': 1},
    {'id': 27, 'name': 'Car Side Mirror', 'color': [209, 0, 151], 'isthing': 1},
    {'id': 28, 'name': 'Bicycle Body', 'color': [188, 208, 182], 'isthing': 1},
    {'id': 29, 'name': 'Bicycle Head', 'color': [0, 220, 176], 'isthing': 1},
    {'id': 30, 'name': 'Bicycle Seat', 'color': [255, 99, 164], 'isthing': 1},
    {'id': 31, 'name': 'Bicycle Tier', 'color': [92, 0, 73], 'isthing': 1},
    {'id': 32, 'name': 'Boat Body', 'color': [133, 129, 255], 'isthing': 1},
    {'id': 33, 'name': 'Boat Sail', 'color': [78, 180, 255], 'isthing': 1},
    {'id': 34, 'name': 'Aeroplane Head', 'color': [0, 228, 0], 'isthing': 1},
    {'id': 35, 'name': 'Aeroplane Body', 'color': [174, 255, 243], 'isthing': 1},
    {'id': 36, 'name': 'Aeroplane Engine', 'color': [45, 89, 255], 'isthing': 1},
    {'id': 37, 'name': 'Aeroplane Wing', 'color': [134, 134, 103], 'isthing': 1},
    {'id': 38, 'name': 'Aeroplane Tail', 'color': [145, 148, 174], 'isthing': 1},
    {'id': 39, 'name': 'Bottle Mouth', 'color': [255, 208, 186], 'isthing': 1},
    {'id': 40, 'name': 'Bottle Body', 'color': [197, 226, 255], 'isthing': 1}
]
# fmt: on

def _get_partimagenet_instances_meta():
    thing_ids = [k["id"] for k in PARTIMAGENET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in PARTIMAGENET_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in PARTIMAGENET_CATEGORIES if k["isthing"] == 1]
    ret = {
        # "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        # "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret