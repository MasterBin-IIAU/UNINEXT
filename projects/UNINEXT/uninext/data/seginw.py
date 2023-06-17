from detectron2.data.datasets.register_coco import register_coco_instances
import os

CATEGORIES = { 
    "seginw/Airplane-Parts": [{"id": 0, "name": "Airplane", "supercategory": "Airplanes"}, {"id": 1, "name": "Body", "supercategory": "Airplanes"}, {"id": 2, "name": "Cockpit", "supercategory": "Airplanes"}, {"id": 3, "name": "Engine", "supercategory": "Airplanes"}, {"id": 4, "name": "Wing", "supercategory": "Airplanes"}],
    "seginw/Bottles": [{"id": 0, "name": "bottle", "supercategory": "bottlelabelscans"}, {"id": 1, "name": "can", "supercategory": "bottlelabelscans"}, {"id": 2, "name": "label", "supercategory": "bottlelabelscans"}],
    "seginw/Brain-Tumor": [{"id": 0, "name": "tumor", "supercategory": "tumor"}],
    "seginw/Chicken": [{"id": 0, "name": "chicken", "supercategory": "Chickens"}],
    "seginw/Cows": [{"id": 0, "name": "cow", "supercategory": "cow"}],
    "seginw/Electric-Shaver": [{"id": 0, "name": "caorau", "supercategory": "caurau"}],
    "seginw/Elephants": [{"id": 0, "name": "elephant", "supercategory": "elephant"}],
    "seginw/Fruits": [{"id": 0, "name": "apple", "supercategory": "fruits"}, {"id": 1, "name": "lemon", "supercategory": "fruits"}, {"id": 2, "name": "orange", "supercategory": "fruits"}, {"id": 3, "name": "pear", "supercategory": "fruits"}, {"id": 4, "name": "strawberry", "supercategory": "fruits"}],
    "seginw/Garbage": [{"id": 0, "name": "bin", "supercategory": "garbage-bin-road-pavement"}, {"id": 1, "name": "garbage", "supercategory": "garbage-bin-road-pavement"}, {"id": 2, "name": "pavement", "supercategory": "garbage-bin-road-pavement"}, {"id": 3, "name": "road", "supercategory": "garbage-bin-road-pavement"}],
    "seginw/Ginger-Garlic": [{"id": 0, "name": "garlic", "supercategory": "object-segmentation"}, {"id": 1, "name": "ginger", "supercategory": "object-segmentation"}],
    "seginw/Hand": [{"id": 0, "name": "Hand-Segmentation", "supercategory": "Hand-Segmentation"}, {"id": 1, "name": "hand", "supercategory": "Hand-Segmentation"}],
    "seginw/Hand-Metal": [{"id": 0, "name": "hand", "supercategory": "metal"}, {"id": 1, "name": "metal", "supercategory": "metal"}],
    "seginw/House-Parts": [{"id": 0, "name": "aluminium door", "supercategory": "build"}, {"id": 1, "name": "aluminium window", "supercategory": "build"}, {"id": 2, "name": "cellar window", "supercategory": "build"}, {"id": 3, "name": "mint cond roof", "supercategory": "build"}, {"id": 4, "name": "plaster", "supercategory": "build"}, {"id": 5, "name": "plastic door", "supercategory": "build"}, {"id": 6, "name": "plastic window", "supercategory": "build"}, {"id": 7, "name": "plate fascade", "supercategory": "build"}, {"id": 8, "name": "wooden door", "supercategory": "build"}, {"id": 9, "name": "wooden fascade", "supercategory": "build"}, {"id": 10, "name": "wooden window", "supercategory": "build"}, {"id": 11, "name": "worn cond roof", "supercategory": "build"}],
    "seginw/HouseHold-Items": [{"id": 0, "name": "bottle", "supercategory": "Household-items"}, {"id": 1, "name": "mouse", "supercategory": "Household-items"}, {"id": 2, "name": "perfume", "supercategory": "Household-items"}, {"id": 3, "name": "phone", "supercategory": "Household-items"}],
    "seginw/Nutterfly-Squireel": [{"id": 0, "name": "butterfly", "supercategory": "Animal"}, {"id": 1, "name": "squirrel", "supercategory": "Animal"}],
    "seginw/Phones": [{"id": 0, "name": "phone", "supercategory": "yolov5"}],
    "seginw/Poles": [{"id": 0, "name": "poles", "supercategory": "utility-pole"}],
    "seginw/Puppies": [{"id": 0, "name": "puppy", "supercategory": "puppies"}],
    "seginw/Rail": [{"id": 0, "name": "rail", "supercategory": "Rail"}],
    "seginw/Salmon-Fillet": [{"id": 0, "name": "Salmon_fillet", "supercategory": "salmon-fillet"}],
    "seginw/Strawberry": [{"id": 0, "name": "R_strawberry", "supercategory": "strawberry"}, {"id": 1, "name": "people", "supercategory": "strawberry"}],
    "seginw/Tablets": [{"id": 0, "name": "tablets", "supercategory": "tablets-capsule"}],
    "seginw/Toolkits": [{"id": 0, "name": "Allen-key", "supercategory": "label"}, {"id": 1, "name": "block", "supercategory": "label"}, {"id": 2, "name": "gasket", "supercategory": "label"}, {"id": 3, "name": "plier", "supercategory": "label"}, {"id": 4, "name": "prism", "supercategory": "label"}, {"id": 5, "name": "screw", "supercategory": "label"}, {"id": 6, "name": "screwdriver", "supercategory": "label"}, {"id": 7, "name": "wrench", "supercategory": "label"}],
    "seginw/Trash": [{"id": 0, "name": "Aluminium foil", "supercategory": "trash-segmentation"}, {"id": 1, "name": "Cigarette", "supercategory": "trash-segmentation"}, {"id": 2, "name": "Clear plastic bottle", "supercategory": "trash-segmentation"}, {"id": 3, "name": "Corrugated carton", "supercategory": "trash-segmentation"}, {"id": 4, "name": "Disposable plastic cup", "supercategory": "trash-segmentation"}, {"id": 5, "name": "Drink Can", "supercategory": "trash-segmentation"}, {"id": 6, "name": "Egg Carton", "supercategory": "trash-segmentation"}, {"id": 7, "name": "Foam cup", "supercategory": "trash-segmentation"}, {"id": 8, "name": "Food Can", "supercategory": "trash-segmentation"}, {"id": 9, "name": "Garbage bag", "supercategory": "trash-segmentation"}, {"id": 10, "name": "Glass bottle", "supercategory": "trash-segmentation"}, {"id": 11, "name": "Glass cup", "supercategory": "trash-segmentation"}, {"id": 12, "name": "Metal bottle cap", "supercategory": "trash-segmentation"}, {"id": 13, "name": "Other carton", "supercategory": "trash-segmentation"}, {"id": 14, "name": "Other plastic bottle", "supercategory": "trash-segmentation"}, {"id": 15, "name": "Paper cup", "supercategory": "trash-segmentation"}, {"id": 16, "name": "Plastic bag - wrapper", "supercategory": "trash-segmentation"}, {"id": 17, "name": "Plastic bottle cap", "supercategory": "trash-segmentation"}, {"id": 18, "name": "Plastic lid", "supercategory": "trash-segmentation"}, {"id": 19, "name": "Plastic straw", "supercategory": "trash-segmentation"}, {"id": 20, "name": "Pop tab", "supercategory": "trash-segmentation"}, {"id": 21, "name": "Styrofoam piece", "supercategory": "trash-segmentation"}], 
    "seginw/Watermelon": [{"id": 0, "name": "watermelon", "supercategory": "melon"}]
}


def _get_builtin_metadata(categories):
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS_SEGINW = {
    "seginw/Airplane-Parts": ("seginw/Airplane-Parts/valid", "seginw/Airplane-Parts/valid/_annotations_min1cat.coco.json"),
    "seginw/Bottles": ("seginw/Bottles/valid", "seginw/Bottles/valid/_annotations_min1cat.coco.json"),
    "seginw/Brain-Tumor": ("seginw/Brain-Tumor/valid", "seginw/Brain-Tumor/valid/_annotations_min1cat.coco.json"),
    "seginw/Chicken": ("seginw/Chicken/valid", "seginw/Chicken/valid/_annotations_min1cat.coco.json"),
    "seginw/Cows": ("seginw/Cows/valid", "seginw/Cows/valid/_annotations_min1cat.coco.json"),
    "seginw/Electric-Shaver": ("seginw/Electric-Shaver/valid", "seginw/Electric-Shaver/valid/_annotations_min1cat.coco.json"),
    "seginw/Elephants": ("seginw/Elephants/valid", "seginw/Elephants/valid/_annotations_min1cat.coco.json"),
    "seginw/Fruits": ("seginw/Fruits/valid", "seginw/Fruits/valid/_annotations_min1cat.coco.json"),
    "seginw/Garbage": ("seginw/Garbage/valid", "seginw/Garbage/valid/_annotations_min1cat.coco.json"),
    "seginw/Ginger-Garlic": ("seginw/Ginger-Garlic/valid", "seginw/Ginger-Garlic/valid/_annotations_min1cat.coco.json"),
    "seginw/Hand": ("seginw/Hand/valid", "seginw/Hand/valid/_annotations_min1cat.coco.json"),
    "seginw/Hand-Metal": ("seginw/Hand-Metal/valid", "seginw/Hand-Metal/valid/_annotations_min1cat.coco.json"),
    "seginw/HouseHold-Items": ("seginw/HouseHold-Items/valid", "seginw/HouseHold-Items/valid/_annotations_min1cat.coco.json"),
    "seginw/House-Parts": ("seginw/House-Parts/valid", "seginw/House-Parts/valid/_annotations_min1cat.coco.json"),
    "seginw/Nutterfly-Squireel": ("seginw/Nutterfly-Squireel/valid", "seginw/Nutterfly-Squireel/valid/_annotations_min1cat.coco.json"),
    "seginw/Phones": ("seginw/Phones/valid", "seginw/Phones/valid/_annotations_min1cat.coco.json"),
    "seginw/Poles": ("seginw/Poles/valid", "seginw/Poles/valid/_annotations_min1cat.coco.json"),
    "seginw/Puppies": ("seginw/Puppies/valid", "seginw/Puppies/valid/_annotations_min1cat.coco.json"),
    "seginw/Rail": ("seginw/Rail/valid", "seginw/Rail/valid/_annotations_min1cat.coco.json"),
    "seginw/Salmon-Fillet": ("seginw/Salmon-Fillet/valid", "seginw/Salmon-Fillet/valid/_annotations_min1cat.coco.json"),
    "seginw/Strawberry": ("seginw/Strawberry/valid", "seginw/Strawberry/valid/_annotations_min1cat.coco.json"),
    "seginw/Tablets": ("seginw/Tablets/valid", "seginw/Tablets/valid/_annotations_min1cat.coco.json"),
    "seginw/Toolkits": ("seginw/Toolkits/valid", "seginw/Toolkits/valid/_annotations_min1cat.coco.json"),
    "seginw/Trash": ("seginw/Trash/valid", "seginw/Trash/valid/_annotations_min1cat.coco.json"),
    "seginw/Watermelon": ("seginw/Watermelon/valid", "seginw/Watermelon/valid/_annotations_min1cat.coco.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_SEGINW.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(CATEGORIES[key]),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
        dataset_name_in_dict="seginw"
    )
