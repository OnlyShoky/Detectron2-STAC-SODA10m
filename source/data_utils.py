from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np
import os
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
import random
import json

def get_soda10m_val_dicts(img_dir, json_file, num_samples=1000):
    with open(json_file) as f:
        imgs_anns = json.load(f)

    # Mapping from original dataset IDs to contiguous IDs
    id_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns['images']):
        record = {}
        filename = os.path.join(img_dir, v["file_name"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = imgs_anns['annotations']
        objs = []
        for anno in annos:
            if anno['image_id'] == idx:
                # Map category_id to contiguous ID
                mapped_category_id = id_mapping[anno["category_id"]]
                obj = {
                    "bbox": anno["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": mapped_category_id,
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return random.sample(dataset_dicts, num_samples)
    

def create_val_subset(original_json, subset_json, num_samples=1000):
    with open(original_json) as f:
        data = json.load(f)

    # Randomly sample `num_samples` images
    sampled_images = random.sample(data['images'], num_samples)

    # Find all annotations that correspond to the sampled images
    image_ids = {img['id'] for img in sampled_images}
    sampled_annotations = [anno for anno in data['annotations'] if anno['image_id'] in image_ids]

    # Create new dataset in COCO format
    subset_data = {
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": data['categories']
    }

    with open(subset_json, 'w') as f:
        json.dump(subset_data, f)




class LimitedDataLoader:
    def __init__(self, data_loader, limit):
        self.data_loader = data_loader
        self.limit = limit

    def __iter__(self):
        count = 0
        for data in self.data_loader:
            if count >= self.limit:
                break
            yield data
            count += 1

    def __len__(self):
        return min(len(self.data_loader), self.limit)

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, limit=1000):
        self._model = model
        self._period = eval_period
        self._data_loader = LimitedDataLoader(data_loader, limit)

    def _do_loss_eval(self):
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start

            # Print every 100 iterations
            if idx % 100 == 0 and (idx >= num_warmup * 2 or seconds_per_img > 5):
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                print(
                    f"Loss on Validation done {idx + 1}/{total}. "
                    f"{seconds_per_img:.4f} s / img. ETA={str(eta)}"
                )

            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)

        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses


    def _get_loss(self, data):
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
