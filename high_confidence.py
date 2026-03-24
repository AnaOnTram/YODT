import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import timm

assert timm.__version__ >= "0.4.5"  # version check

from ultralytics import YOLO
from util.misc import make_grid
import models_mae_cross

device = torch.device('mps')

"""
python high_confidence.py
"""


def get_memory_mb():
    """Return currently allocated accelerator/process memory in MB."""
    if device.type == 'mps':
        return torch.mps.current_allocated_memory() / 1024 ** 2
    elif device.type == 'cuda':
        return torch.cuda.memory_allocated() / 1024 ** 2
    else:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


class measure_time(object):
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, typ, value, traceback):
        self.duration = (time.perf_counter_ns() - self.start) / 1e9


def detect_exemplar_bboxes(im_path, W, H):
    """Use YOLO11n to detect at least 3 people and return their bounding boxes.
    Returns a list of [[x1,y1],[x2,y2]] in original image coordinates,
    or None if fewer than 3 people are detected.
    Uses the top 3 highest-confidence detections as exemplars."""
    yolo = YOLO('Model/YOLO/yolo11n.pt')
    results = yolo(im_path, verbose=False)
    person_boxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # class 0 = person in COCO
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                person_boxes.append([[x1, y1], [x2, y2]])
    if len(person_boxes) >= 3:
        return person_boxes[:3]
    return None


def load_image(im_path):
    image = Image.open(im_path)
    image.load()
    W, H = image.size

    # Resize so height = 384 and width is a multiple of 16.
    # For portrait images this produces new_W < 384, which means the horizontal
    # sliding window never fires (it requires width >= 384). Fix: clamp new_W
    # to a minimum of 384 and re-resize, which applies a slight horizontal
    # stretch but keeps all image content intact (no black padding).
    new_H = 384
    new_W = 16 * int((W / H * 384) / 16)
    if new_W < 384:
        new_W = 384
    scale_factor_H = float(new_H) / H
    scale_factor_W = float(new_W) / W
    image = transforms.Resize((new_H, new_W))(image)
    Normalize = transforms.Compose([transforms.ToTensor()])
    image = Normalize(image)

    # Auto-detect exemplar bounding boxes using YOLO11n
    bboxes = detect_exemplar_bboxes(im_path, W, H)
    if bboxes is not None:
        print("Auto detection succeeded. Using detected person bounding boxes as exemplars.")
    else:
        print("Auto detection failed, please manually input the coordinate.")
        bboxes = []
        for i in range(3):
            print(f"Enter bounding box {i + 1} (x1 y1 x2 y2, space-separated):")
            coords = input().strip().split()
            x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
            bboxes.append([[x1, y1], [x2, y2]])

    boxes = list()
    rects = list()
    for bbox in bboxes:
        x1 = int(bbox[0][0] * scale_factor_W)
        y1 = int(bbox[0][1] * scale_factor_H)
        x2 = int(bbox[1][0] * scale_factor_W)
        y2 = int(bbox[1][1] * scale_factor_H)
        rects.append([y1, x1, y2, x2])
        bbox = image[:, y1:y2 + 1, x1:x2 + 1]
        bbox = transforms.Resize((64, 64))(bbox)
        boxes.append(bbox.numpy())

    boxes = np.array(boxes)
    boxes = torch.Tensor(boxes)

    return image, boxes, rects


def run_one_image(samples, boxes, pos, model):
    _, _, h, w = samples.shape

    s_cnt = 0
    for rect in pos:
        if rect[2] - rect[0] < 10 and rect[3] - rect[1] < 10:
            s_cnt += 1
    if s_cnt >= 1:
        r_densities = []
        r_images = []
        r_images.append(TF.crop(samples[0], 0, 0, int(h / 3), int(w / 3)))  # 1
        r_images.append(TF.crop(samples[0], 0, int(w / 3), int(h / 3), int(w / 3)))  # 3
        r_images.append(TF.crop(samples[0], 0, int(w * 2 / 3), int(h / 3), int(w / 3)))  # 7
        r_images.append(TF.crop(samples[0], int(h / 3), 0, int(h / 3), int(w / 3)))  # 2
        r_images.append(TF.crop(samples[0], int(h / 3), int(w / 3), int(h / 3), int(w / 3)))  # 4
        r_images.append(TF.crop(samples[0], int(h / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))  # 8
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), 0, int(h / 3), int(w / 3)))  # 5
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w / 3), int(h / 3), int(w / 3)))  # 6
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))  # 9

        pred_cnt = 0
        with measure_time() as et:
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                density_map = torch.zeros([h, w])
                density_map = density_map.to(device, non_blocking=True)
                start = 0
                prev = -1
                with torch.no_grad():
                    while start + 383 < w:
                        output, = model(r_image[:, :, :, start:start + 384], boxes, 3)
                        output = output.squeeze(0)
                        b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                        d1 = b1(output[:, 0:prev - start + 1])
                        b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                        d2 = b2(output[:, prev - start + 1:384])

                        b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                        density_map_l = b3(density_map[:, 0:start])
                        density_map_m = b1(density_map[:, start:prev + 1])
                        b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                        density_map_r = b4(density_map[:, prev + 1:w])

                        density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                        prev = start + 383
                        start = start + 128
                        if start + 383 >= w:
                            if start == w - 384 + 128:
                                break
                            else:
                                start = w - 384

                pred_cnt += torch.sum(density_map / 60).item()
                r_densities += [density_map]
    else:
        density_map = torch.zeros([h, w])
        density_map = density_map.to(device, non_blocking=True)
        start = 0
        prev = -1
        with measure_time() as et:
            with torch.no_grad():
                while start + 383 < w:
                    output, = model(samples[:, :, :, start:start + 384], boxes, 3)
                    output = output.squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                    d1 = b1(output[:, 0:prev - start + 1])
                    b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                    d2 = b2(output[:, prev - start + 1:384])

                    b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                    density_map_l = b3(density_map[:, 0:start])
                    density_map_m = b1(density_map[:, start:prev + 1])
                    b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                    density_map_r = b4(density_map[:, prev + 1:w])

                    density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                    prev = start + 383
                    start = start + 128
                    if start + 383 >= w:
                        if start == w - 384 + 128:
                            break
                        else:
                            start = w - 384

            pred_cnt = torch.sum(density_map / 60).item()

    e_cnt = 0
    for rect in pos:
        e_cnt += torch.sum(density_map[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1] / 60).item()
    e_cnt = e_cnt / 3
    if e_cnt > 1.8:
        pred_cnt /= e_cnt

    # Visualize the prediction
    fig = samples[0]
    box_map = torch.zeros([fig.shape[1], fig.shape[2]])
    box_map = box_map.to(device, non_blocking=True)
    for rect in pos:
        for i in range(rect[2] - rect[0]):
            box_map[min(rect[0] + i, fig.shape[1] - 1), min(rect[1], fig.shape[2] - 1)] = 10
            box_map[min(rect[0] + i, fig.shape[1] - 1), min(rect[3], fig.shape[2] - 1)] = 10
        for i in range(rect[3] - rect[1]):
            box_map[min(rect[0], fig.shape[1] - 1), min(rect[1] + i, fig.shape[2] - 1)] = 10
            box_map[min(rect[2], fig.shape[1] - 1), min(rect[1] + i, fig.shape[2] - 1)] = 10
    box_map = box_map.unsqueeze(0).repeat(3, 1, 1)
    pred = density_map.unsqueeze(0).repeat(3, 1, 1) if s_cnt < 1 \
        else make_grid(r_densities, h, w).unsqueeze(0).repeat(3, 1, 1)
    pred_vis = pred / (pred.max() + 1e-8)
    fig = fig + box_map + pred_vis * 0.6
    fig = torch.clamp(fig, 0, 1)
    torchvision.utils.save_image(fig, f'./Image/Visualisation.png')
    density_norm = pred / (pred.max() + 1e-8)
    torchvision.utils.save_image(density_norm, f'./Image/DensityMap.png')
    import matplotlib.pyplot as plt
    density_np = pred[0].detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(density_np, cmap='jet')
    plt.colorbar()
    plt.title(f'Predicted count: {pred_cnt:.1f}')
    plt.savefig('./Image/DensityHeatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    # GT map needs coordinates for all GT dots, which is hard to input and is not a must for the demo. You can provide it yourself.
    return pred_cnt, et


# Prepare model
mem_before_load = get_memory_mb()
t_load_start = time.perf_counter()

model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
model.to(device)
model_without_ddp = model

checkpoint = torch.load('Model/FSC147.pth', map_location='cpu', weights_only=False)
model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
print('Model Check Point Loaded.')

model.eval()

t_load_end = time.perf_counter()
mem_after_load = get_memory_mb()
model_load_time = t_load_end - t_load_start
model_mem_mb = mem_after_load - mem_before_load

# Prompt user for the image file to process
im_path = input("Enter the path to the image file: ").strip()

# Test on the new image
t_infer_start = time.perf_counter()

samples, boxes, pos = load_image(im_path)
samples = samples.unsqueeze(0).to(device, non_blocking=True)
boxes = boxes.unsqueeze(0).to(device, non_blocking=True)

result, elapsed_time = run_one_image(samples, boxes, pos, model)

t_infer_end = time.perf_counter()

print(f"\nPredicted count: {result:.1f}")
print("\n--- Performance Metrics ---")
print(f"Model loading time      : {model_load_time:.3f} s")
print(f"VRAM used by model      : {model_mem_mb:.1f} MB")
print(f"CounTR inference time   : {elapsed_time.duration:.3f} s")
print(f"Total pipeline time     : {t_infer_end - t_infer_start:.3f} s")
print("---------------------------")
