# YODT
YOLO + Cross check vision transformer

## Project Breifing
As DETR requires great computational power which not suits for edge deployment, traditional CNN based solutions like YOLO also struggled at scenario with occlusion. Through testing, the [CounTR](https://github.com/Verg-Avesta/CounTR) solution proposed by Liu has been found with the balance of both accuracy for object detection with occlusion and easy to run. However, it requires pre-input of exeplar bounded boxes for the later cross check process. To automate this process, YOLO was being deployed in the initial stage to output three examplar coordinates.

## Quick Start
- Dependencies
```
pip install torch torchvision torchaudio timm scipy numpy gdown
```
- Download the model check point
```
cd Model
gdown 1CzYyiYqLshMdqJ9ZPFJyIzXBa7uFUIYZ
cd YOLO
wget https://github.com/AbelKidaneHaile/Reports/raw/refs/heads/main/models/best_re_final.pt
```
- Inference
```bash
python demo.py

------Sample Output---------
MPS device found: mps:0
Model Check Point Loaded.
Enter the path to the image file: sample_1.jpeg
YOLO detected 19 people in total.
Auto detection succeeded. Using detected person bounding boxes as exemplars.

Predicted count: 41.3

--- Performance Metrics ---
Model loading time      : 0.754 s
VRAM used by model      : 380.3 MB
CounTR inference time   : 0.168 s
Total pipeline time     : 0.774 s
---------------------------
```

## Acknowledgement
The main supporting scripts of the current solutions are from two following projects:
- [CounTR](https://github.com/Verg-Avesta/CounTR)
- [YOLO](https://docs.ultralytics.com)
The yolo model was being used from [AbelKidaneHaile](https://github.com/AbelKidaneHaile/Reports) to detect human heads appeared in the image.