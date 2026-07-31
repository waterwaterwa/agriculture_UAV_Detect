# AgriUAV YOLOv8 Agent Guide / AgriUAV YOLOv8 智能体指南

## Project Role / 项目定位

中文：本项目是一个基于 Ultralytics YOLOv8 的农业无人机场景五类目标检测工程，重点面向小目标检测。当前主模型为 `YOLOv8n-P2-ECA-SimAM`，在标准 YOLOv8n 的基础上增加 P2/4 检测分支，并在 P2 小目标分支加入 ECA 与 SimAM 注意力模块。

English: This project is a five-class agricultural UAV object detection project based on Ultralytics YOLOv8. It focuses on small-object detection. The main model is `YOLOv8n-P2-ECA-SimAM`, which adds a P2/4 detection branch to YOLOv8n and applies ECA plus SimAM attention on the P2 small-object branch.

## Repository Layout / 目录结构

中文：主要代码位于 `agriculture_UAV_Detect`，数据集默认位于同级目录 `../AgriUAV_yolo_dataset`。

English: The main code lives in `agriculture_UAV_Detect`, and the default dataset is expected at the sibling path `../AgriUAV_yolo_dataset`.

```text
agriculture_UAV_Detect/
  data/agri_uav.yaml                         # dataset config / 数据集配置
  scripts/train_agri.py                      # training wrapper / 训练入口
  scripts/train_agri_p2_eca_simam.sh         # main training script / 主训练脚本
  scripts/val_agri.py                        # YOLO-format dataset validation / 数据集验证脚本
  scripts/predict_agri.py                    # image/folder visualization inference / 图片可视化推理脚本
  ultralytics/                               # local modified Ultralytics code / 本地改造版 Ultralytics
  runs/detect/agri_p2_eca_simam/weights/     # trained weights / 已训练权重
../AgriUAV_yolo_dataset/
  images/train, labels/train
  images/test, labels/test
  data.yaml
```

## Dataset / 数据集

中文：当前数据集为 YOLO 格式，训练集 6400 张，验证/测试集 1600 张，类别数为 5。

English: The dataset uses YOLO format. It contains 6400 training images and 1600 validation/test images with 5 classes.

```text
0 person
1 pole_tower
2 agricultural_machine
3 animal
4 hay_bale
```

中文：`data/agri_uav.yaml` 中的 `path: ../AgriUAV_yolo_dataset` 是相对于项目根目录运行时的正确路径。

English: In `data/agri_uav.yaml`, `path: ../AgriUAV_yolo_dataset` is correct when commands are run from the project root.

## Model Variants / 模型变体

中文：相关模型配置位于 `ultralytics/cfg/models/v8/`。

English: Model YAML files are under `ultralytics/cfg/models/v8/`.

```text
yolov8-p2.yaml
yolov8n-p2-eca-agri.yaml
yolov8n-p2-simam-agri.yaml
yolov8n-p2-eca-simam-agri.yaml
```

中文：当前主模型使用 `yolov8n-p2-eca-simam-agri.yaml`。Detect 输入为 P2/P3/P4/P5 四个尺度，其中 P2 分支经过 ECA 和 SimAM，P3/P4/P5 不直接经过这两个注意力模块。

English: The main model uses `yolov8n-p2-eca-simam-agri.yaml`. The Detect head receives P2/P3/P4/P5 features. The P2 branch passes through ECA and SimAM, while P3/P4/P5 do not directly pass through these attention modules.

## Environment / 运行环境

中文：默认使用 conda 环境 `yolo`。该环境已验证可加载本项目本地 `ultralytics`，并可运行训练、验证和推理。

English: The default conda environment is `yolo`. It has been verified to load the local modified `ultralytics` package and run training, validation, and inference.

```bash
conda activate yolo
cd /home/qc004/task1/task1/agriculture_UAV_Detect
```

## Training / 训练

中文：主训练入口是 shell 脚本调用 `scripts/train_agri.py`。训练脚本会优先导入项目内的 `ultralytics`，以支持自定义 ECA/SimAM 模块。

English: Training is launched through shell scripts that call `scripts/train_agri.py`. The wrapper prioritizes the local `ultralytics` package so custom ECA/SimAM modules are available.

```bash
bash scripts/train_agri_p2_eca_simam.sh
```

中文：当前主训练脚本默认参数包括 `imgsz=640`、`epochs=350`、`batch=32`、`optimizer=SGD`、`amp=False`、`pretrained=False`。如显存不足，优先降低 batch 到 16 或 12。

English: The current main training script defaults to `imgsz=640`, `epochs=350`, `batch=32`, `optimizer=SGD`, `amp=False`, and `pretrained=False`. If CUDA memory is insufficient, reduce batch size to 16 or 12.

## Validation / 数据集验证

中文：对 YOLO 格式数据集计算指标时使用：

English: Use the following command to evaluate a YOLO-format dataset:

```bash
bash scripts/val_agri.sh \
  --model runs/detect/agri_p2_eca_simam/weights/best.pt \
  --data data/agri_uav.yaml \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --name agri_val_best
```

## Prediction / 可视化推理

中文：对单张图片或文件夹保存可视化检测结果时使用：

English: Use the following command to save visualized predictions for an image or image folder:

```bash
bash scripts/predict_agri.sh /path/to/images \
  --model runs/detect/agri_p2_eca_simam/weights/best.pt \
  --conf 0.25 \
  --device 0 \
  --name agri_predict
```

## Current Result Notes / 当前结果说明

中文：已训练的 `agri_p2_eca_simam` 模型整体效果较好，但 `person` 类召回偏低。其他类别 `pole_tower`、`agricultural_machine`、`animal`、`hay_bale` 指标相对理想。后续优化重点应放在补充无人机视角下的人类小目标样本，并用当前 `best.pt` 进行低学习率微调。

English: The trained `agri_p2_eca_simam` model performs well overall, but the `person` class has low recall. The other classes, `pole_tower`, `agricultural_machine`, `animal`, and `hay_bale`, are comparatively strong. Future improvement should focus on adding UAV-view small-person samples and fine-tuning from the current `best.pt` with a lower learning rate.

## Editing Rules / 修改约定

中文：修改模型结构时同步更新 YAML、测试文件和文档。修改类别顺序时必须同步更新数据集标签、`data/agri_uav.yaml`、模型验证脚本和 README。不要在没有确认的情况下删除训练权重或数据集。

English: When changing model structure, update YAML files, tests, and documentation together. When changing class order, update dataset labels, `data/agri_uav.yaml`, validation scripts, and README consistently. Do not delete trained weights or datasets without explicit confirmation.
