from ultralytics import YOLO

# Load the model
model = YOLO("./number_11x.pt")

# Perform detection on an image
results = model(
    "./data/1.png",
    conf=0.50,        # 置信度阈值，降低以检测更多目标（默认 0.25）
    iou=0.7,         # IoU 阈值用于 NMS（默认 0.7）
    imgsz=780,       # 输入图像尺寸（默认 640）
    max_det=100,     # 最大检测数量（默认 300）
    verbose=False,    # 关闭详细输出
    agnostic_nms=False
)
# Display or process the results
results[0].show()  # This will display the image with detected objects


# 方法 2: 获取检测框信息
for i, result in enumerate(results):
    boxes = result.boxes  # 获取边界框
    if boxes is not None:
        print(f"图片 {i+1}: 检测到 {len(boxes)} 个目标")
        for box in boxes:
            cls = int(box.cls[0])  # 类别索引
            conf = float(box.conf[0])  # 置信度
            xyxy = box.xyxy[0].tolist()  # 边界框坐标 [x1, y1, x2, y2]
            print(f"  - 类别：{cls}, 置信度：{conf:.2f}, 位置：{xyxy}")

for result in results:
    print(result.boxes)   # Bounding boxes
    print(result.names)   # Detected classes
    print(result.scores)  # Confidence scores
