import os
import gradio as gr
import torch
from utils.general import non_max_suppression, scale_coords
from models.experimental import attempt_load
import numpy as np
import cv2
from PIL import Image, ImageDraw
from utils.datasets import letterbox  # 引入 letterbox 函数

image_size = 640  # 输入图片尺寸
conf_thres = 0.01  # 置信度阈值
iou_thres = 0.6  # IoU阈值
weights = 'runs/train/bt-uncategorized/weights/best.pt'  # 替换为你的权重文件路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights, map_location=device)  # 加载FP32模型
model.eval()  # 设置为评估模式


def preprocess(image):
    # 调整图片大小，保持长宽比

    # resize image to img_size
    # h0, w0 = image.shape[:2]  # orig hw
    # r = img_size / max(h0, w0)
    # if r != 1:
    #     interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    #     image = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)

    image, _, _ = letterbox(image, image_size, auto=False)  # 调整大小，保持长宽比

    # 转换通道顺序：BGR -> RGB
    image = image[:, :, ::-1]

    # 转换为 float32 并归一化到 [0, 1]
    image = image.astype(np.float32) / 255.0

    # 转换为 PyTorch 张量格式：HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # 添加 batch 维度
    image = np.expand_dims(image, axis=0)

    return torch.from_numpy(image)


def detect_objects(image):
    with torch.no_grad():
        pred = model(image)[0]  # 模型推理
        print("Raw predictions:", pred)

    pred = non_max_suppression(pred, conf_thres, iou_thres)[0]  # 非极大值抑制
    print("NMS predictions:", pred)

    return pred


def draw_boxes(image, detections):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    if detections is not None:
        detections[:, :4] = scale_coords((image_size, image_size), detections[:, :4], image.size).round()
        for *xyxy, conf, cls in detections.tolist():
            draw.rectangle(xyxy, outline='red', width=3)
            draw.text((xyxy[0], xyxy[1]), f'{cls:.0f} {conf:.2f}', fill='red')
    return image


# Gradio 接口的处理函数
def process_image(image):
    image_tensor = preprocess(image)
    image_tensor = image_tensor.to(device, non_blocking=True)
    detections = detect_objects(image_tensor)
    result_image = draw_boxes(image, detections)
    return result_image


# 创建 Gradio 接口
interface = gr.Interface(
    fn=process_image,  # 处理函数
    inputs=gr.Image(type="numpy"),  # 输入为图片
    outputs=gr.Image(type="numpy"),  # 输出为图片
    title="脑部图像病变检测平台",
    description="上传脑部图像，模型将检测其中的病变并返回结果图片。"
)

# 添加本地非GUI测试功能
def local_test(image_path, output_dir="test_out"):
    """
    从指定路径读取图片，进行目标检测，并显示处理后的图片，同时保存到指定目录。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取图片
    image = cv2.imread(image_path)
    assert image is not None, 'Image Not Found ' + image_path

    result_image = process_image(image)

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    result_image.save(output_path)
    print(f"结果图片已保存到: {output_path}")


# 启动 Gradio 应用
if __name__ == "__main__":
    interface.launch()

    # 本地测试
    # test_image_path = "dataset-brain-tumor/uncategorized/valdata/3.jpg"
    # local_test(test_image_path)
