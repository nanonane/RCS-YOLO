import os
import gradio as gr
import torch
from utils.general import non_max_suppression, scale_coords
from models.experimental import attempt_load
import numpy as np
import cv2
from PIL import Image, ImageDraw
from utils.datasets import letterbox  # 引入 letterbox 函数

tumor_class = ['pituitary tumor', 'meningioma', 'glioma']  # 类别名称

image_size = 640  # 模型输入图片尺寸
conf_thres = 0.01  # 置信度阈值
iou_thres = 0.6  # IoU阈值
weights = 'runs/train/bt-withno-decoupled/weights/last.pt'  # 替换为你的权重文件路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights, map_location=device)  # 加载FP32模型
model.eval()  # 设置为评估模式


def preprocess(image: np.ndarray) -> tuple:
    # 调整图片大小，保持长宽比

    # resize image to img_size
    h0, w0 = image.shape[:2]  # orig hw
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

    return torch.from_numpy(image), (h0, w0)  # 返回张量和原始图片尺寸


def detect_objects(image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        pred = model(image)[0]  # 模型推理
        print("Raw predictions:", pred)

    pred = non_max_suppression(pred, conf_thres, iou_thres)[0]  # 非极大值抑制
    print("NMS predictions:", pred)

    return pred


def annotate_image(image: np.ndarray, detections: torch.Tensor) -> tuple:
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    results_text = []
    if detections is not None:
        detections[:, :4] = scale_coords((image_size, image_size), detections[:, :4], image.size).round()
        for *xyxy, conf, cls in detections.tolist():
            draw.rectangle(xyxy, outline='red', width=3)
            results_text.append(f"类别: {tumor_class[int(cls)]};  置信度: {conf:.2f}")

    return image, "\n".join(results_text) if results_text else "未检测到目标"


def resize_img(size0: tuple, image: Image.Image) -> Image.Image:
    # resize image to original size
    h0, w0 = size0
    # 使用 PIL.Image 的 resize 方法
    image = image.resize((w0, h0), Image.ANTIALIAS)
    return image


# Gradio 接口的处理函数
def process_image(image: np.ndarray) -> tuple:
    image_tensor, size0 = preprocess(image)
    image_tensor = image_tensor.to(device, non_blocking=True)
    detections = detect_objects(image_tensor)
    result_image, results_text = annotate_image(image, detections)
    # result_image = resize_img(size0, result_image)  # 调整结果图片大小
    return result_image, results_text


# 创建 Gradio 接口
interface = gr.Interface(
    fn=process_image,  # 处理函数
    inputs=gr.Image(type="numpy"),  # 输入为图片
    outputs=[
        gr.Image(type="numpy", label="检测结果图像"),  # 输出为图片
        gr.Textbox(label="检测结果详情")  # 输出为文本
    ],
    title="脑部图像病变检测平台",
    description="上传脑部图像，模型将检测其中的病变并返回结果图片和详细信息。"
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

    result_image, results_text = process_image(image)

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    result_image.save(output_path)
    print(f"结果图片已保存到: {output_path}")
    print(f"检测结果详情:\n{results_text}")


# 启动 Gradio 应用
if __name__ == "__main__":
    interface.launch()

    # 本地测试
    # test_image_path = "dataset-brain-tumor/uncategorized/valdata/3.jpg"
    # local_test(test_image_path)
