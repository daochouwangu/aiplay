# 在文件开头添加编码声明
# -*- coding: utf-8 -*-

import sys
import site
import io
import logging
import cv2
import numpy as np
import pyautogui
import torch
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import time
import traceback
import os
import json
from ultralytics import YOLO
import base64
import platform

# 设置日志级别和编码
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# 手动添加 ultralytics 的路径
ultralytics_path = "/Users/liyibing/projects/GitHub/aiplay/.venv/lib/python3.10/site-packages"
if ultralytics_path not in sys.path:
    sys.path.append(ultralytics_path)
    logging.info(f"Added {ultralytics_path} to sys.path")

# 尝试导入 YOLO
try:
    from ultralytics import YOLO
    logging.info("成功导入 YOLO")
except ImportError as e:
    logging.error(f"导入 YOLO 失败: {e}")

@st.cache_resource
def load_model():
    # 暂时注释掉YOLO模型加载
    model = YOLO('yolo11n.pt')
    return model

def detect_objects(model, image):
    results = model(image)
    return results

def capture_screen_region(left, top, width, height):
    try:
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Error in capture_screen_region: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def save_item_annotation(item_name, x, y, width, height):
    annotation = {
        "item_name": item_name,
        "x": x,
        "y": y,
        "width": width,
        "height": height
    }
    
    annotations = load_annotations()
    annotations.append(annotation)
    
    with open("item_annotations.json", "w", encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)

def load_annotations():
    try:
        with open("item_annotations.json", "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def execute_instruction(instruction, model, screenshot):
    # 使用YOLO模型检测物体
    results = detect_objects(model, screenshot)
    
    # 解析指令并执行
    instructions = instruction.split('\n')
    execution_log = []

    # 打印 results
    logging.info(f"检测到的物体: {results}")
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            logging.info(f"检测到 {class_name}，置信度：{confidence:.2f}，位置：({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")

    for step in instructions:
        if "点击" in step:
            item = step.split("点击")[1].strip()
            found = False
            for r in results:
                if len(r.boxes) > 0:  # 检查是否检测到任何物体
                    if r.names[int(r.boxes.cls[0])] == item:
                        x, y = r.boxes.xyxy[0][:2].tolist()
                        execution_log.append(f"点击 {item} 坐标 ({x}, {y})")
                        found = True
                        break
            if not found:
                execution_log.append(f"未找到 {item}")
        elif "查找" in step:
            item = step.split("查找")[1].strip()
            found = False
            for r in results:
                if len(r.boxes) > 0:  # 检查是否检测到任何物体
                    if r.names[int(r.boxes.cls[0])] == item:
                        x, y, w, h = r.boxes.xyxy[0].tolist()
                        execution_log.append(f"找到 {item} 在坐标 ({x}, {y}) 大小为 {w}x{h}")
                        found = True
                        break
            if not found:
                execution_log.append(f"未找到 {item}")
    
    if not execution_log:
        execution_log.append("未检测到任何物体")
    
    return "\n".join(execution_log)

def draw_annotations(image, annotations):
    draw = ImageDraw.Draw(image)
    
    # 根据操作系统选择合适的字体
    if platform.system() == "Darwin":  # macOS
        font_path = "/System/Library/Fonts/PingFang.ttc"
    elif platform.system() == "Windows":
        font_path = "C:\\Windows\\Fonts\\msyh.ttc"
    else:  # Linux 或其他系统
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    
    try:
        font = ImageFont.truetype(font_path, 20)
    except OSError:
        st.warning(f"无法加载字体文件 {font_path}，将使用默认字体。")
        font = ImageFont.load_default()
    
    for ann in annotations:
        draw.rectangle([ann['x'], ann['y'], ann['x'] + ann['width'], ann['y'] + ann['height']], outline="red", width=2)
        draw.text((ann['x'], ann['y'] - 20), ann['item_name'], fill="red", font=font)
    return image

def main():
    st.set_page_config(page_title="智能游戏辅助系统", page_icon="🎮", layout="wide")
    st.title("智能游戏辅助系统")

    # 初始化session_state
    if 'annotations' not in st.session_state:
        st.session_state.annotations = load_annotations()

    try:
        model = load_model()
    except Exception as e:
        st.error(f"加载模型时出错：{str(e)}")
        logging.error(traceback.format_exc())
        return

    col1, col2 = st.columns(2)
    with col1:
        left = st.number_input("左边界 (像素)", value=50, min_value=0)
        top = st.number_input("上边界 (像素)", value=100, min_value=0)
    with col2:
        width = st.number_input("宽度 (像素)", value=650, min_value=1)
        height = st.number_input("高度 (像素)", value=350, min_value=1)

    if st.button("捕获屏幕"):
        screenshot = capture_screen_region(left, top, width, height)
        if screenshot is not None:
            st.session_state.screenshot = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            st.session_state.annotations = []  # 清除之前的标注
        else:
            st.error("无法捕获屏幕截图")

    if 'screenshot' in st.session_state:
        # 显示带有标注的截图
        annotated_image = draw_annotations(st.session_state.screenshot.copy(), st.session_state.annotations)
        st.image(annotated_image, caption="当前游戏画面", use_column_width=True)

        # 添加框选功能
        if st_canvas:
            st.write("在图像上动鼠标来框选物体")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#e00",
                background_image=annotated_image,
                height=annotated_image.height,
                width=annotated_image.width,
                drawing_mode="rect",
                key="canvas",
            )

            # 处理框选结果
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if len(objects) > 0:
                    last_object = objects[-1]
                    item_name = st.text_input("输入物品名")
                    if st.button("添加标注"):
                        save_item_annotation(
                            item_name,
                            int(last_object["left"]),
                            int(last_object["top"]),
                            int(last_object["width"]),
                            int(last_object["height"])
                        )
                        st.success(f"已添加物品 '{item_name}' 的标注")
                        st.rerun()
        else:
            st.warning("无法加载 streamlit_drawable_canvas。框选功能不可用。")

    # 显示现有的物品标注
    if 'annotations' in st.session_state:
        st.write("已标注的物品：", ", ".join([ann["item_name"] for ann in st.session_state.annotations]))
    else:
        st.write("还没有标注的物品。")

    # 添加指令输入和执行功能
    st.header("执行指令")
    instruction = st.text_area("输入指令序列", value="1. 点击背包\n2. 查找金币\n3. 点击使用按钮")
    
    if st.button("执行指令"):
        with st.spinner("正在执行指令..."):
            if 'screenshot' not in st.session_state:
                st.error("请先捕获屏幕截图")
            else:
                # 将图像转换为 numpy 数组并确保是 BGR 格式
                image_np = cv2.cvtColor(np.array(st.session_state.screenshot), cv2.COLOR_RGB2BGR)
                execution_result = execute_instruction(instruction, model, image_np)
                st.write("执行结果：", execution_result)

if __name__ == "__main__":
    main()
