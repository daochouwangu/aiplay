Hugging Face's logo
Hugging Face
Search models, datasets, users...
Models
Datasets
Spaces
Posts
Docs
Solutions
Pricing

Log In
Sign Up

llava-hf
/
llava-onevision-qwen2-0.5b-ov-hf

like
10
llava-hf  
/  
llava-onevision-qwen2-0.5b-ov-hf  
喜欢  
10
Image-Text-to-Text
ONNX
Safetensors

lmms-lab/LLaVA-OneVision-Data
English
Chinese
llava_onevision
vision
conversational

arxiv:
2408.03326

License:
apache-2.0
Model card
Files and versions
Community
4
LLaVA-Onevision Model Card
image/png

Check out also the Google Colab demo to run Llava on a free-tier Google Colab instance:Open In Colab
还可以查看 Google Colab 演示，以在免费层 Google Colab 实例上运行 Llava：

Below is the model card of 0.5B LLaVA-Onevision model which is copied from the original LLaVA-Onevision model card that you can find here.
以下是 0.5B LLaVA-Onevision 模型的模型卡，复制自您可以在此处找到的原始 LLaVA-Onevision 模型卡。

Model details
Model type: LLaVA-Onevision is an open-source multimodal LLM trained by fine-tuning Qwen2 on GPT-generated multimodal instruction-following data. LLaVA-OneVision is the first single model that can simultaneously push the performance boundaries of open LMMs in three important computer vision scenarios: single-image, multi-image, and video scenarios. Importantly, the design of LLaVA-OneVision allows strong transfer learning across different modalities/scenarios, yielding new emerging capabilities. In particular, strong video understanding and cross-scenario capabilities are demonstrated through task transfer from images to videos.
模型类型：LLaVA-Onevision 是一个开源的多模态 LLM，通过对 Qwen2 进行微调，使用 GPT 生成的多模态指令跟随数据进行训练。LLaVA-OneVision 是第一个能够同时在三个重要计算机视觉场景（单图像、多图像和视频场景）中推动开放 LMM 性能边界的单一模型。重要的是，LLaVA-OneVision 的设计允许在不同模态/场景之间进行强大的迁移学习，从而产生新的新兴能力。特别是，通过从图像到视频的任务迁移，展示了强大的视频理解和跨场景能力。

Model date: LLaVA-Onevision-0.5-ov was added in August 2024.
模型日期：LLaVA-Onevision-0.5-ov 于 2024 年 8 月添加。

Paper or resources for more information: https://llava-vl.github.io/
获取更多信息的论文或资源： https://llava-vl.github.io/

Architecture: SO400M + Qwen2
架构：SO400M + Qwen2
Pretraining Stage: LCS-558K, 1 epoch, projector
预训练阶段：LCS-558K，1 个周期，投影仪
Mid Stage: A mixture of 4.7M high-quality synthetic data, 1 epoch, full model
中期：4.7M 高质量合成数据的混合，1 个周期，完整模型
Final-Image Stage: A mixture of 3.6M single-image data, 1 epoch, full model
最终图像阶段：3.6M 单图像数据的混合，1 个周期，完整模型
OneVision Stage: A mixture of 1.6M single-image/multi-image/video data, 1 epoch, full model
OneVision Stage：1.6M 单图像/多图像/视频数据的混合，1 个周期，完整模型
Precision: bfloat16
How to use the model
First, make sure to have transformers installed from branch or transformers >= 4.45.0. The model supports multi-image and multi-prompt generation. Meaning that you can pass multiple images in your prompt. Make sure also to follow the correct prompt template by applyong chat template:
首先，请确保安装了来自分支的 transformers 或 transformers 版本大于等于 4.45.0。该模型支持多图像和多提示生成。这意味着您可以在提示中传递多个图像。还请确保通过应用聊天模板遵循正确的提示模板：

Using pipeline:
Below we used "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" checkpoint.
下面我们使用了“llava-hf/llava-onevision-qwen2-0.5b-ov-hf”检查点。

from transformers import pipeline
from PIL import Image  
import requests
from transformers import AutoProcessor

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline("image-to-text", model=model_id)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt

# Each value in "content" has to be a list of dicts with types ("text", "image")

conversation = [
{

      "role": "user",
      "content": [
          {"type": "text", "text": "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"},
          {"type": "image"},
        ],
    },

]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
{"generated_text": "user\n\nWhat does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud\nassistant\nLava"}

Using pure transformers:
Below is an example script to run generation in float16 precision on a GPU device:
以下是一个在 GPU 设备上以 float16 精度运行生成的示例脚本：

import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
model_id,
torch_dtype=torch.float16,
low_cpu_mem_usage=True,
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt

# Each value in "content" has to be a list of dicts with types ("text", "image")

conversation = [
{

      "role": "user",
      "content": [
          {"type": "text", "text": "What are these?"},
          {"type": "image"},
        ],
    },

]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(\*\*inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

Model optimization
4-bit quantization through bitsandbytes library
First make sure to install bitsandbytes, pip install bitsandbytes and make sure to have access to a CUDA compatible GPU device. Simply change the snippet above with:
首先确保安装 bitsandbytes，使用 pip install bitsandbytes，并确保可以访问兼容 CUDA 的 GPU 设备。只需将上面的代码片段更改为：

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
model_id,
torch_dtype=torch.float16,
low_cpu_mem_usage=True,

- load_in_4bit=True
  )

Use Flash-Attention 2 to further speed-up generation
使用 Flash-Attention 2 进一步加速生成
First make sure to install flash-attn. Refer to the original repository of Flash Attention regarding that package installation. Simply change the snippet above with:
首先确保安装 flash-attn。有关该软件包安装的详细信息，请参考 Flash Attention 的原始仓库。只需将上面的代码片段更改为：

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
model_id,
torch_dtype=torch.float16,
low_cpu_mem_usage=True,

- use_flash_attention_2=True
  ).to(0)

Citation
@misc{li2024llavaonevisioneasyvisualtask,
title={LLaVA-OneVision: Easy Visual Task Transfer},
author={Bo Li and Yuanhan Zhang and Dong Guo and Renrui Zhang and Feng Li and Hao Zhang and Kaichen Zhang and Yanwei Li and Ziwei Liu and Chunyuan Li},
year={2024},
eprint={2408.03326},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2408.03326},
}

Downloads last month
111,703
Safetensors
Model size
894M params
Tensor type
FP16

Inference Examples
Image-Text-to-Text
Inference API (serverless) has been turned off for this model.
该模型的推理 API（无服务器）已被关闭。
Dataset used to train
llava-hf/llava-onevision-qwen2-0.5b-ov-hf
用于训练的 数据集  
llava-hf/llava-onevision-qwen2-0.5b-ov-hf
lmms-lab/LLaVA-OneVision-Data
Viewer
•
Updated Sep 10
•
3.72M
•
31k
•
127
Spaces using
llava-hf/llava-onevision-qwen2-0.5b-ov-hf
2
使用
llava-hf/llava-onevision-qwen2-0.5b-ov-hf
2
💻
RaushanTurganbay/llava-onevision
💻
kavaliha/llava-onevision
Collection including
llava-hf/llava-onevision-qwen2-0.5b-ov-hf
集合包括  
llava-hf/llava-onevision-qwen2-0.5b-ov-hf
LLaVA-Onevision
Collection
LLaVa_Onevision models for single-image, multi-image, and video scenarios
LLaVa_Onevision 模型用于单图像、多图像和视频场景
•
9 items
•
Updated about 1 month ago
大约一个月前更新
•
11
© Hugging Face
TOS
Privacy
About
Jobs
Models
Datasets
Spaces
Pricing
Docs
