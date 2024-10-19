Skip to content
Navigation Menu
ultralytics
/
ultralytics

Type / to search
Code
Issues
1k
Pull requests
177
Discussions
Actions
Projects
1
Wiki
Security
Insights
Owner avatar
ultralytics
Public
ultralytics/ultralytics
Go to file
t
Add file
Folders and files
Name
Latest commit
glenn-jocher
glenn-jocher
ultralytics 8.3.18 fix is_jupyter() to globals() (#17023)
92cc8b8
·
8 hours ago
History
.github
ultralytics 8.3.16 PyTorch 2.5.0 support (#16998)
20 hours ago
docker
Update Dockerfile-runner v2.320.0 (#16912)
5 days ago
docs
ultralytics 8.3.16 PyTorch 2.5.0 support (#16998)
20 hours ago
examples
Optimize Example YOLO post-processing speed (#16821)
3 days ago
tests
ultralytics 8.3.16 PyTorch 2.5.0 support (#16998)
20 hours ago
ultralytics
ultralytics 8.3.18 fix is_jupyter() to globals() (#17023)
8 hours ago
.gitignore
Fix gitignore to format Docs datasets (#16071)
last month
CITATION.cff
ultralytics 8.3.0 YOLO11 Models Release (#16539)
3 weeks ago
CONTRIBUTING.md
Docs improvements and redirect fixes (#16287)
last month
LICENSE
Update LICENSE to AGPL-3.0 (#2031)
last year
README.md
Update README links (#16996)
2 days ago
README.zh-CN.md
Update README links (#16996)
2 days ago
mkdocs.yml
ultralytics 8.3.16 PyTorch 2.5.0 support (#16998)
20 hours ago
pyproject.toml
ultralytics 8.3.16 PyTorch 2.5.0 support (#16998)
20 hours ago
Repository files navigation
README
Code of conduct
AGPL-3.0 license
Security
YOLO Vision banner

中文 | 한국어 | 日本語 | Русский | Deutsch | Français | Español | Português | Türkçe | Tiếng Việt | العربية

Ultralytics CI Ultralytics YOLO Citation Ultralytics Docker Pulls Ultralytics Discord Ultralytics Forums Ultralytics Reddit
Run Ultralytics on Gradient Open Ultralytics In Colab Open Ultralytics In Kaggle

Ultralytics YOLO11 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLO11 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

We hope that the resources here will help you get the most out of YOLO. Please browse the Ultralytics Docs for details, raise an issue on GitHub for support, questions, or discussions, become a member of the Ultralytics Discord, Reddit and Forums!

To request an Enterprise License please complete the form at Ultralytics Licensing.

YOLO11 performance plots

Ultralytics GitHub space Ultralytics LinkedIn space Ultralytics Twitter space Ultralytics YouTube space Ultralytics TikTok space Ultralytics BiliBili space Ultralytics Discord
Documentation
See below for a quickstart install and usage examples, and see our Docs for full documentation on training, validation, prediction and deployment.

Install
Pip install the ultralytics package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.

PyPI - Version Downloads PyPI - Python Version

pip install ultralytics
For alternative installation methods including Conda, Docker, and Git, please refer to the Quickstart Guide.

Conda Version Docker Image Version

Usage
CLI
YOLO may be used directly in the Command Line Interface (CLI) with a yolo command:

yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
yolo can be used for a variety of tasks and modes and accepts additional arguments, i.e. imgsz=640. See the YOLO CLI Docs for examples.

Python
YOLO may also be used directly in a Python environment, and accepts the same arguments as in the CLI example above:

from ultralytics import YOLO

# Load a model

model = YOLO("yolo11n.pt")

# Train the model

train_results = model.train(
data="coco8.yaml", # path to dataset YAML
epochs=100, # number of training epochs
imgsz=640, # training image size
device="cpu", # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set

metrics = model.val()

# Perform object detection on an image

results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format

path = model.export(format="onnx") # return path to exported model
See YOLO Python Docs for more examples.

Models
YOLO11 Detect, Segment and Pose models pretrained on the COCO dataset are available here, as well as YOLO11 Classify models pretrained on the ImageNet dataset. Track mode is available for all Detect, Segment and Pose models.

Ultralytics YOLO supported tasks

All Models download automatically from the latest Ultralytics release on first use.

Detection (COCO)
See Detection Docs for usage examples with these models trained on COCO, which include 80 pre-trained classes.

Model size
(pixels) mAPval
50-95 Speed
CPU ONNX
(ms) Speed
T4 TensorRT10
(ms) params
(M) FLOPs
(B)
YOLO11n 640 39.5 56.1 ± 0.8 1.5 ± 0.0 2.6 6.5
YOLO11s 640 47.0 90.0 ± 1.2 2.5 ± 0.0 9.4 21.5
YOLO11m 640 51.5 183.2 ± 2.0 4.7 ± 0.1 20.1 68.0
YOLO11l 640 53.4 238.6 ± 1.4 6.2 ± 0.1 25.3 86.9
YOLO11x 640 54.7 462.8 ± 6.7 11.3 ± 0.2 56.9 194.9
mAPval values are for single-model single-scale on COCO val2017 dataset.
Reproduce by yolo val detect data=coco.yaml device=0
Speed averaged over COCO val images using an Amazon EC2 P4d instance.
Reproduce by yolo val detect data=coco.yaml batch=1 device=0|cpu
Segmentation (COCO)
Classification (ImageNet)
Pose (COCO)
OBB (DOTAv1)
Integrations
Our key integrations with leading AI platforms extend the functionality of Ultralytics' offerings, enhancing tasks like dataset labeling, training, visualization, and model management. Discover how Ultralytics, in collaboration with W&B, Comet, Roboflow and OpenVINO, can optimize your AI workflow.

Ultralytics active learning integrations

Ultralytics HUB logo space ClearML logo space Comet ML logo space NeuralMagic logo
Ultralytics HUB 🚀 W&B Comet ⭐ NEW Neural Magic
Streamline YOLO workflows: Label, train, and deploy effortlessly with Ultralytics HUB. Try now! Track experiments, hyperparameters, and results with Weights & Biases Free forever, Comet lets you save YOLO11 models, resume training, and interactively visualize and debug predictions Run YOLO11 inference up to 6x faster with Neural Magic DeepSparse
Ultralytics HUB
Experience seamless AI with Ultralytics HUB ⭐, the all-in-one solution for data visualization, YOLO11 🚀 model training and deployment, without any coding. Transform images into actionable insights and bring your AI visions to life with ease using our cutting-edge platform and user-friendly Ultralytics App. Start your journey for Free now!

Ultralytics HUB preview image
Contribute
We love your input! Ultralytics YOLO would not be possible without help from our community. Please see our Contributing Guide to get started, and fill out our Survey to send us feedback on your experience. Thank you 🙏 to all our contributors!

Ultralytics open-source contributors
License
Ultralytics offers two licensing options to accommodate diverse use cases:

AGPL-3.0 License: This OSI-approved open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the LICENSE file for more details.
Enterprise License: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your scenario involves embedding our solutions into a commercial offering, reach out through Ultralytics Licensing.
Contact
For Ultralytics bug reports and feature requests please visit GitHub Issues. Become a member of the Ultralytics Discord, Reddit, or Forums for asking questions, sharing projects, learning discussions, or for help with all things Ultralytics!

Ultralytics GitHub space Ultralytics LinkedIn space Ultralytics Twitter space Ultralytics YouTube space Ultralytics TikTok space Ultralytics BiliBili space Ultralytics Discord
About
Ultralytics YOLO11 🚀

docs.ultralytics.com
Topics
tracking machine-learning deep-learning hub pytorch yolo image-classification object-detection obb pose instance-segmentation ultralytics yolov8 yolo-world yolov9 yolo-world-v2 yolov10 yolo11
Resources
Readme
License
AGPL-3.0 license
Code of conduct
Code of conduct
Security policy
Security policy
Citation
Activity
Custom properties
Stars
31.1k stars
Watchers
176 watching
Forks
6k forks
Report repository
Releases 95
v8.3.17 - `ultralytics 8.3.17` accept spaces in CLI args (#16641)
Latest
13 hours ago

- 94 releases
  Sponsor this project
  @glenn-jocher
  glenn-jocher Glenn Jocher
  patreon
  patreon.com/ultralytics
  open_collective
  opencollective.com/ultralytics
  Learn more about GitHub Sponsors
  Used by 26.6k
  @Afnanksalal
  @kell18
  @gabrieladvent
  @joanfmendo
  @Shay-Ostrovsky
  @fzi-forschungszentrum-informatik
  @Aaryan140
  @info-wind
- 26,556
  Contributors
  354
  @glenn-jocher
  @pre-commit-ci[bot]
  @UltralyticsAssistant
  @Laughing-q
  @RizwanMunawar
  @AyushExel
  @Burhan-Q
  @abirami-vina
  @lakshanthad
  @ambitious-octopus
  @Kayzwer
  @jk4e
  @dependabot[bot]
  @developer0hye
- 340 contributors
  Deployments
  29
  github-pages 10 months ago
- 28 deployments
  Languages
  Python
  99.6%

Other
0.4%
Footer
© 2024 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact
Manage cookies
Do not share my personal information
