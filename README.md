# MNIST 手写数字识别系统

基于深度学习的手写数字识别项目，支持 CNN、KNN、XGBoost 三种算法，并提供 Web 界面进行实时识别。

## 项目简介

本项目实现了一个完整的手写数字识别系统，包括模型训练、评估和 Web 部署。使用 MNIST 数据集进行训练，支持多种机器学习算法，并提供美观的 Web 界面供用户手写数字进行实时识别。

## 功能特性

- 🤖 **多算法支持**：CNN、KNN、XGBoost 三种算法
- 🎨 **Web 界面**：现代化的手写识别界面，支持鼠标和触摸屏
- 📊 **实时识别**：即时显示识别结果和置信度
- 📈 **概率展示**：显示所有数字的预测概率
- 🚀 **高性能**：基于 PyTorch 的深度学习框架
- 📱 **响应式设计**：支持桌面端和移动端
- 💾 **模型保存**：自动保存最佳模型参数

## 项目结构

```
mnist_digit_recognition/
├── app.py              # Flask Web 应用
├── model.py            # 模型定义和训练代码
├── inference.py         # 模型推理代码
├── evaluate.py         # 模型评估代码
├── requirements.txt     # 项目依赖
├── README.md          # 项目说明文档
├── templates/
│   └── index.html    # Web 界面
├── models/           # 训练好的模型（不提交到 Git）
└── dataset/         # MNIST 数据集（不提交到 Git）
```

## 技术栈

### 后端
- **Python 3.9+**
- **PyTorch**：深度学习框架
- **Flask**：Web 框架
- **scikit-learn**：机器学习工具库
- **XGBoost**：梯度提升框架

### 前端
- **HTML5/CSS3/JavaScript**
- **Canvas API**：手写输入
- **响应式设计**：移动端适配

### 数据集
- **MNIST**：60,000 训练样本 + 10,000 测试样本
- 图像尺寸：28×28 像素
- 类别：0-9 共 10 个数字

## 安装说明

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/mnist_digit_recognition.git
cd mnist_digit_recognition
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载 MNIST 数据集

首次运行训练代码时会自动下载 MNIST 数据集到 `dataset/` 目录。

## 使用方法

### 训练模型

#### 训练 CNN 模型

```bash
python model.py
```

训练过程中会显示每个 epoch 的训练和测试准确率，并自动保存最佳模型到 `models/cnn.pth`。

#### 训练 KNN 模型

```python
# 在 model.py 中调用
train_knn()
```

#### 训练 XGBoost 模型

```python
# 在 model.py 中调用
train_xgboost()
```

### 运行 Web 应用

#### 1. 确保模型已训练

确保 `models/cnn.pth` 文件存在，如果没有，请先运行训练代码。

#### 2. 启动 Flask 服务

```bash
python app.py
```

服务将在 `http://localhost:5000` 启动。

#### 3. 打开浏览器

访问 `http://localhost:5000`，在黑色画布上用鼠标手写白色数字，点击"开始识别"按钮即可看到识别结果。

### 模型推理

```python
from model import cnn
import torch
from torchvision import transforms
from PIL import Image

# 加载模型
model = cnn()
model.load_state_dict(torch.load('models/cnn.pth'))
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

# 加载图像并预测
image = Image.open('your_image.png').convert('L')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    predicted = torch.argmax(output, dim=1).item()
    print(f'预测结果: {predicted}')
```

## 模型架构

### CNN 模型结构

```
输入层 (1×28×28)
    ↓
卷积层1 (6个 3×3 卷积核)
    ↓
批归一化1
    ↓
ReLU 激活
    ↓
最大池化1 (2×2, stride=2)
    ↓
卷积层2 (16个 3×3 卷积核)
    ↓
批归一化2
    ↓
ReLU 激活
    ↓
最大池化2 (2×2, stride=1)
    ↓
展平 (1936 维)
    ↓
全连接层1 (1936 → 120)
    ↓
ReLU 激活
    ↓
全连接层2 (120 → 84)
    ↓
ReLU 激活
    ↓
全连接层3 (84 → 32)
    ↓
Tanh 激活
    ↓
全连接层4 (32 → 10)
    ↓
输出层 (10 类)
```

### 模型参数

- 总参数量：约 250,000
- 训练轮数：20 epochs
- 批量大小：128
- 优化器：AdamW (lr=0.001, weight_decay=1e-4)
- 损失函数：交叉熵损失
- 学习率调度：固定学习率

## 性能指标

### CNN 模型

| 指标 | 数值 |
|------|------|
| 训练准确率 | ~99.5% |
| 测试准确率 | ~99.0% |
| 推理时间 | <10ms (CPU) |
| 模型大小 | ~1MB |

### KNN 模型

| 指标 | 数值 |
|------|------|
| 测试准确率 | ~97.0% |
| 推理时间 | ~100ms (k=5) |
| 模型大小 | ~500MB |

### XGBoost 模型

| 指标 | 数值 |
|------|------|
| 测试准确率 | ~98.0% |
| 推理时间 | ~5ms |
| 模型大小 | ~50MB |

## Web 界面功能

### 主要功能

- **手写输入**：支持鼠标和触摸屏手写
- **实时识别**：点击识别按钮立即显示结果
- **置信度显示**：显示预测置信度和进度条
- **概率分布**：显示所有数字的预测概率
- **清除画布**：一键清除手写内容
- **加载动画**：识别过程中显示加载状态

### 界面特点

- 现代化渐变设计
- 响应式布局，适配各种设备
- 流畅的动画效果
- 直观的结果展示

## 配置说明

### 数据预处理

```python
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量，值域 [0, 1]
    transforms.Normalize(mean=[0.1307], std=[0.3081])  # MNIST 标准化参数
])
```

### 训练参数

在 `model.py` 中可以调整以下参数：

```python
epochs = 20              # 训练轮数
batch_size = 128         # 批量大小
learning_rate = 0.001     # 学习率
weight_decay = 1e-4       # 权重衰减
```

### Web 应用配置

在 `app.py` 中可以调整：

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

- `debug=True`：开发模式，自动重载
- `host='0.0.0.0'`：允许外部访问
- `port=5000`：端口号

## 常见问题

### 1. 数据集下载失败

如果 MNIST 数据集下载失败，可以手动下载：

```bash
# 创建数据集目录
mkdir -p dataset/MNIST/raw

# 手动下载并解压到 dataset/MNIST/raw/
# 下载地址：http://yann.lecun.com/exdb/mnist/
```

### 2. 模型加载失败

确保模型文件存在：

```bash
# 检查模型文件
ls models/cnn.pth

# 如果不存在，重新训练模型
python model.py
```

### 3. CUDA 不可用

如果遇到 CUDA 相关错误，确保：

- 安装了正确版本的 PyTorch（CUDA 版本）
- NVIDIA 驱动已正确安装
- 或者使用 CPU 模式（代码已自动处理）

### 4. Web 端口被占用

如果 5000 端口被占用，修改 `app.py` 中的端口号：

```python
app.run(port=8080)  # 使用其他端口
```

### 5. 识别结果不准确

如果识别结果不准确，可以尝试：

- 重新训练模型，增加训练轮数
- 使用数据增强提高模型泛化能力
- 调整模型架构（增加层数或神经元数量）
- 确保手写数字清晰、居中

## 改进方向

### 模型优化

- 使用更深的网络架构（ResNet、VGG）
- 添加注意力机制（SE 模块）
- 使用数据增强（旋转、平移、缩放）
- 实现学习率调度（Cosine Annealing）
- 添加 Dropout 防止过拟合

### 训练优化

- 使用混合精度训练（AMP）
- 实现早停机制
- 使用验证集进行模型选择
- 添加梯度裁剪
- 实现模型集成

### Web 优化

- 添加用户注册和登录功能
- 保存识别历史记录
- 支持批量识别
- 添加模型切换功能
- 实现实时预测（边写边识别）

## 依赖项

```
pandas~=2.3.3
numpy~=1.26.4
joblib~=1.5.2
xgboost~=2.1.4
torchsummary~=1.5.1
scikit-learn~=1.5.1
torchvision~=0.15.2
torch~=2.1.0
flask~=3.0.0
tqdm~=4.66.0
```

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- MNIST 数据集：Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- PyTorch 框架：Facebook AI Research
- Flask 框架：Pallets Projects

## 联系方式

- 项目主页：[GitHub Repository](https://github.com/yourusername/mnist_digit_recognition)
- 问题反馈：[Issues](https://github.com/yourusername/mnist_digit_recognition/issues)
- 邮箱：your.email@example.com

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 实现 CNN、KNN、XGBoost 三种算法
- 添加 Web 界面
- 支持实时识别
- 完善文档和注释

---

**注意**：本项目仅供学习和研究使用。在生产环境中使用前，请进行充分的测试和优化。
