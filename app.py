from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import io
import base64
from model import cnn

app = Flask(__name__)


# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cnn()
model.load_state_dict(torch.load('models/cnn.pth', map_location=device))
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

# 根路由
@app.route('/')
def index():
    return render_template('index.html')

# 预测路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从请求中获取图像
        data = request.json
        image_data = data['image'].split(',')[1]  # 移除 "data:image/png;base64," 前缀
        image_bytes = io.BytesIO(base64.b64decode(image_data))
        image = Image.open(image_bytes).convert('L')  # 转换为灰度图
        
        # 预处理
        image = image.resize((28, 28))
        # 网页已经是黑底白字，不需要反转
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_digit = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_digit].item()
            all_probabilities = probabilities[0].cpu().numpy().tolist()
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': round(confidence, 4),
            'all_probabilities': all_probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)