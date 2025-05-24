import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Car Damage Detector", layout="centered")
st.title("Car Damage Detector")

classes = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']

class CarClassifierResnet(nn.Module):
    def __init__(self, num_classes=6):
        super(CarClassifierResnet, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)

@st.cache_data
def load_model():
    model = CarClassifierResnet(num_classes=len(classes))
    state_dict = torch.load("Model_Resnet.pth", map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key.replace("model.", "resnet.")] = value
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

model = load_model()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict(image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]

def visualize_prediction(image, prediction):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"Prediction: {prediction}", fontsize=14, color='red')
    ax.axis("off")
    st.pyplot(fig)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    prediction = predict(image)
    st.write(f"### Predicted Class: {prediction}")
    visualize_prediction(image, prediction)
