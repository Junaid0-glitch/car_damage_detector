import base64
import io
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models

# ---------------- YOLO ----------------
yolo_model = YOLO("artifacts/damage_detector.pt")

# ---------------- CLASSIFIER ----------------
class Car_Classifier_Resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights="DEFAULT")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class_names = [
    "F_Breakage",
    "F_Crushed",
    "F_Normal",
    "R_Breakage",
    "R_Crushed",
    "R_Normal"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf_model = Car_Classifier_Resnet(num_classes=6).to(device)
clf_model.load_state_dict(
    torch.load("artifacts/Damage_Classifier_Resnet_18.pth", map_location=device)
)
clf_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_damage(image: Image.Image):
    image = image.convert("RGB")
    
    # -------- 1. CLASSIFICATION (ResNet) --------
    # Run classification first as requested
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = clf_model(img_tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)
    
    damage_type = class_names[idx.item()]
    confidence_score = round(conf.item(), 4)

    # -------- 2. YOLO DETECTION --------
    yolo_results = yolo_model.predict(
        source=image,
        conf=0.05,
        imgsz=640,
        verbose=False
    )
    
    result = yolo_results[0]
    
    # Check if any boxes were detected
    damage_detected = result.boxes is not None and len(result.boxes) > 0

    # Generate the image with bounding boxes drawn
    # plot() returns a numpy array in BGR format (OpenCV style)
    plotted_image_bgr = result.plot()
    
    # Convert BGR to RGB
    plotted_image_rgb = plotted_image_bgr[..., ::-1]
    
    # Convert numpy array back to PIL Image
    final_image = Image.fromarray(plotted_image_rgb)

    # Encode image to Base64 to send to frontend
    buffered = io.BytesIO()
    final_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "damage_detected": damage_detected,
        "damage_type": damage_type,
        "confidence": confidence_score,
        "annotated_image": img_str  # Base64 string of the image
    }
