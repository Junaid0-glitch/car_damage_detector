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

# here

def predict_damage(image: Image.Image):
    image = image.convert("RGB")

    # -------- YOLO --------
    yolo_results = yolo_model.predict(
        source=image,
        conf=0.05,
        imgsz=640,
        verbose=False
    )

    bboxes = []
    if yolo_results[0].boxes is not None:
        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            bboxes.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4)
            })

    # -------- CLASSIFICATION --------
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = clf_model(img_tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)

    return {
        "damage_detected": len(bboxes) > 0,
        "damage_type": class_names[idx.item()],
        "confidence": round(conf.item(), 4),
        "bboxes": bboxes
    }
