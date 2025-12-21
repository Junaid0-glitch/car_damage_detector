# 🚗 Car Damage Detection & Classification System

🔗 **Live Streamlit App**: [https://cardamagedetector-e3smtwusr8p8kmhywkq7zy.streamlit.app/](https://cardamagedetector-e3smtwusr8p8kmhywkq7zy.streamlit.app/)

🤗 **Hugging Face Repository (Models + API)**: [https://huggingface.co/spaces/junaid17/car_damage_detector/tree/main](https://huggingface.co/spaces/junaid17/car_damage_detector/tree/main)

---

An end‑to‑end **Computer Vision application** that automatically **detects car damage regions** and **classifies the type of damage** using a **YOLO + ResNet18 hybrid architecture**. The system is fully productionized with an **API backend**, **Streamlit frontend**, and **cloud deployment on Hugging Face**.

---

## 🔍 Project Overview

This project solves a real‑world insurance and automotive inspection problem by answering two key questions from a single car image:

1. **Where is the damage?** → Object detection using **YOLO**
2. **What type of damage is it?** → Image classification using **ResNet‑18**

The pipeline first detects damage regions (bounding boxes) and then classifies the damage type, producing a clean, interpretable output suitable for insurance automation, vehicle inspection, and claim validation systems.

---

## 🧠 Architecture

```
Input Image
    │
    ├──► YOLO (Damage Detection)
    │        └── Bounding Boxes + Confidence
    │
    └──► ResNet‑18 (Damage Classification)
             └── Damage Type + Confidence

Final Output → JSON + Visual Overlay
```

---

## 🏷️ Damage Classes

The classifier is trained on **6 damage categories**:

* `F_Breakage`
* `F_Crushed`
* `F_Normal`
* `R_Breakage`
* `R_Crushed`
* `R_Normal`

(F = Front, R = Rear)

---

## 📊 Model Performance (Summary)

### Custom CNN (Baseline)

* Underfitting observed
* Limited generalization on validation data

### Transfer Learning (ResNet‑18)

* **~74% validation accuracy**
* Strong improvement in precision, recall, and F1‑score
* Robust performance across all damage classes

This confirms that **transfer learning is essential** for small/medium‑sized vision datasets.

---

## 🛠️ Tech Stack

### Core

* **Python 3.10+**
* **PyTorch**
* **Torchvision**
* **Ultralytics YOLOv8**

### Backend

* **FastAPI** (Inference API)
* **Pydantic** (Schema validation)

### Frontend

* **Streamlit** (Interactive UI)

### Deployment

* **Hugging Face Spaces** (API + Model Hosting)

## 🖼️ Screenshots

### 🔹 Streamlit Web Interface
<img src="https://github.com/user-attachments/assets/240383b0-ff5b-4de5-b4cf-0eed36f3a6fb" width="100%" />

---

### 🔹 Damage Detection & Classification Output
<img src="https://github.com/user-attachments/assets/1f2112fd-5b9d-4cfe-baaa-de049b9789b0" width="100%" />

---

### 🔹 API Response & Bounding Boxes
<img src="https://github.com/user-attachments/assets/a5877b84-fa94-4770-b8f1-b3449baa4b58" width="100%" />


---

## 📦 API Output Format

```json
{
  "damage_detected": true,
  "damage_type": "F_Breakage",
  "confidence": 0.87,
  "bboxes": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.91
    }
  ]
}
```

---

## 🧪 Key Features

* ✅ Hybrid **Detection + Classification** pipeline
* ✅ Transfer learning with ResNet‑18
* ✅ YOLO‑based spatial localization
* ✅ REST API for easy integration
* ✅ Interactive Streamlit frontend
* ✅ Cloud‑deployed and production‑ready

---

## 📁 Project Structure (High‑Level)

```
car-damage-detector/
│
├── app.py                  # FastAPI entry point
├── predict_helper.py       # YOLO + ResNet inference logic
├── models/                 # (Hosted on Hugging Face)
├── main.py        # Frontend UI
├── requirements.txt
└── README.md
```

---

## 💡 Use Cases

* 🚘 Insurance claim automation
* 🔍 Vehicle inspection systems
* 🧾 Damage assessment & reporting
* 🧠 AI‑powered automotive analytics

---

## 📌 Future Improvements

* Instance‑level classification per bounding box
* Damage severity estimation (minor / major)
* Multi‑angle image support
* Mobile‑friendly frontend

---

## 👨‍💻 Author

**Juddy**
AI / ML Engineer | Data Science & Computer Vision

---

## ⭐ Acknowledgements

* PyTorch & Torchvision Team
* Ultralytics YOLO
* Hugging Face Spaces

---

If you found this project useful, consider giving it a ⭐ on GitHub.
