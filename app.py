from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from predict_helper import predict_damage

app = FastAPI(
    title="Car Damage Detection API",
    version="1.0"
)

@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict_damage")
async def predict(image: UploadFile = File(...)):
    try:
        img = Image.open(image.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = predict_damage(img)
    return result
