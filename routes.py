from fastapi import APIRouter, File, UploadFile, HTTPException
from model.predict import ImagePredictor
import os
import shutil

router = APIRouter()
predictor = ImagePredictor("model/image_classifier.h5")

# Ensure the temp directory exists
os.makedirs("temp", exist_ok=True)

@router.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file to the temp directory
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform prediction
        result = predictor.predict(file_path)

        # Cleanup temporary file
        os.remove(file_path)

        return {"file_name": file.filename, "prediction": result}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
