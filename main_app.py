from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path
import shutil
import json

# Initialize FastAPI app
app = FastAPI(title="Dog Breed Identification API", version="1.0.0")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and class labels
model = None
class_labels = []

def load_ml_components():
    """
    Load the trained dog breed classification model and class indices
    """
    global model, class_labels
    try:
        model_path = 'dog_breed_model.h5'
        if os.path.exists(model_path):
            model = load_model(model_path)
            print("✅ Model loaded successfully!")
        else:
            print(f"⚠️ Warning: {model_path} not found. Prediction will not work.")

        indices_path = 'class_indices.json'
        if os.path.exists(indices_path):
            with open(indices_path, 'r') as f:
                indices = json.load(f)
                # Invert dictionary: {breed: index} -> {index: breed}
                class_labels = {v: k for k, v in indices.items()}
            print("✅ Class indices loaded.")
        else:
            print(f"⚠️ Warning: {indices_path} not found. Using default list.")
            # Fallback list
            default_breeds = [
                'Golden_Retriever', 'Labrador_Retriever', 'German_Shepherd', 'Bulldog', 'Beagle',
                'Poodle', 'Rottweiler', 'Yorkshire_Terrier', 'Boxer', 'Siberian_Husky',
                'Dachshund', 'Border_Collie', 'Great_Dane', 'Shih_Tzu', 'Doberman_Pinscher',
                'Australian_Shepherd', 'Chihuahua', 'Pug', 'Cocker_Spaniel', 'Shiba_Inu'
            ]
            class_labels = {i: breed for i, breed in enumerate(default_breeds)}
            
        return True
    except Exception as e:
        print(f"❌ Error loading ML components: {e}")
        return False

def preprocess_image(img_path):
    """
    Preprocess uploaded image for model prediction
    """
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_breed(img_array):
    """
    Predict dog breed from preprocessed image array
    """
    global model, class_labels
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        # Get breed name
        predicted_breed = class_labels.get(predicted_class_index, "Unknown Breed")
        
        return predicted_breed, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction Error", 0.0

@app.on_event("startup")
async def startup_event():
    load_ml_components()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_dog_breed(request: Request, file: UploadFile = File(...)):
    """
    Handle image upload and prediction, rendering result on the same page
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "File must be an image"
            })
        
        # Save uploaded image
        uploads_dir = Path("static/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = uploads_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Preprocess image
        img_array = preprocess_image(str(file_path))
        
        if img_array is None:
             return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Error processing image"
            })

        # Make prediction
        breed, confidence = predict_breed(img_array)
        
        # Use breed name directly (retaining underscores as shown in user's design)
        formatted_breed = breed
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": formatted_breed,
            "confidence": f"{confidence:.2f}",
            "image_url": f"/static/uploads/{file.filename}"
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"An error occurred: {str(e)}"
        })

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
