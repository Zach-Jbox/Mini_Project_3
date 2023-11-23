from flask import Flask, session, redirect, render_template,request,jsonify,url_for
from transformers import ViTImageProcessor, ViTForImageClassification
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/VIT_model', methods=['GET', 'POST'])
def VIT():
    if request.method == 'POST':
        
        img = request.files['photo']
        
        if img.filename == '':
            return render_template('VIT_model.html', error="No file uploaded")  
              
        image = Image.open(img)
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        
        if '.' in img.filename and img.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return render_template('VIT_model.html', error="Invalid file type. Please upload a JPEG or PNG file.")
        
        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        prediction = model.config.id2label[predicted_class_idx]
        
        return render_template('VIT_prediction.html', prediction=prediction)
    
    return render_template('VIT_model.html')

@app.route('/yolo_model', methods=['GET', 'POST'])
def yolo():
    if request.method == 'POST':
        
        img = request.files['photo']
        
        if img.filename == '':
            return render_template('yolo_model.html', error="No file uploaded")  
              
        image = Image.open(img)
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        
        if '.' in img.filename and img.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return render_template('yolo_model.html', error="Invalid file type. Please upload a JPEG or PNG file.")
        
        model = YOLO('yolov8n.pt')
        results = model(img, verbose=False)  # results list
        if img == img:
            for r in results:
                prediction_array = r.plot()  # plot a BGR numpy array of predictions
                prediction = Image.fromarray(prediction_array[..., ::-1])  # RGB PIL image
                save_path = os.path.join('static', 'media', 'prediction.jpg')
                prediction.save(save_path)
                return render_template('yolo_prediction.html', image_path=save_path)
            
    return render_template('yolo_model.html')

app.run(debug=True,port=5000)
