import os
from flask import Flask, render_template, request, jsonify
from mlproject.pipeline.inference import InferencePipeline

app = Flask(__name__)

# Instantiate the InferencePipeline only once
inference_pipeline = InferencePipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python3 main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file:
            img_bytes = file.read()
            prediction = inference_pipeline.predict_image(img_bytes)
            return jsonify({'prediction': str(prediction)})
        
        return jsonify({'error': 'Something went wrong'})
    
    except Exception as e:
        # Return a detailed error message in case of failure
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0", port = 8080)
