import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import time

from mlproject.pipeline.inference import InferencePipeline

# Update static_folder to point to the exported static files directory
app = Flask(__name__, static_folder='frontend/out', static_url_path='')
app.wsgi_app = ProxyFix(app.wsgi_app)
CORS(app)

inference_pipeline = InferencePipeline()

# Serve Next.js static files (e.g., JavaScript, CSS)
@app.route('/_next/static/<path:path>')
def next_static(path):
    return send_from_directory(os.path.join(app.static_folder, '_next', 'static'), path)

# Serve other static assets in the `public` folder
@app.route('/public/<path:path>')
def public_files(path):
    return send_from_directory(os.path.join(app.static_folder, 'public'), path)

# Serve the exported HTML files, defaulting to index.html
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # Check if the requested path matches a file directly
    file_path = os.path.join(app.static_folder, path)
    if os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    
    # Default to index.html for any other paths (for SPA routing)
    return send_from_directory(app.static_folder, 'index.html')

# API endpoint for image classification
@app.route('/api/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image = request.files['image']
        if image:
            img_bytes = image.read()
            classification = inference_pipeline.predict_image(img_bytes)
            return jsonify({'classification': classification})
            
        return jsonify({'error': 'Something went wrong'})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
    time.sleep(2)  # Simulate processing time

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
