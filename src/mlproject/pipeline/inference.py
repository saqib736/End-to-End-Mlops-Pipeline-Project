import torch
from torch import load
from torchvision.transforms import transforms

import io
from PIL import Image
from pathlib import Path
from mlproject.components.image_classifier_model import ImageClassifier

class InferencePipeline:
    def __init__(self):
        self.model = ImageClassifier()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        model_path = Path('artifacts/model_trainer/model.pt')
        self.model.load_state_dict(load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
                        transforms.ToTensor(),  
                        transforms.Normalize((0.5,), (0.5,))
                    ])

    def predict_image(self, image):
        try:
            image = Image.open(io.BytesIO(image)).convert("L") 
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image)
                _, prediction = torch.max(output, 1)
            
            return prediction.item()  
        except Exception as e:
            print(f"Error processing image: {e}")
            return None