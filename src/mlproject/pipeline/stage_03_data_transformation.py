from pathlib import Path
from mlproject import logger
from torchvision.transforms import transforms
from mlproject.components.data_transformation import DataTransformation

class DataTransformationPipeline:
    def __init__(self, dataset):
        self.data = dataset
        
    def run(self):
        train_dataset = None
        test_dataset = None
        
        try:
            status_file = Path("artifacts/data_validation/status.txt")
            if status_file.exists():
                with open(status_file, "r") as f:
                    status = f.read().split(" ")[-1]
            else:
                raise FileNotFoundError("Status file not found in the specified path.")
            
            if status == "True":
                (x_train, y_train), (x_test, y_test) = self.data
                
                # Define transformations
                transform = transforms.Compose([
                    transforms.ToTensor(),  
                    transforms.Normalize((0.5,), (0.5,))
                ])
                
                train_dataset = DataTransformation(images=x_train, 
                                                   labels=y_train, 
                                                   transform=transform)
                
                test_dataset = DataTransformation(images=x_test,
                                                  labels=y_test,
                                                  transform=transform)
            else:
                raise Exception("Data schema is not valid")
        
        except Exception as e:
            logger.info(f"An error occurred during the data transformation pipeline: {e}")
        
        return train_dataset, test_dataset
