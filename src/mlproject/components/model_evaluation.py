from datetime import datetime
import torch
from torch import load
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import dagshub
import numpy as np
import mlflow.pytorch
from pathlib import Path
from mlproject import logger
from mlflow.models import infer_signature
from mlproject.utils.common import save_json
from mlproject.entity.config_entity import ModelEvaluationConfig
from mlproject.components.image_classifier_model import ImageClassifier

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, test_dataset):
        self.config = config
        self.test_dataset = test_dataset
        
    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        
        return accuracy, precision, recall, f1
        
    def log_into_mlflow(self):
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"Evaluation_Run_{current_time}"
    
        test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, num_workers=2)
        
        model = ImageClassifier()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        model.load_state_dict(load(self.config.model_path))
        model.eval()
        
        logger.info("initializing dagshub and mlflow...")
        dagshub.init(repo_owner='saqib736', repo_name='End-to-End-Mlops-Pipeline-Project', mlflow=True)
        logger.info("initialization completed")
        
        logger.info("mlflow logging started...")
        with mlflow.start_run(run_name=run_name):
            
            true_labels = []
            pred_labels = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to('cpu')
                    
                    output = model(images)
                    _, preds = torch.max(output, 1)
                    
                    true_labels.extend(labels.numpy())
                    pred_labels.extend(preds.cpu().numpy())       
            
            signature = infer_signature(np.array(true_labels), np.array(pred_labels))
            (accuracy, precision, recall, f1) = self.eval_metrics(true_labels, pred_labels)
            
            scores = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
            save_json(path=Path(self.config.metric_file_name), data=scores) 
            
            logger.info("logging model parameters to mlflow")
            mlflow.log_params({
                        "epochs": self.config.num_epoch,
                        "learning_rate": self.config.learning_rate,
                        "batch_size": self.config.batch_size
                    })
            
            logger.info("logging evaluation metrics to mlflow")
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1', f1)
            
            logger.info("registering model to mlflow")
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="pytorch-model",
                signature=signature,
                registered_model_name="pytorch_CNN_classifier_mnist"
            )
            
            logger.info("mlflow logging completed")
                    
                
                
        