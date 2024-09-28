import torch
from torch import nn, save
from torch.optim import Adam
from torch.utils.data import DataLoader

import os
from mlproject import logger
from mlproject.entity.config_entity import ModelTrainerConfig
from mlproject.components.image_classifier_model import ImageClassifier

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, train_dataset):
        self.config = config
        self.train_dataset = train_dataset
        
    def train(self):
        
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2)
        
        model = ImageClassifier()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.num_epoch):
            model.train()
            for images, labels in train_loader:
                
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
            
            logger.info(f"Epoch [{epoch+1}/{self.config.num_epoch}], Loss {loss.item():.4f}")
        
        save(model.state_dict(), os.path.join(self.config.root_dir, self.config.model_name))
        logger.info(f"Model saved at {os.path.join(self.config.root_dir, self.config.model_name)}")
            
        