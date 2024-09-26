from mlproject.config.configuration import ConfigurationManager
from mlproject.components.model_trainer import ModelTrainer

class ModelTrainerPipeline:
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        
    def run(self):
        config = ConfigurationManager()
        trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(trainer_config, self.train_dataset)
        model_trainer.train()