from mlproject.config.configuration import ConfigurationManager
from mlproject.components.model_evaluation import ModelEvaluation 

class ModeLEvaluationPipeline:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        
    def run(self):
        config = ConfigurationManager()
        evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(evaluation_config, self.test_dataset)
        model_evaluation.log_into_mlflow()