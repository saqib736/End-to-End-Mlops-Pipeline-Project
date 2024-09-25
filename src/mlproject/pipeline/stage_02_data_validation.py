from mlproject.config.configuration import ConfigurationManager
from mlproject.components.data_ingestion import DataIngestion
from mlproject.components.data_validation import DataValidation

class DataValidationTrainingPipeline:
    def __init__(self, dataset):
        self.data = dataset
    
    def run(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(data_validation_config)
        validation_status = data_validation.validate(self.data)
        return validation_status
        
        