from mlproject.constants import *
from mlproject.entity.config_entity import DataIngestionConfig, DataValidationConfig, ModelTrainerConfig
from mlproject.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            train_images_path=config.train_images_path,
            train_labels_path=config.train_labels_path,
            test_images_path=config.test_images_path,
            test_labels_path=config.test_labels_path
        )
        
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            schema=self.schema
        )
        
        return data_validation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        model_params = self.params.CNN_model
        
        
        create_directories([config.root_dir])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            model_name=config.model_name,
            num_epoch=model_params.num_epoch,
            batch_size=model_params.batch_size,
            learning_rate=model_params.learning_rate
        )
        
        return model_trainer_config