from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    train_images_path: str   
    train_labels_path: str
    test_images_path: str
    test_labels_path: str
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    schema: dict

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_name: str
    num_epoch: int
    batch_size: int
    learning_rate: float        
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    metric_file_name: Path
    num_epoch: int
    batch_size: int
    learning_rate: float
    