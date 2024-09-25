import os
import gdown
import zipfile
from pathlib import Path
from mlproject import logger
from mlproject.utils.common import get_szie
from mlproject.entity.config_entity import DataIngestionConfig
from mlproject.utils.mnist_dataloader import MnistDataloader


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            gdown.download(self.config.source_URL, 
                           self.config.local_data_file, 
                           quiet=False)
            
            logger.info(f"{Path(self.config.local_data_file).name} downloaded!")
        else:
            logger.info(f"File already exists of size: {get_szie(Path(self.config.local_data_file))}")
            
    def extract_zip_file(self):
        unzip_path = Path(self.config.unzip_dir)
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_file:
            zip_file.extractall(unzip_path)
        logger.info(f"{Path(self.config.local_data_file).name} extracted!")
            
        self.train_images_path = os.path.join(unzip_path, self.config.train_images_path)
        self.train_labels_path = os.path.join(unzip_path, self.config.train_labels_path)
        self.test_images_path = os.path.join(unzip_path, self.config.test_images_path)
        self.test_labels_path = os.path.join(unzip_path, self.config.test_labels_path)
        
    def load_dataset(self):
        mnist_loader = MnistDataloader(
            training_images_filepath=self.train_images_path,
            training_labels_filepath=self.train_labels_path,
            test_images_filepath=self.test_images_path,
            test_labels_filepath=self.test_labels_path
        )
        logger.info(f"Data has been loaded!")
        return mnist_loader.load_data()