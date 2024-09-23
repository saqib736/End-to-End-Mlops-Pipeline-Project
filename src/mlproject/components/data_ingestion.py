import os
import gdown
from pathlib import Path
import zipfile
from mlproject import logger
from mlproject.utils.common import get_szie
from mlproject.entity.config_entity import DataIngestionConfig


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