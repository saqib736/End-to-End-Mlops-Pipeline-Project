import numpy as np
from mlproject import logger
from mlproject.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.validation_status = True  # Initialize as True
    
    def validate(self, dataset) -> bool:
        try:
            (x_train, y_train), (x_test, y_test) = dataset
            self.validate_dataset(x_train, y_train)
            self.validate_dataset(x_test, y_test)
        except ValueError as e:
            self.validation_status = False
            logger.info(f"Validation failed: {e}")
        finally:
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f'Validation status: {self.validation_status}')
        
        return self.validation_status
    
    def validate_dataset(self, images, labels):
        
        image_shape = self.config.schema.data.images.shape
        pixel_min = self.config.schema.data.images.pixel_value_range.min
        pixel_max = self.config.schema.data.images.pixel_value_range.max
        label_min = self.config.schema.data.labels.range.min
        label_max = self.config.schema.data.labels.range.max  
        
        for image, label in zip(images, labels):
            self.validate_image_shape(image, image_shape)
            self.validate_pixel_values(image, pixel_min, pixel_max)
            self.validate_label_range(label, label_min, label_max)
            
        if self.config.schema.integrity_checks.num_images_equals_num_labels:
            self.validate_integrity(images, labels)
            
        if self.config.schema.duplicate_check.enabled:
            self.check_for_duplicates(images)
            
        if self.config.schema.missing_data_check.enabled:
            self.check_for_missing_data(images, labels)

    def validate_image_shape(self, image, expected_shape):
        if image.shape != tuple(expected_shape):
            self.validation_status = False
            raise ValueError(f"Image shape mismatch. Expected {expected_shape}, but got {image.shape}")
            
    def validate_pixel_values(self, image, min_value, max_value):
        if image.min() < min_value or image.max() > max_value:
            self.validation_status = False
            raise ValueError(f"Pixel values out of range. Expected [{min_value}, {max_value}], but got [{image.min()}, {image.max()}]")
            
    def validate_label_range(self, label, min_label, max_label):
        if not (min_label <= label <= max_label):
            self.validation_status = False
            raise ValueError(f"Label {label} out of range. Expected between {min_label} and {max_label}")
            
    def validate_integrity(self, images, labels):
        if len(images) != len(labels):
            self.validation_status = False
            raise ValueError(f"Number of images ({len(images)}) does not match the number of labels ({len(labels)})")
            
    def check_for_duplicates(self, images):
        unique_images = set(map(lambda x: x.tobytes(), images))
        if len(unique_images) != len(images):
            self.validation_status = False
            raise ValueError("Duplicate images found in the dataset")
            
    def check_for_missing_data(self, images, labels):
        if any(np.isnan(image).any() for image in images):
            self.validation_status = False
            raise ValueError("NaN values found in image data")
            
        if any(label is None for label in labels):
            self.validation_status = False
            raise ValueError("Missing label values")
