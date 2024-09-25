from mlproject import logger
from mlproject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlproject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

STAGE_ONE = "Data Ingestion"
STAGE_TWO = "Data Validation"

try:
    logger.info(f">>>>>> stage {STAGE_ONE} has started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    loaded_data = data_ingestion.run()
    logger.info(f">>>>>> stage {STAGE_ONE} completed <<<<<<\n\nx===================x")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f">>>>>> stage {STAGE_TWO} has started <<<<<<")
    data_validation = DataValidationTrainingPipeline(loaded_data)
    Validation_status = data_validation.run()
    logger.info(f">>>>>> stage {STAGE_TWO} completed <<<<<<\n\nx===================x")
except Exception as e:
    logger.exception(e)
    raise e



