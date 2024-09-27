from mlproject import logger
from mlproject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlproject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlproject.pipeline.stage_03_data_transformation import DataTransformationPipeline
from mlproject.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from mlproject.pipeline.stage_05_model_evaluation import ModeLEvaluationPipeline

STAGE_ONE = "Data Ingestion"
STAGE_TWO = "Data Validation"
STAGE_THREE = "Data Transformation"
STAGE_FOUR = "Model Training"
STAGE_FIVE = "Model Evaluation"

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

try:
    logger.info(f">>>>>> stage {STAGE_THREE} has started <<<<<<")
    data_transformation = DataTransformationPipeline(loaded_data)
    train_dataset, test_dataset = data_transformation.run()
    logger.info(f">>>>>> stage {STAGE_THREE} completed <<<<<<\n\nx===================x")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f">>>>>> stage {STAGE_FOUR} has started <<<<<<")
    model_trainer = ModelTrainerPipeline(train_dataset)
    model_trainer.run()
    logger.info(f">>>>>> stage {STAGE_FOUR} completed <<<<<<\n\nx===================x")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f">>>>>> stage {STAGE_FIVE} has started <<<<<<")
    model_evaluation = ModeLEvaluationPipeline(test_dataset)
    model_evaluation.run()
    logger.info(f">>>>>> stage {STAGE_FIVE} completed <<<<<<\n\nx===================x")
except Exception as e:
    logger.exception(e)
    raise e



