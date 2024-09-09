from src.GNN_PEMS import logger 
from src.GNN_PEMS.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.GNN_PEMS.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
    
    
STAGE_NAME = "Data Preprocessing stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")    
    obj = DataPreprocessingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

