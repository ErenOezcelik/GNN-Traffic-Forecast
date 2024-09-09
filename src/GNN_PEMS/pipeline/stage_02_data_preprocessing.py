import numpy as np 
from GNN_PEMS import logger 
from src.GNN_PEMS.config.configuration import ConfigurationManager
from src.GNN_PEMS.components.data_preprocessing import DataPreprocessing

STAGE_NAME = "Preprocessing stage"

class DataPreprocessingPipeline:
    def __init__(self):
        pass
    
    def main(self): 
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data = data_preprocessing.preprocess_data()
        data_preprocessing.safe_data(data)

