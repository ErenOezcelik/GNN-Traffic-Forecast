from src.GNN_PEMS.constants import *
from src.GNN_PEMS.utils.common import read_yaml, create_directories
from src.GNN_PEMS.entity.config_entity import DataIngestionConfig
from src.GNN_PEMS.entity.config_entity import DataPreprocessingConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 

        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])
        
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir = config.root_dir, 
            preprocessed_data_file = config.preprocessed_data_file,
            graph_signal_matrix_filename = config.graph_signal_matrix_filename,
            params_num_of_vertices = self.params.num_of_vertices,             
            params_points_per_hour = self.params.points_per_hour, 
            params_num_for_predict = self.params.num_for_predict,
            params_num_of_weeks = self.params.num_of_weeks,
            params_num_of_days = self.params.num_of_days,
            params_num_of_hours = self.params.num_of_hours,       
        )
        return data_preprocessing_config