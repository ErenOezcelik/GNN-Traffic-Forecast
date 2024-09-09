from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir: Path 
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen = True)
class DataPreprocessingConfig:
    root_dir: Path
    preprocessed_data_file: Path 
    graph_signal_matrix_filename : str 
    params_num_of_vertices : int
    params_points_per_hour : int
    params_num_for_predict : int
    params_num_of_weeks : int
    params_num_of_days : int
    params_num_of_hours : int    
    