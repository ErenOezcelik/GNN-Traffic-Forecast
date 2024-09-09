import os
import numpy as np
from pathlib import Path
from src.GNN_PEMS.entity.config_entity import DataPreprocessingConfig
from GNN_PEMS import logger


class DataPreprocessing: 
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

        self.root_dir = self.config.root_dir
        self.graph_signal_matrix_filename = self.config.graph_signal_matrix_filename
        self.num_of_vertices = self.config.params_num_of_vertices
        self.points_per_hour = self.config.params_points_per_hour
        self.num_for_predict = self.config.params_num_for_predict
        self.num_of_weeks = self.config.params_num_of_weeks
        self.num_of_days = self.config.params_num_of_days
        self.num_of_hours = self.config.params_num_of_hours
        
    def search_data(self, sequence_length, num_of_depend, label_start_idx, units):
        '''
        Parameters
        ----------
        sequence_length: int, length of all history data
        num_of_depend: int,
        label_start_idx: int, the first index of predicting target
        num_for_predict: int, the number of points will be predicted for each sample
        units: int, week: 7 * 24, day: 24, recent(hour): 1
        points_per_hour: int, number of points per hour, depends on data
        Returns
        ----------
        list[(start_idx, end_idx)]
        '''
        
        if self.points_per_hour < 0:
            raise ValueError("points_per_hour should be greater than 0!")

        if label_start_idx + self.num_for_predict > sequence_length:
            return None

        x_idx = []
        for i in range(1, num_of_depend + 1):
            start_idx = label_start_idx - self.points_per_hour * units * i
            end_idx = start_idx + self.num_for_predict
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))
            else:
                return None

        if len(x_idx) != num_of_depend:
            return None

        return x_idx[::-1]
    
    
    def get_sample_indices(self, data_sequence, label_start_idx):
        '''
        Parameters
        ----------
        data_sequence: np.ndarray shape is (sequence_length, num_of_vertices, num_of_features)
        num_of_weeks, num_of_days, num_of_hours: int
        label_start_idx: int, the first index of predicting target
        num_for_predict: int,the number of points will be predicted for each sample
        points_per_hour: int, default 12, number of points per hour
        Returns
        ----------
        week_sample: np.ndarray shape is (num_of_weeks * points_per_hour, num_of_vertices, num_of_features)
        day_sample: np.ndarray shape is (num_of_days * points_per_hour,  num_of_vertices, num_of_features)
        hour_sample: np.ndarray   shape is (num_of_hours * points_per_hour, num_of_vertices, num_of_features)
        target: np.ndarray shape is (num_for_predict, num_of_vertices, num_of_features)
        '''
        week_sample, day_sample, hour_sample = None, None, None
        
        #------------------------------------Ignore
        if label_start_idx + self.num_for_predict > data_sequence.shape[0]: 
            return week_sample, day_sample, hour_sample, None

        if self.num_of_weeks > 0:
            week_indices = self.search_data(data_sequence.shape[0], self.num_of_weeks, label_start_idx, 7 * 24)
            if not week_indices:
                return None, None, None, None

            week_sample = np.concatenate([data_sequence[i: j] for i, j in week_indices], axis=0)

        if self.num_of_days > 0:
            day_indices = self.search_data(data_sequence.shape[0], self.num_of_days,  label_start_idx, 24)
            if not day_indices:
                return None, None, None, None

            day_sample = np.concatenate([data_sequence[i: j] for i, j in day_indices], axis=0)
        #----------------------------------Continue
        if self.num_of_hours > 0:
            hour_indices = self.search_data(data_sequence.shape[0], self.num_of_hours, label_start_idx, 1)
            if not hour_indices:
                return None, None, None, None
            hour_sample = np.concatenate([data_sequence[i: j] for i, j in hour_indices], axis=0)
        
        if self.num_of_hours > 10:
            return 1
        target = data_sequence[label_start_idx: label_start_idx + self.num_for_predict]

        return week_sample, day_sample, hour_sample, target
    
    
    def read_and_generate_dataset(self):
        '''
        Parameters
        ----------
        graph_signal_matrix_filename: str, path of graph signal matrix file
        num_of_weeks, num_of_days, num_of_hours: int
        num_for_predict: int
        points_per_hour: int, default 12, depends on data
        Returns
        ----------
        feature: np.ndarray, shape is (num_of_samples, num_of_depend * points_per_hour, num_of_vertices, num_of_features)
        target: np.ndarray, shape is (num_of_samples, num_of_vertices, num_for_predict)
        '''
        # Read original data 
        data_seq = np.load(self.graph_signal_matrix_filename)['data']  # (sequence_length, num_of_vertices, num_of_features) (16992, 307, 3)
        
        all_samples = []
        for idx in range(data_seq.shape[0]):
            sample = self.get_sample_indices(data_seq, idx)
            if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
                continue

            week_sample, day_sample, hour_sample, target = sample #  week_sample, day_sample are None because we are predicting per hour
            #print(target.shape) # hour_sample and target (12, 307, 3)
            sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]
            #Ignore
            if self.num_of_weeks > 0:
                week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(week_sample)

            if self.num_of_days > 0:
                day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(day_sample)
            #Continue
            if self.num_of_hours > 0:
                hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
                sample.append(hour_sample)

            target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
            sample.append(target)
            time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
            sample.append(time_sample)
            all_samples.append(sample)#sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

        split_line1 = int(len(all_samples) * 0.6)
        split_line2 = int(len(all_samples) * 0.8)

        training_set = [np.concatenate(i, axis=0)  for i in zip(*all_samples[:split_line1])] #[(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
        validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1: split_line2])]
        testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]

        return training_set, validation_set, testing_set
    
    def normalization(train, val, test):
        '''
        Parameters
        ----------
        train, val, test: np.ndarray (B,N,F,T)
        Returns
        ----------
        stats: dict, two keys: mean and std
        train_norm, val_norm, test_norm: np.ndarray,
                                        shape is the same as original
        '''

        assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
        mean = train.mean(axis=(0,1,3), keepdims=True)
        std = train.std(axis=(0,1,3), keepdims=True)
        print('mean.shape:',mean.shape)
        print('std.shape:',std.shape)

        def normalize(x):
            return (x - mean) / std

        train_norm = normalize(train)
        val_norm = normalize(val)
        test_norm = normalize(test)

        return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm
        

    def preprocess_data(self):

        training_set, validation_set, testing_set = DataPreprocessing.read_and_generate_dataset(self)

        train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
        val_x = np.concatenate(validation_set[:-2], axis=-1)
        test_x = np.concatenate(testing_set[:-2], axis=-1)

        train_target = training_set[-2]  # (B,N,T)
        val_target = validation_set[-2]
        test_target = testing_set[-2]

        train_timestamp = training_set[-1]  # (B,1)
        val_timestamp = validation_set[-1]
        test_timestamp = testing_set[-1]

        (stats, train_x_norm, val_x_norm, test_x_norm) = DataPreprocessing.normalization(train_x, val_x, test_x)

        all_data = {'train': { 'x': train_x_norm, 'target': train_target,'timestamp': train_timestamp},
                    'val': {'x': val_x_norm, 'target': val_target, 'timestamp': val_timestamp},
                    'test': {'x': test_x_norm, 'target': test_target, 'timestamp': test_timestamp},
                    'stats': {'_mean': stats['_mean'], '_std': stats['_std']} }

        print('train x:', all_data['train']['x'].shape)
        print('train target:', all_data['train']['target'].shape)
        print('train timestamp:', all_data['train']['timestamp'].shape)
        print()
        print('val x:', all_data['val']['x'].shape)
        print('val target:', all_data['val']['target'].shape)
        print('val timestamp:', all_data['val']['timestamp'].shape)
        print()
        print('test x:', all_data['test']['x'].shape)
        print('test target:', all_data['test']['target'].shape)
        print('test timestamp:', all_data['test']['timestamp'].shape)
        print()
        print('train data _mean :', all_data['stats']['_mean'].shape, all_data['stats']['_mean'])
        print('train data _std :', all_data['stats']['_std'].shape, all_data['stats']['_std'])
        
        return all_data
        
    def safe_data(self, all_data): 
        file = os.path.basename(self.graph_signal_matrix_filename).split('.')[0]
        filename = os.path.join(self.root_dir, file + '_r' + str(self.num_of_hours) + '_d' + str(self.num_of_days) + '_w' + str(self.num_of_weeks)) + '_astcgn'
        print('save file:', filename)
        np.savez_compressed(filename,
                        train_x=all_data['train']['x'],train_target=all_data['train']['target'],train_timestamp=all_data['train']['timestamp'],
                        val_x=all_data['val']['x'], val_target=all_data['val']['target'],val_timestamp=all_data['val']['timestamp'],
                        test_x=all_data['test']['x'], test_target=all_data['test']['target'], test_timestamp=all_data['test']['timestamp'],
                        mean=all_data['stats']['_mean'], std=all_data['stats']['_std'])