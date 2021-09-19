''' This method loads processed data and combines them based on user-specified videos '''

import json
from natsort import natsorted
import numpy as np

class DataLoader():

    def __init__(self, MetaData, work_dir):
        self.MetaData = MetaData
        self.processed_data_dir = work_dir
        self.fps = 29.69 # think of better way to integrate this

    def infer_hierarchy(self, video_ids):
        ''' infer hierarchy based on timestamps and animal_id'''
        video_ids = natsorted(video_ids)
        hierarchy = {}
        for id in video_ids:
            animal = self.MetaData.get_animal(id)
            if animal not in self.combined_data:
                hierarchy[animal] = []
            hierarchy[animal].append(id)
        return hierarchy

    def get_data(self, video_ids, invert = False):
        ''' combine data from json files '''
        combined_data = {}
        if type(video_ids) == list:
            video_ids = natsorted(video_ids)
        else:
            video_ids = [video_ids]
        if len(video_ids) > 1:
            print(f'* Combining data (json) for {len(video_ids)} videos..')
        for id in video_ids:
            combined_data[id] = self._load_processed_data(id)
        if invert:
            combined_data = self.invert_data_hierarchy(combined_data)
        print('  Data combined.')
        return combined_data

    def invert_data_hierarchy(self, combined_data):
        ''' backward compatibility requires datatype to be of
        data[datatype][video_id] for plotting functions to work
        input: data[video_id][datatype]. output: data[datatype][video_id]'''
        inverted_data = {}
        videos = list(combined_data.keys())
        datatypes = list(combined_data[videos[0]].keys())
        for datatype in datatypes:
            inverted_data[datatype] = {}
            for video in videos:
                inverted_data[datatype][video] = combined_data[video][datatype]
        return inverted_data

    def convert_to_numpy(self, data):
        for entry, d in data.items():
            data[entry] = np.array(d)
        return data

    def _convert_hours_to_frames(self, hours):
        return int(hours * self.fps * 3600)

    def _truncate_data(self, data, truncate_frame):
        truncate_bout = np.where(np.array(data['traj indices'])[:,-1] > truncate_frame)[0][0]
        for type, d in data.items():
            if type in ['warped numpy', 'presence', 'discrete positions']:
                data[type] = d[:truncate_frame]
            elif type in ['traj indices', 'improved discrete positions (bouts)']:
                data[type] = d[:truncate_bout]
        return data

    def _load_processed_data(self, video_id, convert_to_numpy = True,
                             include_raw_keypoints = False, include_raw_tiles = False):
        print(f'* Loading data: [{video_id}]')
        data = json.load(open(f'{self.processed_data_dir}/{video_id}.json', "r"))
        truncate_in_hours = self.MetaData.get_data_truncation(video_id)
        if truncate_in_hours:
            truncate_in_frames = self._convert_hours_to_frames(truncate_in_hours)
            data = self._truncate_data(data, truncate_in_frames)
        # if convert_to_numpy:
        #     self.convert_to_numpy(data)
        if not include_raw_keypoints:
            data.pop('keypoints (raw)', None)
        if not include_raw_tiles == False:
            data.pop('discrete positions', None)
        return data

