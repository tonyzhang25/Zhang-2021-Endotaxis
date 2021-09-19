import os, glob
import numpy as np
import pandas

class MetaData():

    def __init__(self, path):
        self.metafile_path = path
        self.init_paths(os.path.dirname(path))
        self.read_metafile()
        self.add_videos_to_metafile()
        self.add_pose_to_metafile()
        self.add_LED_to_metafile()

    def init_paths(self, working_dir):
        self.working_dir = working_dir
        self.video_dir = f'{working_dir}/videos'
        self.pose_path = f'{working_dir}/pose'
        self.processed_data_dir = f'{working_dir}/processed_data'
        self.LED_dir = f'{working_dir}/LED_luminance'
        self.analysis_dir = f'{working_dir}/analysis'

    def read_metafile(self):
        self.meta_data = pandas.read_csv(self.metafile_path)
        # set ID column
        self.meta_data.set_index('ID', inplace=True)
        self.meta_data.index.name = 'ID'

    def add_LED_to_metafile(self):
        ''' add existing video paths to csv '''
        if os.path.exists(self.LED_dir):
            LED_names = os.listdir(self.LED_dir)
            for LED_name in LED_names:
                video_id = os.path.splitext(LED_name)[0]
                if video_id in self.meta_data.index and pandas.isnull(self.meta_data.loc[video_id, 'LED_path']):
                    write_path = f'{os.path.basename(self.LED_dir)}/{LED_name}'
                    self.meta_data.loc[video_id, 'LED_path'] = write_path
                    self.meta_data.to_csv(self.metafile_path)
                    print(f'* LED file path added to metafile for ID: {video_id}')

    def add_videos_to_metafile(self):
        ''' add existing video paths to csv (only if video ID exists already)'''
        if os.path.exists(self.video_dir): # only check if video folder is present
            video_names = os.listdir(self.video_dir)
            video_names = [video for video in video_names if video.endswith('.avi')]
            for video_name in video_names:
                video_id = os.path.splitext(video_name)[0]
                if video_id in self.meta_data.index and pandas.isnull(self.meta_data.loc[video_id, 'video_path']):
                    write_path = f'{os.path.basename(self.video_dir)}/{video_name}'
                    self.meta_data.loc[video_id, 'video_path'] = write_path
                    self.meta_data.to_csv(self.metafile_path)
                    print(f'* VIDEO added to metafile for ID: {video_id}')

    def add_pose_to_metafile(self):
        ''' add existing pose paths to csv '''
        if os.path.exists(self.pose_path):  # only check if folder is present
            pose_files = os.listdir(self.pose_path)
            pose_files = [pose for pose in pose_files if pose.endswith('.h5')]
            for pose_file in pose_files:
                video_id = pose_file.split('DeepCut')[0]
                if video_id in self.meta_data.index and pandas.isnull(self.meta_data.loc[video_id, 'pose_path']):
                    pose_path = f'{os.path.basename(self.pose_path)}/{pose_file}'
                    self.meta_data.loc[video_id, 'pose_path'] = pose_path
                    self.meta_data.to_csv(self.metafile_path)
                    print(f'* POSE added to metafile for ID: {video_id}')

    def return_entries_with_LED(self, overwrite = False):
        entries_w_LED_loc = self.meta_data.loc[self.meta_data['LED_location'].isnull() == False]
        if overwrite == False:
            entries_w_LED_loc = entries_w_LED_loc[entries_w_LED_loc['LED_path'].isnull()]
        return entries_w_LED_loc

    def return_entries_with_pose(self, overwrite = False):
        if overwrite:
            return self.meta_data
        else:
            entries_w_pose = self.meta_data.loc[self.meta_data['pose_path'].isnull() == False]
        return entries_w_pose

    def return_videos_no_posepath(self):
        entries_without_pose_path = self.meta_data.loc[self.meta_data['pose_path'].isnull()]
        videos_list = entries_without_pose_path.index.tolist()
        return videos_list

    def return_entries_with_processed_data(self, overwrite = False):
        if overwrite:
            return self.meta_data
        else:
            entries_w_processed = self.meta_data.loc[self.meta_data['processed_data_path'].isnull() == False]
        return entries_w_processed

    def return_entries_not_analyzed(self):
        entries_w_pose = self.meta_data.loc[self.meta_data['processed_data_path'].isnull() == False]
        entries_w_pose_not_analyzed = entries_w_pose.loc[entries_w_pose['analysis_path'].isnull()]
        return entries_w_pose_not_analyzed

    def update_meta_data(self, video_name, entry_type, entry):
        self.meta_data.loc[video_name, entry_type] = entry
        self.meta_data.to_csv(self.metafile_path) # save csv
        print(f'Meta data updated for [{entry_type}] of {video_name}.')

    def get_maze(self, ID):
        return int(self.query(ID, 'maze'))

    def query(self, ID, field):
        ''' get entry from meta table based on ID (video name) and field looking for'''
        if type(ID) == list:
            output = []
            for id in ID:
                output.append(self.meta_data.loc[id, field])
            return output
        else:
            return self.meta_data.loc[ID, field]

    def get_animal(self, ID):
        return self.query(ID, 'animal')

    def get_data_truncation(self, ID):
        truncate = self.query(ID, 'truncate')
        if pandas.isnull(truncate):
            return False
        else:
            return truncate

    def check_if_rewarding(self, ID, reward_field = 'reward_config'):
        reward_config = self.query(ID, reward_field)
        if pandas.isnull(reward_config) or reward_config == 'None': # empty field
            return False
        else:
            return True

    def get_reward_config_id_video(self, ID, field = 'reward_config'):
        config_id = self.query(ID, field)
        if pandas.isnull(config_id):
            raise Exception(f'* Error: missing reward_config entry for {ID}. Check metafile.')
        return config_id