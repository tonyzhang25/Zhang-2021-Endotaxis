from binarymaze_utils.meta_data_loader import MetaData
from binarymaze_utils.data_loader import DataLoader
import json

class QueryData():

    def __init__(self, working_dir):
        self.init_paths(working_dir)
        self.init_dataloaders()

    def init_paths(self, working_dir):
        self.working_dir = working_dir
        self.video_dir = f'{working_dir}/videos'
        self.pose_dir = f'{working_dir}/pose'
        self.reward_config_path = f'{working_dir}/reward_configs.json'
        self.processed_data_dir = f'{working_dir}/processed_data'
        self.LED_dir = f'{working_dir}/LED_luminance'
        self.metafile_path = f'{working_dir}/experiment_metafile.csv'

    def init_dataloaders(self):
        self.load_metadata()
        self.load_dataloader()

    def load_metadata(self):
        self.MetaData = MetaData(self.metafile_path)

    def load_dataloader(self):
        self.DataLoader = DataLoader(self.MetaData, self.processed_data_dir)

    def get_data(self, IDs, save_dir = None):
        if type(IDs) != list:
            IDs = [IDs]
        data = self.DataLoader.get_data(IDs, invert=True)
        # save data
        save_name = '-'.join(IDs)
        if save_dir is not None:
            json.dump(data, open(f'{save_dir}/{save_name}.json', 'w'))
        return data

