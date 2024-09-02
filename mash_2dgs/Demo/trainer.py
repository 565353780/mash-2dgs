from mash_2dgs.Config.custom_path import DATA_DICTS, TEST_DATA_NAME
from mash_2dgs.Module.trainer import Trainer

def demo():
    data_dict = DATA_DICTS[TEST_DATA_NAME]

    source_path = data_dict['source_path']
    images = data_dict['images']

    trainer = Trainer(source_path, images)
    trainer.train(35000)
    return True
