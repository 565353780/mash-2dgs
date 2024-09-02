from mash_2dgs.Config.custom_path import DATA_DICTS, TEST_DATA_NAME
from mash_2dgs.Module.joint_trainer import JointTrainer

def demo(anchor_num: int = 400):
    data_dict = DATA_DICTS[TEST_DATA_NAME]

    source_path = data_dict['source_path']
    images = data_dict['images']
    ply_file_path = data_dict['ply_file_path']
    # anchor_num = 400

    joint_trainer = JointTrainer(source_path, images, ply_file_path, anchor_num)
    joint_trainer.train(35000)
    return True
