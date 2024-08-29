from mash_2dgs.Module.joint_trainer import JointTrainer

def demo():
    source_path = '/home/chli/Dataset/BlenderNeRF/bunny/'
    ply_file_path = './output/20240827_18:01:11/point_cloud/iteration_35000/point_cloud.ply'

    joint_trainer = JointTrainer(source_path, ply_file_path)
    joint_trainer.train()
    return True
