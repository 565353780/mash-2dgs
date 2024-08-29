from mash_2dgs.Module.joint_trainer import JointTrainer

def demo():
    source_path = '/home/chli/Dataset/BlenderNeRF/bunny/'
    source_path = '/home/chli/Dataset/NeRF/hotdog_train/'
    images = 'dense/images'
    ply_file_path = None

    joint_trainer = JointTrainer(source_path, images, ply_file_path)
    joint_trainer.train(35000)
    return True
