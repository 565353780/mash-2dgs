from mash_2dgs.Module.joint_trainer import JointTrainer

def demo():
    source_path = '/home/chli/Dataset/BlenderNeRF/bunny/'

    joint_trainer = JointTrainer(source_path)
    joint_trainer.train()
    return True
