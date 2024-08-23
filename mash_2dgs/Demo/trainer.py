from mash_2dgs.Module.trainer import Trainer

def demo():
    source_path = '/home/chli/Dataset/BlenderNeRF/bunny/'

    trainer = Trainer(source_path)
    trainer.train()
    return True
