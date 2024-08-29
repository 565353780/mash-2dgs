from mash_2dgs.Module.trainer import Trainer

def demo():
    source_path = '/home/chli/Dataset/BlenderNeRF/bunny/'
    source_path = '/home/chli/Dataset/NeRF/hotdog_train/'
    images = 'dense/images'

    trainer = Trainer(source_path, images)
    trainer.train(35000)
    return True
