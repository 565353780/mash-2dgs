from mash_2dgs.Module.trainer import Trainer
from mash_2dgs.Module.mash_refiner import MashRefiner


class JointTrainer(object):
    def __init__(self,
                 source_path: str,
                 ) -> None:
        self.trainer = Trainer(source_path)
        self.mash_refiner = MashRefiner()
        return

    def train(self) -> bool:
        return True
