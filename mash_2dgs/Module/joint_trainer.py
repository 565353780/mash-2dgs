from tqdm import tqdm
from random import randint

from mash_2dgs.Config.joint_optimization_params import JointOptimizationParams
from mash_2dgs.Module.trainer import Trainer
from mash_2dgs.Module.mash_refiner import MashRefiner

class JointTrainer(object):
    def __init__(self,
                 source_path: str,
                 ) -> None:
        self.trainer = Trainer(source_path, JointOptimizationParams)

        anchor_num = 400
        mask_degree_max = 3
        sh_degree_max = 2
        mask_boundary_sample_num = 90
        sample_polar_num = 1000
        sample_point_scale = 0.8
        self.mash_refiner = MashRefiner(
            anchor_num,
            mask_degree_max,
            sh_degree_max,
            mask_boundary_sample_num,
            sample_polar_num,
            sample_point_scale,
        )

        self.save_freq = 5000

        self.viewpoint_stack = None

        self.iteration = 1
        return

    def getImage(self):
        # Pick a random Camera
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
        return viewpoint_cam

    def trainStep(self,
                  lambda_dssim: float = 0.2,
                  lambda_normal: float = 0.01,
                  lambda_dist: float = 100000.0,
                  lambda_opacity: float = 0.001,
                  lambda_scaling: float = 0.001,
                  maximum_opacity: bool = False,
                  ):
        return self.trainer.trainStepWithSuperParams(
            self.iteration,
            self.getImage(),
            lambda_dssim,
            lambda_normal,
            lambda_dist,
            lambda_opacity,
            lambda_scaling,
            maximum_opacity,
        )

    def logStep(self, loss_dict: dict) -> bool:
        self.trainer.logStep(self.iteration, loss_dict)
        return True

    def updateProgressBar(self, progress_bar, loss_dict: dict) -> bool:
        if self.iteration % 10 != 0:
            return True

        bar_loss_dict = {
            "rgb": f"{loss_dict['rgb']:.{5}f}",
            "distort": f"{loss_dict['dist']:.{5}f}",
            "normal": f"{loss_dict['normal']:.{5}f}",
            "Points": f"{len(self.trainer.gaussians.get_xyz)}"
        }
        progress_bar.set_postfix(bar_loss_dict)
        progress_bar.update(10)
        return True

    def saveScene(self) -> bool:
        if self.iteration % self.save_freq != 0:
            return True

        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
        self.trainer.saveScene(self.iteration)
        return True

    def densifyStep(self, render_pkg) -> bool:
        if self.iteration < self.trainer.opt.densify_until_iter or self.trainer.opt.densify_until_iter < 0:
            self.trainer.recordGrads(render_pkg)
            if self.iteration > self.trainer.opt.densify_from_iter and self.iteration % self.trainer.opt.densification_interval == 0:
                self.trainer.densifyStep()
        return True

    def resetOpacity(self) -> bool:
        if self.iteration % self.trainer.opt.opacity_reset_interval == 0 or (self.trainer.dataset.white_background and self.iteration == self.trainer.opt.densify_from_iter):
            self.trainer.resetOpacity()
        return True

    def preProcessGS(self, progress_bar, loss_dict: dict) -> bool:
        self.updateProgressBar(progress_bar, loss_dict)
        self.logStep(loss_dict)
        self.saveScene()
        return True

    def postProcessGS(self, loss_dict: dict) -> bool:
        self.trainer.updateGSParams()
        self.trainer.renderForViewer(self.iteration, loss_dict)
        self.iteration += 1
        return True

    def train(self) -> bool:
        progress_bar = tqdm(desc="Joint training progress")
        while True:
            render_pkg, loss_dict = self.trainStep()

            self.preProcessGS(progress_bar, loss_dict)

            self.densifyStep(render_pkg)

            self.resetOpacity()

            self.postProcessGS(loss_dict)
        return True
