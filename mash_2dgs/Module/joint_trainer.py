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
        return

    def train(self) -> bool:
        viewpoint_stack = None

        progress_bar = tqdm(desc="Joint training progress")
        iteration = 0
        while True:
            iteration += 1

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            render_pkg, loss_dict = self.trainer.trainStep(iteration, viewpoint_cam)

            if iteration % 10 == 0:
                bar_loss_dict = {
                    "rgb": f"{loss_dict['rgb']:.{5}f}",
                    "distort": f"{loss_dict['dist']:.{5}f}",
                    "normal": f"{loss_dict['normal']:.{5}f}",
                    "Points": f"{len(self.trainer.gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(bar_loss_dict)
                progress_bar.update(10)

            self.trainer.logStep(iteration, loss_dict)

            if iteration % self.save_freq == 0:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                self.trainer.saveScene(iteration)

            # Densification
            if iteration < self.trainer.opt.densify_until_iter or self.trainer.opt.densify_until_iter < 0:
                self.trainer.recordGrads(render_pkg)
                if iteration > self.trainer.opt.densify_from_iter and iteration % self.trainer.opt.densification_interval == 0:
                    self.trainer.densifyStep()

                if iteration % self.trainer.opt.opacity_reset_interval == 0 or (self.trainer.dataset.white_background and iteration == self.trainer.opt.densify_from_iter):
                    self.trainer.resetOpacity()

            self.trainer.updateGSParams()

            self.trainer.renderForViewer(iteration, loss_dict)
        return True
