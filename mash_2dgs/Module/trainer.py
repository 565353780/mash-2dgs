import os
import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple, Union
from random import randint

import mash_cpp

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


from mash_2dgs.Method.time import getCurrentTime
from mash_2dgs.Module.logger import Logger

class Trainer(object):
    def __init__(self,
                 source_path: str,
                 ply_file_path: Union[str, None]=None,
                 op_func = OptimizationParams,
                 ) -> None:
        self.logger = Logger()
        self.save_result_folder_path = "auto"
        self.save_log_folder_path = "auto"
        self.initRecords()

        self.test_freq = 5000
        self.save_freq = 5000

        quiet = False
        ip = "127.0.0.1"
        port = 6009

        images = 'images'
        resolution = 2

        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = op_func(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args()

        args.source_path = source_path
        args.images = images
        args.resolution = resolution
        args.model_path = self.save_result_folder_path

        print("Optimizing " + args.model_path)
        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))

        self.dataset = lp.extract(args)
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

        safe_state(quiet)

        network_gui.init(ip, port)
        # torch.autograd.set_detect_anomaly(True)

        self.gaussians = GaussianModel(self.dataset.sh_degree)
        if os.path.exists(ply_file_path):
            self.gaussians.load_ply(ply_file_path)
        self.scene = Scene(self.dataset, self.gaussians)
        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return

    def initRecords(self) -> bool:
        self.save_file_idx = 0

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./output/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def renderImage(self, viewpoint_cam, scaling_modifer: float = 1.0) -> dict:
        return render(viewpoint_cam, self.gaussians, self.pipe, self.background, scaling_modifer)

    def trainStep(self, iteration: int, viewpoint_cam) -> Tuple[dict, dict]:
        self.gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        render_pkg = self.renderImage(viewpoint_cam)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        reg_loss = l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image, gt_image)
        rgb_loss = (1.0 - self.opt.lambda_dssim) * reg_loss + self.opt.lambda_dssim * ssim_loss

        lambda_normal = self.opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = self.opt.lambda_dist if iteration > 3000 else 0.0

        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - rend_normal * surf_normal)
        normal_loss = lambda_normal * normal_error.mean()

        rend_dist = render_pkg["rend_dist"]
        dist_loss = lambda_dist * rend_dist.mean()

        opacity_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        scaling_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        surface_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)

        # loss
        total_loss = rgb_loss + dist_loss + normal_loss + opacity_loss + scaling_loss + surface_loss

        total_loss.backward()

        loss_dict = {
            'reg': reg_loss.item(),
            'ssim': ssim_loss.item(),
            'rgb': rgb_loss.item(),
            'dist': dist_loss.item(),
            'normal': normal_loss.item(),
            'opacity': opacity_loss.item(),
            'scaling': scaling_loss.item(),
            'surface': surface_loss.item(),
            'total': total_loss.item(),
        }

        return render_pkg, loss_dict

    def trainStepV2(self, iteration: int, viewpoint_cam, surface_points: Union[torch.Tensor, None]=None) -> Tuple[dict, dict]:
        self.gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        render_pkg = self.renderImage(viewpoint_cam)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        reg_loss = l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image, gt_image)
        rgb_loss = (1.0 - self.opt.lambda_dssim) * reg_loss + self.opt.lambda_dssim * ssim_loss

        lambda_normal = self.opt.lambda_normal if iteration > 7000 else 0.0
        normal_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_normal > 0:
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_dot = torch.sum(rend_normal * surf_normal, dim=0)
            valid_dot_idxs = torch.where(normal_dot != 0)
            valid_normal_dot = torch.abs(normal_dot[valid_dot_idxs])
            normal_error = (1 - valid_normal_dot)
            normal_loss = lambda_normal * normal_error.mean()

        lambda_dist = self.opt.lambda_dist if iteration > 3000 else 0.0
        #FIXME: seems this dist not work
        lambda_dist = 0.0
        dist_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_dist > 0:
            rend_dist = render_pkg["rend_dist"]
            valid_dist_idxs = torch.where(rend_dist != 0)
            valid_rend_dist = rend_dist[valid_dist_idxs]
            dist_loss = lambda_dist * valid_rend_dist.mean()

        lambda_opacity = self.opt.lambda_opacity if hasattr(self.opt, 'lambda_opacity') else 1e-3
        opacity_loss = lambda_opacity * nn.MSELoss()(self.gaussians.get_opacity, torch.zeros_like(self.gaussians._opacity))
        # opacity_loss = 1e-3 * nn.MSELoss()(self.gaussians.get_opacity, torch.ones_like(self.gaussians._opacity))

        lambda_scaling = self.opt.lambda_scaling if hasattr(self.opt, 'lambda_scaling') else 1e-3
        scaling_loss = lambda_scaling * nn.MSELoss()(self.gaussians.get_scaling, torch.zeros_like(self.gaussians._scaling))

        lambda_surface = self.opt.lambda_surface if hasattr(self.opt, 'lambda_surface') else 1e-3
        surface_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if surface_points is not None:
            fit_loss, coverage_loss = mash_cpp.toChamferDistanceLoss(
                self.gaussians.get_xyz, surface_points
            )
            surface_loss = lambda_surface * (fit_loss + coverage_loss)

        # loss
        total_loss = rgb_loss + dist_loss + normal_loss + opacity_loss + scaling_loss + surface_loss

        total_loss.backward()

        loss_dict = {
            'reg': reg_loss.item(),
            'ssim': ssim_loss.item(),
            'rgb': rgb_loss.item(),
            'dist': dist_loss.item(),
            'normal': normal_loss.item(),
            'opacity': opacity_loss.item(),
            'scaling': scaling_loss.item(),
            'surface': surface_loss.item(),
            'total': total_loss.item(),
        }

        return render_pkg, loss_dict

    def trainStepWithSuperParams(self,
                                 iteration: int,
                                 viewpoint_cam,
                                 lambda_dssim: float = 0.2,
                                 lambda_normal: float = 0.01,
                                 lambda_dist: float = 100000.0,
                                 lambda_opacity: float = 0.001,
                                 lambda_scaling: float = 0.001,
                                 lambda_surface: float = 0.001,
                                 maximum_opacity: bool = False,
                                 surface_points: Union[torch.Tensor, None]=None,
                                 ) -> Tuple[dict, dict]:
        self.gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        render_pkg = self.renderImage(viewpoint_cam)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        reg_loss = l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image, gt_image)
        rgb_loss = (1.0 - lambda_dssim) * reg_loss + lambda_dssim * ssim_loss

        lambda_normal = lambda_normal if iteration > 7000 else 0.0
        normal_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_normal > 0:
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_dot = torch.sum(rend_normal * surf_normal, dim=0)
            valid_dot_idxs = torch.where(normal_dot != 0)
            valid_normal_dot = torch.abs(normal_dot[valid_dot_idxs])
            normal_error = (1 - valid_normal_dot)
            normal_loss = lambda_normal * normal_error.mean()

        lambda_dist = lambda_dist if iteration > 3000 else 0.0
        #FIXME: seems this dist not work
        lambda_dist = 0.0
        dist_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_dist > 0:
            rend_dist = render_pkg["rend_dist"]
            valid_dist_idxs = torch.where(rend_dist != 0)
            valid_rend_dist = rend_dist[valid_dist_idxs]
            dist_loss = lambda_dist * valid_rend_dist.mean()

        if maximum_opacity:
            opacity_loss = lambda_opacity * nn.MSELoss()(self.gaussians.get_opacity, torch.ones_like(self.gaussians._opacity))
        else:
            opacity_loss = lambda_opacity * nn.MSELoss()(self.gaussians.get_opacity, torch.zeros_like(self.gaussians._opacity))

        scaling_loss = lambda_scaling * nn.MSELoss()(self.gaussians.get_scaling, torch.zeros_like(self.gaussians._scaling))

        surface_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if surface_points is not None:
            fit_loss, coverage_loss = mash_cpp.toChamferDistanceLoss(
                self.gaussians.get_xyz, surface_points
            )
            surface_loss = lambda_surface * (fit_loss + coverage_loss)

        # loss
        total_loss = rgb_loss + dist_loss + normal_loss + opacity_loss + scaling_loss + surface_loss

        total_loss.backward()

        loss_dict = {
            'reg': reg_loss.item(),
            'ssim': ssim_loss.item(),
            'rgb': rgb_loss.item(),
            'dist': dist_loss.item(),
            'normal': normal_loss.item(),
            'opacity': opacity_loss.item(),
            'scaling': scaling_loss.item(),
            'surface': surface_loss.item(),
            'total': total_loss.item(),
        }

        return render_pkg, loss_dict

    @torch.no_grad
    def logStep(self, iteration: int, loss_dict: dict) -> bool:
        reg_loss = loss_dict['reg']
        ssim_loss = loss_dict['ssim']
        rgb_loss = loss_dict['rgb']
        dist_loss = loss_dict['dist']
        normal_loss = loss_dict['normal']
        opacity_loss = loss_dict['opacity']
        scaling_loss = loss_dict['scaling']
        surface_loss = loss_dict['surface']
        total_loss = loss_dict['total']

        # Log and save
        self.logger.addScalar('Loss/reg', reg_loss, iteration)
        self.logger.addScalar('Loss/ssim', ssim_loss, iteration)
        self.logger.addScalar('Loss/rgb', rgb_loss, iteration)
        self.logger.addScalar('Loss/dist', dist_loss, iteration)
        self.logger.addScalar('Loss/normal', normal_loss, iteration)
        self.logger.addScalar('Loss/opacity', opacity_loss, iteration)
        self.logger.addScalar('Loss/scaling', scaling_loss, iteration)
        self.logger.addScalar('Loss/surface', surface_loss, iteration)
        self.logger.addScalar('Loss/total', total_loss, iteration)

        self.logger.addScalar('Gaussian/total_points', self.scene.gaussians.get_xyz.shape[0], iteration)
        self.logger.addScalar('Gaussian/scale', torch.mean(self.gaussians.get_scaling).detach().clone().cpu().numpy(), iteration)
        self.logger.addScalar('Gaussian/opacity', torch.mean(self.gaussians.get_opacity).detach().clone().cpu().numpy(), iteration)
        self.logger.addScalar('Gaussian/split_num', self.gaussians.split_pts_num, iteration)
        self.logger.addScalar('Gaussian/clone_num', self.gaussians.clone_pts_num, iteration)
        self.logger.addScalar('Gaussian/prune_num', self.gaussians.prune_pts_num, iteration)

        # Report test and samples of training set
        if iteration % self.test_freq == 0:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras' : self.scene.getTestCameras()}, 
                                {'name': 'train', 'cameras' : [self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())] for idx in range(5, 30, 5)]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        render_pkg = self.renderImage(viewpoint)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if self.logger.isValid() and (idx < 5):
                            from utils.general_utils import colormap
                            depth = render_pkg["surf_depth"]
                            norm = depth.max()
                            depth = depth / norm
                            depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                            self.logger.summary_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                            self.logger.summary_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                            try:
                                rend_alpha = render_pkg['rend_alpha']
                                rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                                surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                                self.logger.summary_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                                self.logger.summary_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                                self.logger.summary_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                                rend_dist = render_pkg["rend_dist"]
                                rend_dist = colormap(rend_dist.cpu().numpy()[0])
                                self.logger.summary_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                            except:
                                pass

                            if iteration == 1:
                                self.logger.summary_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()

                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    self.logger.addScalar('Val/l1', l1_test, iteration)
                    self.logger.addScalar('Val/psnr', psnr_test, iteration)

            torch.cuda.empty_cache()
        return True

    def recordGrads(self, render_pkg: dict) -> bool:
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        return True

    def densifyStep(self) -> bool:
        size_threshold = 20
        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.opacity_cull, self.scene.cameras_extent, size_threshold)
        return True

    def resetOpacity(self) -> bool:
        self.gaussians.reset_opacity()
        return True

    def updateGSParams(self) -> bool:
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none = True)
        return True

    def saveScene(self, iteration: int) -> str:
        return self.scene.save(iteration)

    @torch.no_grad
    def renderForViewer(self, loss_dict: dict) -> bool:
        if network_gui.conn == None:
            network_gui.try_connect(self.dataset.render_items)
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                if custom_cam != None:
                    render_pkg = self.renderImage(custom_cam, scaling_modifer)
                    net_image = render_net_image(render_pkg, self.dataset.render_items, render_mode, custom_cam)
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                metrics_dict = {
                    "#": self.gaussians.get_opacity.shape[0],
                    "loss": loss_dict['rgb']
                    # Add more metrics as needed
                }
                # Send the data
                network_gui.send(net_image_bytes, self.dataset.source_path, metrics_dict)
                if do_training or not keep_alive:
                    break
            except Exception as e:
                # raise e
                network_gui.conn = None

        return True

    def train(self):
        viewpoint_stack = None

        progress_bar = tqdm(desc="Training progress")
        iteration = 0
        while True:
            iteration += 1

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            render_pkg, loss_dict = self.trainStep(iteration, viewpoint_cam)

            if iteration % 10 == 0:
                bar_loss_dict = {
                    "rgb": f"{loss_dict['rgb']:.{5}f}",
                    "distort": f"{loss_dict['dist']:.{5}f}",
                    "normal": f"{loss_dict['normal']:.{5}f}",
                    "Points": f"{len(self.gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(bar_loss_dict)
                progress_bar.update(10)

            self.logStep(iteration, loss_dict)

            if iteration % self.save_freq == 0:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                self.saveScene(iteration)

            # Densification
            if iteration < self.opt.densify_until_iter:
                self.recordGrads(render_pkg)
                if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                    self.densifyStep()

                if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.resetOpacity()

            self.updateGSParams()

            self.renderForViewer(loss_dict)
        return True
