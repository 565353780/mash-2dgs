import os
import torch
from tqdm import tqdm
from random import randint

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams


from mash_2dgs.Method.time import getCurrentTime
from mash_2dgs.Module.mash_refiner import MashRefiner
from mash_2dgs.Module.logger import Logger

class JointTrainer(object):
    def __init__(self) -> None:
        self.save_result_folder_path = "auto"
        self.save_log_folder_path = "auto"

        self.test_iterations = list(range(0, 30001, 5000))
        self.save_iterations = list(range(0, 30001, 5000))

        quiet = False
        ip = "127.0.0.1"
        port = 6009

        source_path = '/home/chli/Dataset/BlenderNeRF/bunny/'
        images = 'train'
        resolution = 2

        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args()

        args.source_path = source_path
        args.images = images
        args.resolution = resolution

        print("Optimizing " + args.model_path)

        self.dataset = lp.extract(args)
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

        safe_state(quiet)

        network_gui.init(ip, port)
        # torch.autograd.set_detect_anomaly(True)

        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = Scene(self.dataset, self.gaussians)
        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.mash_refiner = MashRefiner()
        self.logger = Logger()

        self.initRecords()
        return

    def initRecords(self) -> bool:
        self.save_file_idx = 0

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def trainStep(self) -> bool:
        return True

    def train(self):
        first_iter = 0

        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        ema_dist_for_log = 0.0
        ema_normal_for_log = 0.0

        progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, self.opt.iterations + 1):

            iter_start.record()

            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            # regularization
            lambda_normal = self.opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = self.opt.lambda_dist if iteration > 3000 else 0.0
            # lambda_normal = opt.lambda_normal
            # lambda_dist = opt.lambda_dist

            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            if iteration > self.opt.densify_until_iter:
                opacity_loss = 1e-3 * torch.nn.MSELoss()(self.gaussians.get_opacity, torch.ones_like(self.gaussians._opacity))
            else:
                opacity_loss = 1e-3 * torch.nn.MSELoss()(self.gaussians.get_opacity, torch.zeros_like(self.gaussians._opacity))

            # loss
            total_loss = loss + dist_loss + normal_loss + opacity_loss

            total_loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
                ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "distort": f"{ema_dist_for_log:.{5}f}",
                        "normal": f"{ema_normal_for_log:.{5}f}",
                        "Points": f"{len(self.gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)

                    progress_bar.update(10)
                if iteration == self.opt.iterations:
                    progress_bar.close()

                # Log and save
                self.logger.addScalar('Loss/dist', ema_dist_for_log, iteration)
                self.logger.addScalar('Loss/normal', ema_normal_for_log, iteration)

                self.logger.addScalar('Gaussian/scale', torch.mean(self.gaussians.get_scaling).detach().clone().cpu().numpy(), iteration)
                self.logger.addScalar('Gaussian/opacity', torch.mean(self.gaussians.get_opacity).detach().clone().cpu().numpy(), iteration)
                self.logger.addScalar('Gaussian/split_num', self.gaussians.split_pts_num, iteration)
                self.logger.addScalar('Gaussian/clone_num', self.gaussians.clone_pts_num, iteration)
                self.logger.addScalar('Gaussian/prune_num', self.gaussians.prune_pts_num, iteration)

                self.training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), render)
                if (iteration in self.save_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    self.scene.save(iteration)

                # Densification
                if iteration < self.opt.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.opacity_cull, self.scene.cameras_extent, size_threshold)

                    if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)

            with torch.no_grad():
                if network_gui.conn == None:
                    network_gui.try_connect(self.dataset.render_items)
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                        if custom_cam != None:
                            render_pkg = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)
                            net_image = render_net_image(render_pkg, self.dataset.render_items, render_mode, custom_cam)
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        metrics_dict = {
                            "#": self.gaussians.get_opacity.shape[0],
                            "loss": ema_loss_for_log
                            # Add more metrics as needed
                        }
                        # Send the data
                        network_gui.send(net_image_bytes, self.dataset.source_path, metrics_dict)
                        if do_training and ((iteration < int(self.opt.iterations)) or not keep_alive):
                            break
                    except Exception as e:
                        # raise e
                        network_gui.conn = None
        return True

    @torch.no_grad()
    def training_report(self, iteration, Ll1, loss, l1_loss, elapsed, renderFunc):
        self.logger.addScalar('Loss/reg', Ll1.item(), iteration)
        self.logger.addScalar('Loss/total', loss.item(), iteration)
        self.logger.addScalar('Gaussian/iter_time', elapsed, iteration)
        self.logger.addScalar('Gaussian/total_points', self.scene.gaussians.get_xyz.shape[0], iteration)

        # Report test and samples of training set
        if iteration in self.test_iterations:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras' : self.scene.getTestCameras()}, 
                                {'name': 'train', 'cameras' : [self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())] for idx in range(5, 30, 5)]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        render_pkg = renderFunc(viewpoint, self.scene.gaussians, self.pipe, self.background)
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

                            if iteration == self.test_iterations[0]:
                                self.logger.summary_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()

                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                    if self.logger:
                        self.logger.addScalar('Val/l1', l1_test, iteration)
                        self.logger.addScalar('Val/psnr', psnr_test, iteration)

            torch.cuda.empty_cache()
        return True
