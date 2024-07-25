#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
sys.path.append('../ma-sh/')

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from plyfile import PlyData, PlyElement

from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation

from ma_sh.Config.weights import W0
from ma_sh.Config.constant import EPSILON
from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Method.pcd import getPointCloud, downSample

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

class MashGS(object):
    def __init__(self, sh_degree: int, anchor_num: int=400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        use_inv: bool = True,
        idx_dtype=torch.int64,
        dtype=torch.float32,
        device: str = "cuda"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.anchor_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.mash = SimpleMash(anchor_num, mask_degree_max, sh_degree_max, sample_phi_num, sample_theta_num, use_inv, idx_dtype, dtype, device)
        self.mash.setGradState(True)

        self.surface_dist = 1.0
        return

    def updateGSParams(self) -> bool:
        centers, axis_lengths, rotate_matrixs = self.mash.toSimpleSampleEllipses()

        self._xyz = centers
        self._scaling = axis_lengths
        self._rotation = matrix_to_quaternion(rotate_matrixs)
        return True

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid


    def capture(self):
        assert self.optimizer is not None

        return (
            self.active_sh_degree,
            self.mash.mask_params,
            self.mash.sh_params,
            self.mash.rotate_vectors,
            self.mash.positions,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self.max_radii2D,
            self.anchor_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        assert self.optimizer is not None

        (self.active_sh_degree, 
            mask_params,
            sh_params,
            rotate_vectors,
            positions,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        anchor_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.mash.loadParams(mask_params, sh_params, rotate_vectors, positions)
        self.training_setup(training_args)
        self.anchor_gradient_accum = anchor_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def initMashFromPcd(self, pcd: BasicPointCloud) -> np.ndarray:
        anchor_pcd = getPointCloud(pcd.points, colors=pcd.colors)
        anchor_pcd.estimate_normals()

        if pcd.points.shape[0] <= self.mash.anchor_num:
            self.mash.anchor_num = pcd.points.shape[0]
            self.mash.reset()
        else:
            anchor_pcd = downSample(anchor_pcd, self.mash.anchor_num)

        if anchor_pcd is None:
            print("[ERROR][MashGS::initMashFromPcd]")
            print("\t downSample failed!")
            return np.ndarray([])

        sample_pts = np.asarray(anchor_pcd.points)
        sample_normals = np.asarray(anchor_pcd.normals)
        anchor_colors = np.asarray(anchor_pcd.colors)

        sh_params = torch.ones_like(self.mash.sh_params) * EPSILON
        sh_params[:, 0] = self.surface_dist / W0[0]

        self.mash.loadParams(
            sh_params=sh_params,
            positions=sample_pts + self.surface_dist * sample_normals,
            face_forward_vectors=-sample_normals,
        )

        self.updateGSParams()

        single_anchor_gs_num = int(self._xyz.shape[0] / self.mash.anchor_num)

        full_colors = anchor_colors.repeat(single_anchor_gs_num, 0)
        return full_colors

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        full_colors = self.initMashFromPcd(pcd)

        if full_colors.shape[0] == 0:
            print("[ERROR][MashGS::create_from_pcd]")
            print("\t initMashFromPcd failed!")
            return

        self.spatial_lr_scale = spatial_lr_scale
        fused_color = RGB2SH(torch.tensor(full_colors).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of anchors at initialisation : ", self.mash.anchor_num)
        print("Number of points at initialisation : ", self._xyz.shape[0])

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda"))

        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.anchor_gradient_accum = torch.zeros((self.mash.anchor_num, 1), device="cuda")
        self.denom = torch.zeros((self.mash.anchor_num, 1), device="cuda")

        l = [
            {'params': [self.mash.mask_params], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "mask_params"},
            {'params': [self.mash.sh_params], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "sh_params"},
            {'params': [self.mash.rotate_vectors], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "rotate_vectors"},
            {'params': [self.mash.positions], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "positions"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_anchors(self, mask):
        valid_anchors_mask = ~mask

        single_anchor_gs_num = int(self._xyz.shape[0] / self.mash.anchor_num)
        valid_points_mask = valid_anchors_mask.unsqueeze(1).repeat(single_anchor_gs_num).reshape(-1)

        optimizable_tensors = self._prune_optimizer(valid_anchors_mask)

        self.mash.loadParams(
            optimizable_tensors["mask_params"],
            optimizable_tensors["sh_params"],
            optimizable_tensors["rotate_vectors"],
            optimizable_tensors["positions"]
        )
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]

        self.anchor_gradient_accum = self.anchor_gradient_accum[valid_anchors_mask]

        self.denom = self.denom[valid_anchors_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_mask_params, new_sh_params, new_rotate_vectors, new_positions, new_features_dc, new_features_rest, new_opacities):
        d = {"mask_params": new_mask_params,
             "sh_params": new_sh_params,
             "rotate_vectors": new_rotate_vectors,
             "positions": new_positions,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.mash.loadParams(
            optimizable_tensors["mask_params"],
            optimizable_tensors["sh_params"],
            optimizable_tensors["rotate_vectors"],
            optimizable_tensors["positions"]
        )
        self.updateGSParams()

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]

        self.anchor_gradient_accum = torch.zeros((self.mash.anchor_num, 1), device="cuda")
        self.denom = torch.zeros((self.mash.anchor_num, 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((self.mash.anchor_num), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_anchor_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_anchor_mask = torch.logical_and(selected_anchor_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        single_anchor_gs_num = int(self._xyz.shape[0] / self.mash.anchor_num)
        selected_point_mask = selected_anchor_mask.unsqueeze(1).repeat(single_anchor_gs_num).reshape(-1)

        stds = self.get_scaling[selected_anchor_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_anchor_mask]).repeat(N,1,1)

        new_mask_params = self.mash.mask_params[selected_anchor_mask].repeat(N,1) / (0.8*N)
        new_sh_params = self.mash.sh_params[selected_anchor_mask].repeat(N,1)
        new_rotate_vectors = self.mash.rotate_vectors[selected_anchor_mask].repeat(N,1)
        new_positions = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.mash.positions[selected_anchor_mask].repeat(N, 1)

        new_features_dc = self._features_dc[selected_point_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_point_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_point_mask].repeat(N,1)

        self.densification_postfix(new_mask_params, new_sh_params, new_rotate_vectors, new_positions, new_features_dc, new_features_rest, new_opacity)

        prune_filter = torch.cat((selected_anchor_mask, torch.zeros(N * int(selected_anchor_mask.sum()), device="cuda", dtype=torch.bool)))
        self.prune_anchors(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_anchor_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_anchor_mask = torch.logical_and(selected_anchor_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        single_anchor_gs_num = int(self._xyz.shape[0] / self.mash.anchor_num)
        selected_point_mask = selected_anchor_mask.unsqueeze(1).repeat(single_anchor_gs_num).reshape(-1)

        new_mask_params = self.mash.mask_params[selected_anchor_mask]
        new_sh_params = self.mash.sh_params[selected_anchor_mask]
        new_rotate_vectors = self.mash.rotate_vectors[selected_anchor_mask]
        new_positions = self.mash.positions[selected_anchor_mask]

        new_features_dc = self._features_dc[selected_point_mask]
        new_features_rest = self._features_rest[selected_point_mask]
        new_opacities = self._opacity[selected_point_mask]

        self.densification_postfix(new_mask_params, new_sh_params, new_rotate_vectors, new_positions, new_features_dc, new_features_rest, new_opacities)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.anchor_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_point_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_point_mask = torch.logical_or(torch.logical_or(prune_point_mask, big_points_vs), big_points_ws)

        single_anchor_gs_num = int(self._xyz.shape[0] / self.mash.anchor_num)
        prune_anchor_point_mask = prune_point_mask.reshape(single_anchor_gs_num, -1)
        prune_anchor_mask = torch.sum(prune_anchor_point_mask, 1)
        print(prune_anchor_mask)
        exit()
        self.prune_anchors(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        anchor_point_update_filter = update_filter.reshape(self.mash.anchor_num, -1)
        anchor_update_filter = torch.any(anchor_point_update_filter, dim=1)

        anchor_point_grad = torch.norm(viewspace_point_tensor.grad, dim=-1, keepdim=True).reshape(self.mash.anchor_num, -1)
        anchor_grad = torch.sum(anchor_point_grad, dim=1).unsqueeze(1)

        self.anchor_gradient_accum[anchor_update_filter] += anchor_grad[anchor_update_filter]
        self.denom[anchor_update_filter] += 1
