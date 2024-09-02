import os
from tqdm import tqdm
from shutil import copyfile

from mash_2dgs.Metric.chamfer import toChamferDistanceFromMeshFile

def measureResults(
    gt_mesh_file_path: str,
    mesh_file_path_dict: dict,
    sample_point_num: int,
    overwrite: bool = False,
) -> bool:
    if 'fps' in mesh_file_path_dict.keys():
        copy_gt_mesh_file_path = mesh_file_path_dict['fps']
        if not os.path.exists(copy_gt_mesh_file_path):
            copyfile(gt_mesh_file_path, copy_gt_mesh_file_path)

    cd_dict = {}

    print('start calculate chamfer distances...')
    for key, mesh_file_path in tqdm(mesh_file_path_dict.items()):
        cd_dict[key] = toChamferDistanceFromMeshFile(mesh_file_path, gt_mesh_file_path, sample_point_num, overwrite)

    for key, cd in cd_dict.items():
        print(key, ':', cd)
    return True

if __name__ == "__main__":
    bunny_gt_mesh_file_path = './output/bunny.ply'
    bunny_mesh_file_path_dict = {
        'bunny_2dgs': './output/bunny_2dgs/train/ours_35000/fuse_post.ply',
        'bunny_2dgs-v2': './output/bunny_2dgs-v2/train/ours_35000/fuse_post.ply',
        'bunny_mash-100anc': './output/bunny_mash-100anc/train/ours_35000/fuse_post.ply',
        'bunny_mash-400anc': './output/bunny_mash-400anc/train/ours_35000/fuse_post.ply',
        'bunny_opacity0': './output/bunny_opacity0/train/ours_35000/fuse_post.ply',
        'bunny_opacity0_mash-100anc': './output/bunny_opacity0_mash-100anc/train/ours_35000/fuse_post.ply',
        'bunny_opacity0_mash-200anc': './output/bunny_opacity0_mash-200anc/train/ours_35000/fuse_post.ply',
        'bunny_opacity0_mash-400anc': './output/bunny_opacity0_mash-400anc/train/ours_35000/fuse_post.ply',
        'fps': './output/bunny_copy.ply',
    }
    hotdog_mesh_file_path_dict = {
        'hotdog_opacity0': './output/hotdog_opacity0/train/ours_35000/fuse_post.ply',
        'hotdog_mash': './output/hotdog_mash/train/ours_35000/fuse_post.ply',
    }
    sample_point_num = 40000
    overwrite = False

    measureResults(bunny_gt_mesh_file_path, bunny_mesh_file_path_dict, sample_point_num, overwrite)
