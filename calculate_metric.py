from mash_2dgs.Metric.chamfer import toChamferDistanceFromMeshFile

if __name__ == "__main__":
    gt_mesh_file_path = './output/bunny.ply'
    mesh_file_path_1 = './output/bunny_2DGS/train/ours_35000/fuse_post.ply'
    mesh_file_path_2 = './output/bunny_mash-2dgs/train/ours_35000/fuse_post.ply'
    sample_point_num = 40000
    overwrite = False

    cd1 = toChamferDistanceFromMeshFile(mesh_file_path_1, gt_mesh_file_path, sample_point_num, overwrite)
    cd2 = toChamferDistanceFromMeshFile(mesh_file_path_2, gt_mesh_file_path, sample_point_num, overwrite)
    print('bunny_2DGS :', cd1)
    print('bunny_mash-2dgs :', cd2)
