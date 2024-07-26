import open3d as o3d

from mash_2dgs.Model.mash_gs import MashGS

def renderMashGS(mash_gs: MashGS) -> bool:
    sample_mesh = mash_gs.mash.toSampleMesh().toO3DMesh()
    sample_ellipses = mash_gs.mash.toSimpleSampleO3DEllipses()

    o3d.visualization.draw_geometries(sample_ellipses + [sample_mesh])
    return True
