import open3d as o3d
import numpy as np
from InpaintDataset import InpaintDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def write_obj(color, depth, mask, file_name):
    color = color * 255
    # mask[~mask.bool()] = 1
    # mask[mask.bool()] = 0.5
    # rgba = np.concatenate((color, mask), axis=2)

    color = o3d.geometry.Image(np.ascontiguousarray(color).astype(np.uint8))

    # To do: add mask as alpha channel
    depth = o3d.geometry.Image(np.ascontiguousarray(depth))
    cam = o3d.camera.PinholeCameraIntrinsic()
    
    cam.set_intrinsics(
        width=208,
        height=208,
        fx=147.22,
        fy=147.80,
        cx=104.5,
        cy=104.5
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=cam)

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                   [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.io.write_point_cloud('test.xyzrgb', pcd)
    # o3d.visualization.draw_geometries([pcd])
    # Up until here works

    # o3d.geometry.PointCloud.estimate_normals(pcd)

    # print('run Poisson surface reconstruction')
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    #     poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #         pcd, depth=9, width=0, scale=1.1, linear_fit=False)
    # # print(mesh)
    # o3d.visualization.draw_geometries([poisson_mesh])

    # print('run BPA')
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 3 * avg_dist
    # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    # # dec_mesh = mesh.simplify_quadric_decimation(100000)
    # o3d.visualization.draw_geometries([bpa_mesh])
    
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.010)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    # o3d.io.write_triangle_mesh(file_name, mesh)


if __name__ == '__main__':
    train_set = InpaintDataset(split = 'train', samples=2)
    loader = DataLoader(train_set, batch_size=1, shuffle=True,
                            drop_last=True, pin_memory=True, num_workers=1)

    for batch in loader:
        rgb = batch['rgb'][0].permute(1, 2, 0)
        depth = batch['depth'][0].permute(1, 2, 0)
        mask = batch['mask'][0].permute(1, 2, 0)

        write_obj(rgb, depth, mask, 'test')
        break


    