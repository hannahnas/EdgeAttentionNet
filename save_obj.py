import open3d as o3d
import numpy as np

def write_obj(color, depth, mask, file_name):
    color = o3d.geometry.Image(np.ascontiguousarray(color).astype(np.uint8))
    # To do: add mask as alpha channel
    depth = o3d.geometry.Image(np.ascontiguousarray(depth))
    cam = o3d.camera.PinholeCameraIntrinsic()
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                   [0, 0, -1, 0], [0, 0, 0, 1]])

    cam.set_intrinsics(
        width=256,
        height=256,
        fx=147.22,
        fy=147.80,
        cx=128.5,
        cy=128.5
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=cam)
    # alpha??
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    o3d.io.write_triangle_mesh(file_name, mesh)


if __name__ == '__main__':
    pass

    