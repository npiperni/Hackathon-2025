import open3d as o3d
import numpy as np
import cv2


# Function to load a single RGB frame and depth frame from a video
def load_rgbd_from_video(frame_rgb, frame_depth, depth_scale=1000.0):
    rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Depth frame conversion (if using an RGB-D camera, depth is usually in meters, but might need scaling)
    depth = frame_depth / depth_scale  # Adjust depending on your video format

    # Create Open3D RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        depth_trunc=3.0,  # Optional: Set depth truncation if needed
        convert_rgb_to_intensity=False
    )
    return rgbd_image


# Function to generate point cloud from RGBD image
def rgbd_to_point_cloud(rgbd_image, intrinsics):
    # Create a pinhole camera intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics["width"],
        height=intrinsics["height"],
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
        cx=intrinsics["cx"],
        cy=intrinsics["cy"]
    )

    # Generate point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )

    # Estimate normals for the point cloud
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    return pcd


# Function to process point cloud (downsampling, mesh generation)
def process_point_cloud(pcd):
    voxel_size = 0.05  # Adjust for desired detail level
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)

    # Surface reconstruction using Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )

    # Remove low-density vertices (optional)
    vertices_to_remove = densities < np.percentile(densities, 10)  # Remove bottom 10% density
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


# Visualization function for point cloud and mesh
def visualize(pcd, mesh):
    o3d.visualization.draw_geometries([pcd, mesh])


# Main function for video 3D reconstruction
def main(video_path, num_frames=10):
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video dimensions (width, height) from the video file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Camera intrinsic parameters (adjust for your setup)
    intrinsics = {
        "width": width,
        "height": height,
        "fx": 525.0,  # Focal length in x-axis (adjust according to your camera)
        "fy": 525.0,  # Focal length in y-axis (adjust according to your camera)
        "cx": width / 2.0,  # Optical center in x-axis (assuming the center of the image)
        "cy": height / 2.0  # Optical center in y-axis (assuming the center of the image)
    }

    # Loop through the first `num_frames` in the video
    pcd_combined = o3d.geometry.PointCloud()
    frame_count = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame_rgb = cap.read()
        if not ret:
            break

        # Assuming you also have a depth video stream (or a method for extracting depth from RGB)
        # For now, let's assume the depth map is the same size as the RGB frame and is a dummy array.
        frame_depth = np.random.uniform(0, 3, (frame_rgb.shape[0], frame_rgb.shape[1])).astype(np.float32)

        # Load the RGB-D image
        rgbd_image = load_rgbd_from_video(frame_rgb, frame_depth)

        # Generate point cloud
        pcd = rgbd_to_point_cloud(rgbd_image, intrinsics)

        # Combine point clouds (you can also perform registration or frame alignment here)
        pcd_combined += pcd

        frame_count += 1

    cap.release()  # Close the video file

    # Process the combined point cloud (downsampling, reconstruction)
    mesh = process_point_cloud(pcd_combined)

    # Visualize the final point cloud and mesh
    visualize(pcd_combined, mesh)


if __name__ == "__main__":
    video_path = "test.mp4"  # Set your video path here
    main(video_path, num_frames=1)  # Use only the first 10 frames for debugging
