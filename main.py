import torch
import cv2
import numpy as np
import open3d as o3d
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import concurrent.futures

# Load the MiDaS model (pre-trained model for monocular depth estimation)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
model = model.to(device)
model.eval()

# Transform for the model (standardization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def resize_image(image):
    """Resize image to be divisible by 32."""
    width, height = image.size
    new_width = (width // 32) * 32
    new_height = (height // 32) * 32
    return image.resize((new_width, new_height), Image.BILINEAR)


def estimate_depth(image):
    """Estimate depth map using MiDaS model."""
    image = resize_image(image)
    input_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = model(input_image)
    return depth_map.squeeze().cpu().numpy()


def process_frame(frame_rgb):
    """Process a frame to generate a point cloud."""
    pil_image = Image.fromarray(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
    depth_map = estimate_depth(pil_image)

    # Convert depth map to point cloud
    height, width = depth_map.shape
    fx = 958.0  # Focal length
    fy = 958.0  # Focal length
    cx = width / 2.0  # Optical center
    cy = height / 2.0  # Optical center

    points = []
    colors = []
    for v in range(height):
        for u in range(width):
            z = depth_map[v, u]
            if z == 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            r, g, b = frame_rgb[v, u]
            colors.append([r / 255.0, g / 255.0, b / 255.0])

    return points, colors


def process_batch(batch_frames):
    """Process a batch of frames and combine the results."""
    all_points = []
    all_colors = []

    for frame_rgb in batch_frames:
        points, colors = process_frame(frame_rgb)
        all_points.extend(points)
        all_colors.extend(colors)

    return all_points, all_colors


def register_point_clouds(pcd1, pcd2):
    """Register two point clouds using ICP (Iterative Closest Point)."""
    # Perform point cloud registration using ICP
    voxel_size = 0.05
    threshold = 0.1  # Maximum correspondence distance
    reg_icp = o3d.registration.registration_icp(
        pcd1, pcd2, threshold, np.eye(4),
        o3d.registration.TransformationEstimationPointToPoint()
    )
    return reg_icp.transformation


def main(video_path, num_frames=0, batch_size=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = num_frames if num_frames > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Frames: {total_frames}")

    all_points = []
    all_colors = []
    pcd_prev = None  # Keep track of previous point cloud for registration

    # Process the frames in batches
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        batch_frames = []
        frame_count = 0

        while cap.isOpened() and frame_count < total_frames:
            ret, frame_rgb = cap.read()
            if not ret:
                break

            batch_frames.append(frame_rgb)
            frame_count += 1

            # Process and accumulate points/colors once we reach the batch size
            if len(batch_frames) == batch_size or frame_count == total_frames:
                points, colors = process_batch(batch_frames)
                batch_points = np.array(points)
                batch_colors = np.array(colors)

                pcd_current = o3d.geometry.PointCloud()
                pcd_current.points = o3d.utility.Vector3dVector(batch_points)
                pcd_current.colors = o3d.utility.Vector3dVector(batch_colors)

                if pcd_prev is not None:
                    # Register current point cloud to previous one
                    transformation = register_point_clouds(pcd_prev, pcd_current)
                    pcd_current.transform(transformation)

                # Append the current point cloud to the combined list
                all_points.extend(np.asarray(pcd_current.points))
                all_colors.extend(np.asarray(pcd_current.colors))

                pcd_prev = pcd_current  # Update the previous point cloud
                batch_frames = []  # Clear the batch
                pbar.update(batch_size)  # Update progress bar for the batch size

        cap.release()  # Release the video capture object

    # After processing all frames, create a final point cloud
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined.points = o3d.utility.Vector3dVector(np.array(all_points))
    pcd_combined.colors = o3d.utility.Vector3dVector(np.array(all_colors))

    # Downsample the combined point cloud
    voxel_size = 0.5  # Adjust as needed
    pcd_combined_downsampled = pcd_combined.voxel_down_sample(voxel_size)

    # Visualize the final combined point cloud
    o3d.visualization.draw_geometries([pcd_combined_downsampled])


if __name__ == "__main__":
    print(f"Using device: {device}")
    video_path = "test.mp4"  # Replace with your actual path
    main(video_path, num_frames=0, batch_size=10)  # Process in batches of 10 frames
