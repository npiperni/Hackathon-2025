import torch
import cv2
import numpy as np
import open3d as o3d
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Load the MiDaS model (pre-trained model for monocular depth estimation)
model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
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
    # Resize the image to make its dimensions divisible by 32
    image = resize_image(image)

    # Convert image to the format required by MiDaS
    input_image = transform(image).unsqueeze(0)

    # Run depth estimation
    with torch.no_grad():
        depth_map = model(input_image)

    # Normalize depth map
    depth_map = depth_map.squeeze().cpu().numpy()

    return depth_map


def process_frame(frame_rgb):
    # Convert the frame to a PIL image (OpenCV uses BGR by default, so convert it to RGB first)
    pil_image = Image.fromarray(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))

    # Estimate depth from the frame
    depth_map = estimate_depth(pil_image)

    # Convert the depth map to a 3D point cloud using Open3D (make sure depth is in meters)
    height, width = depth_map.shape
    fx = 525.0  # Focal length in x
    fy = 525.0  # Focal length in y
    cx = width / 2.0  # Optical center x
    cy = height / 2.0  # Optical center y

    # Generate point cloud
    points = []
    colors = []
    for v in range(height):
        for u in range(width):
            z = depth_map[v, u]
            if z == 0:
                continue  # Skip invalid depth values (if any)
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])

            # Assign color based on depth (you can adjust this to use RGB or other)
            color = [z / np.max(depth_map), 0, 1 - (z / np.max(depth_map))]  # Color gradient from blue to red
            colors.append(color)

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))  # Set color based on depth
    return pcd


def main(video_path, num_frames=0):
    cap = cv2.VideoCapture(video_path)

    if (num_frames == 0):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        total_frames = num_frames

    print(f"frames: {num_frames}")

    pcd_combined = o3d.geometry.PointCloud()  # Initialize an empty point cloud to accumulate frames
    frame_count = 0
    with tqdm(total=total_frames, desc="Processing frames") as pbar:

        while cap.isOpened() and frame_count < num_frames:
            ret, frame_rgb = cap.read()
            if not ret:
                break

            # Estimate depth and create point cloud
            pcd = process_frame(frame_rgb)

            # Combine the point clouds from each frame
            pcd_combined += pcd

            frame_count += 1  # Increment the frame count

        cap.release()  # Release the video capture object

    # After processing all frames, downsample the combined point cloud
    voxel_size = 0.1  # Adjust as needed
    pcd_combined_downsampled = pcd_combined.voxel_down_sample(voxel_size)

    # Visualize the final combined point cloud
    o3d.visualization.draw_geometries([pcd_combined_downsampled])


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    video_path = "test.mp4"  # Replace with your actual path
    main(video_path, num_frames=10)  # Process only the first 10 frames for debugging
