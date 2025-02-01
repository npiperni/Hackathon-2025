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
    fx = 525.0  # Focal length
    fy = 525.0  # Focal length
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

def main(video_path, num_frames=0):
    cap = cv2.VideoCapture(video_path)
    total_frames = num_frames if num_frames > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Frames: {total_frames}")

    all_points = []
    all_colors = []

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor for parallel processing
            futures = []
            for _ in range(total_frames):
                ret, frame_rgb = cap.read()
                if not ret:
                    break
                futures.append(executor.submit(process_frame, frame_rgb))

            cap.release()

            for future in concurrent.futures.as_completed(futures):
                points, colors = future.result()
                all_points.extend(points)
                all_colors.extend(colors)
                pbar.update(1)

    # After processing all frames, create a point cloud
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined.points = o3d.utility.Vector3dVector(np.array(all_points))
    pcd_combined.colors = o3d.utility.Vector3dVector(np.array(all_colors))

    # Downsample the combined point cloud
    voxel_size = 1  # Adjust as needed
    pcd_combined_downsampled = pcd_combined.voxel_down_sample(voxel_size)

    # Visualize the final combined point cloud
    o3d.visualization.draw_geometries([pcd_combined_downsampled])

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    video_path = "test.mp4"  # Replace with your actual path
    main(video_path, num_frames=100)  # Process only the first 100 frames for debugging
