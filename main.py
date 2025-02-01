import torch
import cv2
import numpy as np
import open3d as o3d
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import concurrent.futures

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

def process_frame(frame_rgb, pbar=None):
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

            # Extract RGB color from the frame (already in RGB)
            r, g, b = frame_rgb[v, u]
            colors.append([r / 255.0, g / 255.0, b / 255.0])  # Normalize to [0, 1]

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))  # Set color from the original frame

    if pbar:
        pbar.update(1)  # Update tqdm progress bar inside the worker function

    return pcd

def main(video_path, num_frames=0):
    cap = cv2.VideoCapture(video_path)

    if num_frames == 0:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        total_frames = num_frames

    print(f"Frames: {total_frames}")

    pcd_combined = o3d.geometry.PointCloud()  # Initialize an empty point cloud to accumulate frames
    frame_count = 0
    voxel_size = 1  # Adjust the voxel size as needed

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        # Using ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for frame_count in range(total_frames):
                ret, frame_rgb = cap.read()
                if not ret:
                    break
                # Submit the frame processing to the executor
                futures.append(executor.submit(process_frame, frame_rgb, pbar))

            cap.release()  # Release the video capture object

            # Collect the results and accumulate the point clouds
            for future in concurrent.futures.as_completed(futures):
                pcd = future.result()

                # Downsample the point cloud for the current frame
                pcd_downsampled = pcd.voxel_down_sample(voxel_size)

                # Combine the downsampled point clouds from each frame
                pcd_combined += pcd_downsampled

    # Visualize the final combined point cloud
    o3d.visualization.draw_geometries([pcd_combined])

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    video_path = "test.mp4"  # Replace with your actual path
    main(video_path, num_frames=75)  # Process only the first 100 frames for debugging
