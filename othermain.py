import torch
import cv2
import numpy as np
import open3d as o3d
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

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

def feature_tracking(prev_frame, curr_frame, prev_points):
    """Track features using Lucas-Kanade optical flow."""
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, None, **lk_params)
    return curr_points[status == 1], prev_points[status == 1]

def select_keyframes(frames, interval=10):
    """Select keyframes based on interval."""
    keyframes = []
    for i in range(0, len(frames), interval):
        keyframes.append(frames[i])
    return keyframes

def estimate_camera_pose(prev_points, curr_points, K):
    """Estimate the camera pose using the essential matrix."""
    # Find the essential matrix
    E, mask = cv2.findEssentialMat(curr_points, prev_points, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Decompose the essential matrix into R and t
    _, R, t, mask = cv2.recoverPose(E, curr_points, prev_points, K)

    return R, t

def main(video_path, frame_interval=10, batch_size=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames: {total_frames}")

    all_points = []
    all_colors = []
    prev_frame = None
    prev_points = None
    K = np.array([[958.0, 0, 640.0], [0, 958.0, 360.0], [0, 0, 1]])  # Camera intrinsic matrix

    frames = []
    frame_count = 0

    # Extract frames at the specified interval
    while cap.isOpened() and frame_count < total_frames:
        ret, frame_rgb = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame_rgb)
        frame_count += 1

    cap.release()  # Release the video capture object

    # Select keyframes
    keyframes = select_keyframes(frames, interval=frame_interval)

    # Initialize global transformation
    global_R = np.eye(3)
    global_t = np.zeros((3, 1))

    # Process the keyframes
    with tqdm(total=len(keyframes), desc="Processing keyframes") as pbar:
        for i, frame_rgb in enumerate(keyframes):
            if prev_frame is not None:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
                prev_points, curr_points = feature_tracking(prev_gray, curr_gray, prev_points)

                # Estimate the camera pose
                R, t = estimate_camera_pose(prev_points, curr_points, K)

                # Update global transformation
                global_t = global_t + global_R @ t
                global_R = R @ global_R

                # Process the current frame
                points, colors = process_frame(frame_rgb)
                batch_points = np.array(points)
                batch_colors = np.array(colors)

                # Register the point clouds using the global transformation
                transformed_points = np.dot(batch_points, global_R.T) + global_t.T

                # Combine the registered points with the existing point cloud
                all_points.extend(transformed_points)
                all_colors.extend(batch_colors)
            else:
                points, colors = process_frame(frame_rgb)
                all_points.extend(points)
                all_colors.extend(colors)

            prev_frame = frame_rgb
            prev_points = cv2.goodFeaturesToTrack(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY), maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)
            pbar.update(1)

    # After processing all frames, create a final point cloud
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined.points = o3d.utility.Vector3dVector(np.array(all_points))
    pcd_combined.colors = o3d.utility.Vector3dVector(np.array(all_colors))

    # Downsample the combined point cloud
    voxel_size = 0.05  # Adjust as needed
    pcd_combined_downsampled = pcd_combined.voxel_down_sample(voxel_size)

    # Visualize the final combined point cloud
    o3d.visualization.draw_geometries([pcd_combined_downsampled])

if __name__ == "__main__":
    video_path = "PXL_20250201_235449415.mp4"  # Replace with your actual path
    main(video_path, frame_interval=2, batch_size=28)  # Process in batches of 10 frames