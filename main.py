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


def feature_matching(frame1, frame2):
    """Use ORB to detect and match features between two frames."""
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors in both frames
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    # Use BFMatcher to match the descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched points
    points1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    return points1, points2


def estimate_camera_pose(points1, points2, K):
    """Estimate camera pose using essential matrix decomposition."""
    # Find the essential matrix using the matched points
    E, mask = cv2.findEssentialMat(points1, points2, K)

    # Recover relative camera rotation and translation
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K)

    return R, t


def main(video_path, num_frames=0, batch_size=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = num_frames if num_frames > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Frames: {total_frames}")

    all_points = []
    all_colors = []
    prev_frame = None
    prev_kp = None
    prev_des = None
    K = np.array([[958.0, 0, 640.0], [0, 958.0, 360.0], [0, 0, 1]])  # Camera intrinsic matrix

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

                if prev_frame is not None:
                    # Match features between current and previous frame
                    points1, points2 = feature_matching(prev_frame, frame_rgb)

                    # Estimate the camera pose
                    R, t = estimate_camera_pose(points1, points2, K)

                    # Here you can register the point clouds using R, t
                    # Apply the transformation to align the point clouds

                    # For now, just append the new points to the combined list
                    all_points.extend(batch_points)
                    all_colors.extend(batch_colors)

                prev_frame = frame_rgb  # Update the previous frame
                batch_frames = []  # Clear the batch
                pbar.update(batch_size)  # Update progress bar for the batch size

        cap.release()  # Release the video capture object

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
    print(f"Using device: {device}")
    video_path = "test.mp4"  # Replace with your actual path
    main(video_path, num_frames=50, batch_size=10)  # Process in batches of 10 frames
