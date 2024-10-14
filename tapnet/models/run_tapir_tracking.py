import numpy as np
import os
import jax
import mediapy as media
from tapnet.models.tapir_model import ParameterizedTAPIR
from tapnet.utils import model_utils



def load_checkpoint(checkpoint_path):
    """Load pre-trained TAPIR model checkpoint."""
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    return ckpt_state["params"], ckpt_state["state"]

def preprocess_video(video_path):
    """Load and preprocess the video."""
    video = media.read_video(video_path)
    video = video.astype(np.float32) / 255.0 * 2.0 - 1.0  # Normalize to [-1, 1]
    video = np.expand_dims(video, axis=0)  # Add batch dimension [1, num_frames, height, width, 3]
    return video

def save_tracking_results(video, tracks, occlusions, video_save_path, npy_save_path):
    """Save the tracking results as video with tracks and as .npy file."""
    
    # Save the track and occlusion data as .npy file
    np.save(npy_save_path, {'tracks': tracks, 'occlusions': occlusions})

    # Overlay the tracks on the video frames and save as a new video
    num_frames = video.shape[1]
    output_frames = []
    for i in range(num_frames):
        frame = video[0, i]  # Get the current frame
        frame = (frame + 1) / 2.0  # Convert back to [0, 1] range for saving
        frame = (frame * 255).astype(np.uint8)  # Convert to uint8
        # Draw the tracks (in blue) on the frame
        for j in range(tracks.shape[1]):
            x, y = int(tracks[0, j, i, 0]), int(tracks[0, j, i, 1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                frame = media.draw_circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        output_frames.append(frame)
    
    # Save the frames as a video file
    media.write_video(video_save_path, output_frames, fps=25)

def run_tapir_on_folder(video_folder, checkpoint_path, output_folder):
    # Load the checkpoint and initialize the TAPIR model
    params, state = load_checkpoint(checkpoint_path)
    tapir = ParameterizedTAPIR(params=params, state=state)

    # Process each video in the folder
    for video_filename in os.listdir(video_folder):
        if video_filename.endswith('.mp4'):  # Check if it is a video file
            video_path = os.path.join(video_folder, video_filename)
            print(f"Processing video: {video_path}")
            
            # Preprocess the video (load and normalize it)
            video = preprocess_video(video_path)
            
            # Generate random query points (for simplicity)
            num_points = 5  # Number of points to track
            num_frames = video.shape[1]
            height, width = video.shape[2:4]
            query_points = np.random.rand(1, num_points, 3)  # Random query points in [t, y, x]
            query_points[..., 0] = np.random.randint(0, num_frames, (1, num_points))  # Time index (frame)
            query_points[..., 1] *= height  # Y position
            query_points[..., 2] *= width   # X position

            # Call TAPIR model to track the points
            result = tapir(video, is_training=False, query_points=query_points)

            # Extract the tracks and occlusions
            tracks = result['tracks']
            occlusions = result['occlusion']
            
            # Create paths for saving
            npy_save_path = os.path.join(output_folder, f"{os.path.splitext(video_filename)[0]}_tracking.npy")
            video_save_path = os.path.join(output_folder, f"{os.path.splitext(video_filename)[0]}_tracking.mp4")

            # Save the tracking results as both .npy and .mp4 files
            save_tracking_results(video, tracks, occlusions, video_save_path, npy_save_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TAPIR tracking on all videos in a folder.")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the folder containing input video files")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the TAPIR checkpoint file")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder to save output files")

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    run_tapir_on_folder(args.video_folder, args.checkpoint_path, args.output_folder)
