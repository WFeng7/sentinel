"""
Video management utilities for Sentinel.
Handles video file operations and path management.
"""

import os
from pathlib import Path
from typing import List, Optional

from .s3_manager import s3_manager

class VideoManager:
    """Manages video files for Sentinel."""
    
    def __init__(self):
        # Updated video path - now in backend/app/videos/
        self.videos_dir = Path(__file__).parent.parent / "app" / "videos"
        self.videos_dir.mkdir(exist_ok=True)
    
    def get_video_path(self, filename: str) -> Path:
        """Get full path to a video file."""
        return self.videos_dir / filename
    
    def list_videos(self) -> List[str]:
        """List all video files in the videos directory."""
        if not self.videos_dir.exists():
            return []
        
        video_files = []
        for file_path in self.videos_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.mov', '.mp4', '.avi']:
                video_files.append(file_path.name)
        
        return sorted(video_files)
    
    def video_exists(self, filename: str) -> bool:
        """Check if a video file exists."""
        return self.get_video_path(filename).exists()
    
    def upload_video_to_s3(self, filename: str, camera_id: str) -> Optional[str]:
        """Upload a video file to S3."""
        video_path = self.get_video_path(filename)
        return s3_manager.upload_video_file(video_path, camera_id)
    
    def upload_all_videos_to_s3(self) -> int:
        """Upload all videos to S3 and return count of successful uploads."""
        video_files = self.list_videos()
        uploaded_count = 0
        
        for video_file in video_files:
            # Use filename without extension as camera_id for testing
            camera_id = Path(video_file).stem
            if self.upload_video_to_s3(video_file, camera_id):
                uploaded_count += 1
        
        return uploaded_count
    
    def get_fake_camera_path(self, filename: str = "2026-01-3015-25-54.mov") -> Path:
        """Get path to the fake camera video file."""
        video_path = self.get_video_path(filename)
        
        # If video doesn't exist in new location, try to move it from old locations
        if not video_path.exists():
            self._move_fake_video(filename)
        
        return video_path
    
    def _move_fake_video(self, filename: str):
        """Move fake video from old locations to new videos directory."""
        # Check common old locations
        old_locations = [
            Path(__file__).parent.parent.parent / filename,  # Project root
            Path(__file__).parent.parent / "data" / "videos" / filename,  # backend/data/videos
            Path(__file__).parent.parent / "app" / "motion_first" / "test-video-processing" / filename,  # motion_first folder
        ]
        
        for old_path in old_locations:
            if old_path.exists():
                try:
                    # Move to new location
                    new_path = self.get_video_path(filename)
                    import shutil
                    shutil.move(str(old_path), str(new_path))
                    print(f"[Video] Moved fake video from {old_path} to {new_path}")
                    return
                except Exception as e:
                    print(f"[Video] Failed to move {old_path}: {e}")
        
        print(f"[Video] Fake video {filename} not found in any old location")

# Global video manager instance
video_manager = VideoManager()
