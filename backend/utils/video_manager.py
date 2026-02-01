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
# Global video manager instance
video_manager = VideoManager()
