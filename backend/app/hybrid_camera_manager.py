"""
Hybrid Camera Manager
Intelligently routes camera processing between EC2 and local based on availability.
"""

import os
import json
import subprocess
import sys
from typing import Dict, List, Optional
from .ec2_camera_registry import ec2_registry, EC2Camera

class HybridCameraManager:
    """Manages camera processing across EC2 and local resources."""
    
    def __init__(self):
        self.ec2_registry = ec2_registry
        
    def get_camera_processing_status(self, camera_id: str) -> Dict:
        """Get processing status and location for a camera."""
        # Check if camera is on EC2
        ec2_camera = self.ec2_registry.get_camera_instance(camera_id)
        
        if ec2_camera:
            return {
                "camera_id": camera_id,
                "processing_location": "ec2",
                "instance_id": ec2_camera.instance_id,
                "public_ip": ec2_camera.public_ip,
                "instance_type": ec2_camera.instance_type,
                "status": ec2_camera.status,
                "recommendation": "EC2 processing available"
            }
        else:
            return {
                "camera_id": camera_id,
                "processing_location": "local",
                "instance_id": None,
                "public_ip": None,
                "instance_type": None,
                "status": "local_available",
                "recommendation": "Use local processing"
            }
    
    def start_camera_processing(self, camera_id: str, force_local: bool = False) -> Dict:
        """Start processing for a camera, preferring EC2 if available."""
        ec2_camera = self.ec2_registry.get_camera_instance(camera_id)
        
        if ec2_camera and not force_local:
            # Camera is on EC2 - check if processing is active
            ec2_status = self._check_ec2_processing_status(ec2_camera)
            
            if ec2_status["running"]:
                return {
                    "camera_id": camera_id,
                    "status": "already_running",
                    "location": "ec2",
                    "instance_id": ec2_camera.instance_id,
                    "message": f"Camera {camera_id} already processing on EC2 instance {ec2_camera.instance_id}"
                }
            else:
                # Start processing on EC2
                return self._start_ec2_processing(ec2_camera)
        else:
            # Process locally
            return self._start_local_processing(camera_id)
    
    def _check_ec2_processing_status(self, ec2_camera: EC2Camera) -> Dict:
        """Check if camera is actively processing on EC2 instance."""
        try:
            url = f"http://{ec2_camera.public_ip}:8000/camera-status/{ec2_camera.camera_id}"
            
            with httpx.Client(timeout=5) as client:
                response = client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "running": data.get("processing", False),
                        "last_update": data.get("last_update"),
                        "fps": data.get("fps", 0)
                    }
        except Exception as e:
            print(f"Error checking EC2 status: {e}")
        
        return {"running": False}
    
    def _start_ec2_processing(self, ec2_camera: EC2Camera) -> Dict:
        """Start camera processing on EC2 instance."""
        try:
            url = f"http://{ec2_camera.public_ip}:8000/start-camera/{ec2_camera.camera_id}"
            
            with httpx.Client(timeout=10) as client:
                response = client.post(url)
                if response.status_code == 200:
                    return {
                        "camera_id": ec2_camera.camera_id,
                        "status": "started",
                        "location": "ec2",
                        "instance_id": ec2_camera.instance_id,
                        "message": f"Started processing {ec2_camera.camera_id} on EC2 instance {ec2_camera.instance_id}"
                    }
                else:
                    return {
                        "camera_id": ec2_camera.camera_id,
                        "status": "error",
                        "location": "ec2",
                        "error": f"Failed to start: {response.text}"
                    }
        except Exception as e:
            return {
                "camera_id": ec2_camera.camera_id,
                "status": "error",
                "location": "ec2",
                "error": f"Connection error: {str(e)}"
            }
    
    def _start_local_processing(self, camera_id: str) -> Dict:
        """Start camera processing locally."""
        try:
            # Get camera details
            cameras = self._load_local_cameras()
            camera = next((c for c in cameras if c['id'] == camera_id), None)
            
            if not camera:
                return {
                    "camera_id": camera_id,
                    "status": "error",
                    "location": "local",
                    "error": f"Camera {camera_id} not found"
                }
            
            # Start local processing using existing motion_first system
            cmd = [
                sys.executable,
                "-m",
                "backend.app.motion_first.motion_first_tracking",
                camera['stream'],
                "--camera-id", camera['id'],
                "--no-display",
                "--enable-vlm",
                "--enable-rag",
                "--sqs-queue-url", os.environ.get("SENTINEL_SQS_QUEUE_URL")
            ]
            
            # This would typically be run as a background process
            # For now, return the command that would be executed
            return {
                "camera_id": camera_id,
                "status": "command_ready",
                "location": "local",
                "command": " ".join(cmd),
                "message": f"Ready to start local processing for {camera_id}"
            }
            
        except Exception as e:
            return {
                "camera_id": camera_id,
                "status": "error",
                "location": "local",
                "error": f"Failed to prepare local command: {str(e)}"
            }
    
    def _load_local_cameras(self) -> List[Dict]:
        """Load local camera configuration."""
        try:
            with open('cameras_cache.txt', 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def get_all_cameras_status(self) -> List[Dict]:
        """Get status of all cameras with processing location."""
        cameras = self._load_local_cameras()
        status_list = []
        
        for camera in cameras:
            status = self.get_camera_processing_status(camera['id'])
            status.update({
                'label': camera['label'],
                'stream': camera['stream'],
                'lat': camera.get('lat'),
                'lng': camera.get('lng')
            })
            status_list.append(status)
        
        return status_list

# Global manager instance
hybrid_manager = HybridCameraManager()
