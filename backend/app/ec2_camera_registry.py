"""
EC2 Camera Registry Service
Tracks which cameras are running on EC2 instances and provides fallback logic.
"""

import os
import json
import boto3
import httpx
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EC2Camera:
    """Camera running on EC2 instance."""
    camera_id: str
    instance_id: str
    public_ip: str
    private_ip: str
    instance_type: str
    status: str
    stream_url: str
    label: str

class EC2CameraRegistry:
    """Registry for EC2-hosted cameras."""
    
    def __init__(self):
        self.ec2_client = boto3.client("ec2", region_name="us-east-1")
        self.cache_timeout = 300  # 5 minutes
        self._cache = {}
        self._cache_timestamp = 0
        
    def get_ec2_cameras(self) -> List[EC2Camera]:
        """Get all cameras running on EC2 instances."""
        current_time = time.time()
        
        # Return cached data if still valid
        if (current_time - self._cache_timestamp) < self.cache_timeout and self._cache:
            return self._cache
        
        cameras = []
        
        try:
            # Get all running Sentinel Camera instances
            response = self.ec2_client.describe_instances(
                Filters=[
                    {'Name': 'tag:Name', 'Values': ['Sentinel-Camera']},
                    {'Name': 'instance-state-name', 'Values': ['running']}
                ]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    # Get camera assignment from instance
                    camera_data = self._get_camera_assignment(instance['InstanceId'])
                    
                    if camera_data:
                        for instance_name, cam_list in camera_data.items():
                            for cam in cam_list:
                                cameras.append(EC2Camera(
                                    camera_id=cam['id'],
                                    instance_id=instance['InstanceId'],
                                    public_ip=instance.get('PublicIpAddress', ''),
                                    private_ip=instance.get('PrivateIpAddress', ''),
                                    instance_type=instance['InstanceType'],
                                    status=instance['State']['Name'],
                                    stream_url=cam['stream'],
                                    label=cam['label']
                                ))
            
            # Update cache
            self._cache = cameras
            self._cache_timestamp = current_time
            
        except Exception as e:
            print(f"Error getting EC2 cameras: {e}")
        
        return cameras
    
    def _get_camera_assignment(self, instance_id: str) -> Optional[Dict]:
        """Get camera assignment from EC2 instance."""
        try:
            # Try to get camera assignment via HTTP
            instance_ip = self._get_instance_ip(instance_id)
            if not instance_ip:
                return None
            
            # Use environment-aware URL construction
            if self._is_running_on_ec2():
                # If running on EC2, use private IP for internal communication
                instance_ip = self._get_instance_private_ip(instance_id)
                url = f"http://{instance_ip}:8000/camera-assignment"
            else:
                # If running locally (dev), use public IP
                url = f"http://{instance_ip}:8000/camera-assignment"
            
            # Allow override via environment variable for production
            ec2_service_port = os.environ.get('EC2_CAMERA_SERVICE_PORT', '8000')
            ec2_service_protocol = os.environ.get('EC2_CAMERA_SERVICE_PROTOCOL', 'http')
            url = f"{ec2_service_protocol}://{instance_ip}:{ec2_service_port}/camera-assignment"
            
            with httpx.Client(timeout=10) as client:
                response = client.get(url)
                if response.status_code == 200:
                    return response.json()
        
        except Exception as e:
            print(f"Error getting camera assignment for {instance_id}: {e}")
        
        return None
    
    def _get_instance_ip(self, instance_id: str) -> Optional[str]:
        """Get public IP of instance."""
        try:
            response = self.ec2_client.describe_instances(
                InstanceIds=[instance_id]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    return instance.get('PublicIpAddress')
        
        except Exception:
            pass
        
        return None
    
    def _get_instance_private_ip(self, instance_id: str) -> Optional[str]:
        """Get private IP of instance."""
        try:
            response = self.ec2_client.describe_instances(
                InstanceIds=[instance_id]
            )
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    return instance.get('PrivateIpAddress')
        
        except Exception:
            pass
        
        return None
    
    def _is_running_on_ec2(self) -> bool:
        """Check if currently running on EC2 instance."""
        try:
            import urllib.request
            urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
            return True
        except Exception:
            return False
    
    def is_camera_on_ec2(self, camera_id: str) -> bool:
        """Check if a camera is running on EC2."""
        ec2_cameras = self.get_ec2_cameras()
        return any(cam.camera_id == camera_id for cam in ec2_cameras)
    
    def get_camera_instance(self, camera_id: str) -> Optional[EC2Camera]:
        """Get EC2 instance info for a camera."""
        ec2_cameras = self.get_ec2_cameras()
        for cam in ec2_cameras:
            if cam.camera_id == camera_id:
                return cam
        return None
    
    def get_all_cameras_with_ec2_status(self) -> List[Dict]:
        """Get all cameras with EC2 status."""
        # Load local cameras
        local_cameras = []
        try:
            with open('cameras_cache.txt', 'r') as f:
                local_cameras = json.load(f)
        except Exception:
            pass
        
        # Get EC2 cameras
        ec2_cameras = self.get_ec2_cameras()
        ec2_camera_ids = {cam.camera_id for cam in ec2_cameras}
        
        # Merge and mark EC2 status
        result = []
        for cam in local_cameras:
            cam_copy = cam.copy()
            cam_copy['is_ec2'] = cam['id'] in ec2_camera_ids
            
            if cam_copy['is_ec2']:
                ec2_cam = next((c for c in ec2_cameras if c.camera_id == cam['id']), None)
                if ec2_cam:
                    cam_copy['ec2_instance'] = {
                        'instance_id': ec2_cam.instance_id,
                        'public_ip': ec2_cam.public_ip,
                        'instance_type': ec2_cam.instance_type,
                        'status': ec2_cam.status
                    }
            
            result.append(cam_copy)
        
        return result

# Global registry instance
ec2_registry = EC2CameraRegistry()
