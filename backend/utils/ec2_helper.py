"""
EC2 helper utilities for Sentinel deployment.
Handles EC2 instance setup and PEM key management.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Dict

class EC2Helper:
    """Helper for EC2 instance operations."""
    
    def __init__(self, pem_file: Optional[str] = None):
        self.pem_file = pem_file or os.environ.get('AWS_PEM_FILE')
        
    def test_ec2_connectivity(self) -> bool:
        """Test if we're running on EC2 with proper IAM role."""
        try:
            import boto3
            # Try to get instance metadata
            import urllib.request
            response = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
            instance_id = response.read().decode()
            print(f"[EC2] Running on instance: {instance_id}")
            return True
        except Exception as e:
            print(f"[EC2] Not running on EC2 or no metadata access: {e}")
            return False
    
    def setup_aws_cli_with_pem(self) -> bool:
        """Configure AWS CLI to use instance profile instead of credentials."""
        if not self.pem_file or not os.path.exists(self.pem_file):
            print(f"[EC2] PEM file not found: {self.pem_file}")
            return False
        
        print(f"[EC2] Found PEM file: {self.pem_file}")
        print("[EC2] For EC2 instances, AWS CLI should use instance profile IAM role")
        print("[EC2] Make sure the EC2 instance has proper IAM role attached")
        
        return True
    
    def get_instance_info(self) -> Dict:
        """Get EC2 instance information."""
        info = {}
        
        try:
            import urllib.request
            
            # Instance ID
            try:
                response = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
                info['instance_id'] = response.read().decode()
            except:
                info['instance_id'] = 'unknown'
            
            # Instance type
            try:
                response = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-type', timeout=2)
                info['instance_type'] = response.read().decode()
            except:
                info['instance_type'] = 'unknown'
            
            # Region
            try:
                response = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/placement/availability-zone', timeout=2)
                az = response.read().decode()
                info['region'] = az[:-1]  # Remove last character for region
                info['availability_zone'] = az
            except:
                info['region'] = 'unknown'
                info['availability_zone'] = 'unknown'
            
            # Public IP
            try:
                response = urllib.request.urlopen('http://169.254.169.254/latest/meta-data/public-ipv4', timeout=2)
                info['public_ip'] = response.read().decode()
            except:
                info['public_ip'] = 'unknown'
                
        except Exception as e:
            print(f"[EC2] Error getting instance info: {e}")
        
        return info
    
    def check_iam_permissions(self) -> bool:
        """Check if the instance has proper S3 permissions."""
        try:
            import boto3
            s3 = boto3.client('s3')
            
            # Try to list buckets (basic S3 permission test)
            response = s3.list_buckets()
            bucket_names = [bucket['Name'] for bucket in response['Buckets']]
            
            # Check if our bucket exists
            target_bucket = 'sentinel-bucket-hackbrown'
            if target_bucket in bucket_names:
                print(f"[EC2] ✅ Found target bucket: {target_bucket}")
                return True
            else:
                print(f"[EC2] ⚠️  Target bucket not found: {target_bucket}")
                print(f"[EC2] Available buckets: {bucket_names[:5]}...")  # Show first 5
                return False
                
        except Exception as e:
            print(f"[EC2] ❌ IAM permissions check failed: {e}")
            return False
    
    def setup_environment(self) -> Dict:
        """Setup environment for EC2 deployment."""
        results = {
            'is_ec2': self.test_ec2_connectivity(),
            'instance_info': self.get_instance_info(),
            'iam_permissions': self.check_iam_permissions() if self.test_ec2_connectivity() else False
        }
        
        return results

# Global EC2 helper instance
ec2_helper = EC2Helper()
