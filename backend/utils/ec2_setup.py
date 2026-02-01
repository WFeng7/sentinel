"""
EC2 setup script for Sentinel deployment.
Tests EC2 connectivity and IAM permissions.
"""

import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.ec2_helper import ec2_helper
from utils.s3_manager import s3_manager

def test_s3_connection():
    """Test S3 connection and permissions."""
    print("🔍 Testing S3 connection...")
    
    try:
        # List buckets to test basic S3 access
        response = s3_manager.s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        print(f"✅ S3 connection successful. Found {len(buckets)} buckets")
        
        # Check for our target bucket
        target_bucket = 'sentinel-bucket-hackbrown'
        if target_bucket in buckets:
            print(f"✅ Target bucket found: {target_bucket}")
            
            # Test upload/download permissions
            test_key = 'test/connection-test.txt'
            try:
                # Upload test file
                s3_manager.s3_client.put_object(
                    Bucket=target_bucket,
                    Key=test_key,
                    Body='EC2 connection test',
                    ContentType='text/plain'
                )
                print("✅ S3 upload permission: OK")
                
                # Download test file
                response = s3_manager.s3_client.get_object(Bucket=target_bucket, Key=test_key)
                content = response['Body'].read().decode()
                if content == 'EC2 connection test':
                    print("✅ S3 download permission: OK")
                else:
                    print("❌ S3 download test failed")
                
                # Clean up test file
                s3_manager.s3_client.delete_object(Bucket=target_bucket, Key=test_key)
                print("✅ S3 delete permission: OK")
                
            except Exception as e:
                print(f"❌ S3 permissions test failed: {e}")
                return False
                
        else:
            print(f"❌ Target bucket not found: {target_bucket}")
            print(f"Available buckets: {buckets[:5]}...")
            return False
            
    except Exception as e:
        print(f"❌ S3 connection failed: {e}")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🚀 Sentinel EC2 Setup Test")
    print("=" * 40)
    
    # Test EC2 environment
    print("\n📋 EC2 Environment Check:")
    setup_results = ec2_helper.setup_environment()
    
    if setup_results['is_ec2']:
        print("✅ Running on EC2 instance")
        info = setup_results['instance_info']
        print(f"   Instance ID: {info.get('instance_id', 'unknown')}")
        print(f"   Instance Type: {info.get('instance_type', 'unknown')}")
        print(f"   Region: {info.get('region', 'unknown')}")
        print(f"   Public IP: {info.get('public_ip', 'unknown')}")
        
        if setup_results['iam_permissions']:
            print("✅ IAM permissions: OK")
        else:
            print("❌ IAM permissions: FAILED")
            print("   Make sure the EC2 instance has an IAM role with S3 access")
    else:
        print("⚠️  Not running on EC2 (local development)")
        print("   Using AWS credentials from environment")
    
    # Test S3 connection
    print("\n📦 S3 Connection Test:")
    s3_ok = test_s3_connection()
    
    # Summary
    print("\n📊 Setup Summary:")
    if setup_results['is_ec2']:
        print(f"   EC2 Instance: ✅")
        print(f"   IAM Role: {'✅' if setup_results['iam_permissions'] else '❌'}")
    else:
        print(f"   Local Development: ✅")
        print(f"   AWS Credentials: ✅")
    
    print(f"   S3 Access: {'✅' if s3_ok else '❌'}")
    
    if s3_ok:
        print("\n🎉 Setup complete! You can now run:")
        print("   python -m utils.s3_uploader --setup-samples")
        print("   python -m utils.s3_uploader")
    else:
        print("\n❌ Setup failed. Check IAM permissions and S3 bucket access.")
    
    return s3_ok and (not setup_results['is_ec2'] or setup_results['iam_permissions'])

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
