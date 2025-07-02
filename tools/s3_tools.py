#!/usr/bin/env python3
"""
Simple script to upload/download zip files to/from S3 bucket.

Usage examples:
  # Upload a zip file
  python traindata_upload_s3.py path/to/your/file.zip
  
  # Download a zip file  
  python traindata_upload_s3.py --download renderformer/traindata/file.zip
  
  # List files in S3 bucket
  python traindata_upload_s3.py --list
  
  # Test connection
  python traindata_upload_s3.py --test-connection
"""

import os
import sys
import argparse
import logging
from pathlib import Path

try:
    import s3fs
except ImportError:
    print("Error: s3fs not installed. Please run: pip install s3fs")
    sys.exit(1)

# S3 Configuration
S3_ENDPOINT = 'https://lml-qp-s3.shiyak-office.com'
S3_BUCKET = '3d-dataset'

# Get AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')


class SimpleS3Uploader:
    def __init__(self):
        """Initialize S3 uploader."""
        # Check if AWS credentials are provided
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            raise ValueError(
                "AWS credentials not found. Please set AWS_ACCESS_KEY_ID "
                "and AWS_SECRET_ACCESS_KEY environment variables."
            )
            
        self.fs = s3fs.S3FileSystem(
            key=AWS_ACCESS_KEY_ID,
            secret=AWS_SECRET_ACCESS_KEY,
            client_kwargs={'endpoint_url': S3_ENDPOINT}
        )
        self.bucket = S3_BUCKET
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def upload_zip(self, zip_file_path: str, s3_path: str = None):
        """Upload a zip file to S3."""
        zip_path = Path(zip_file_path)
        
        # Check if file exists and is a zip file
        if not zip_path.exists():
            raise FileNotFoundError(f"File not found: {zip_file_path}")
        
        if not zip_path.suffix.lower() == '.zip':
            raise ValueError(f"File must be a .zip file: {zip_file_path}")
        
        # Set S3 path if not provided
        if s3_path is None:
            s3_path = f"renderformer/traindata/{zip_path.name}"
        
        s3_full_path = f"{self.bucket}/{s3_path}"
        
        try:
            # Create parent directories if needed
            s3_dir = os.path.dirname(s3_full_path)
            if s3_dir and not self.fs.exists(s3_dir):
                self.fs.makedirs(s3_dir, exist_ok=True)
            
            # Upload file
            self.logger.info(f"Uploading {zip_path.name} to S3...")
            self.fs.put(str(zip_path), s3_full_path)
            
            # Get file size for confirmation
            file_size = zip_path.stat().st_size / (1024 * 1024)  # MB
            
            self.logger.info(
                f"Successfully uploaded {zip_path.name} "
                f"({file_size:.1f} MB) to s3://{s3_full_path}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload {zip_path.name}: {str(e)}")
            return False

    def download_zip(self, s3_path: str, local_path: str = None):
        """Download a zip file from S3."""
        s3_full_path = f"{self.bucket}/{s3_path}"
        
        # Check if S3 file exists
        if not self.fs.exists(s3_full_path):
            raise FileNotFoundError(
                f"S3 file not found: s3://{s3_full_path}"
            )
        
        # Set local path if not provided
        if local_path is None:
            local_path = os.path.basename(s3_path)
        
        local_file_path = Path(local_path)
        
        # Create local directory if needed
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download file
            file_name = os.path.basename(s3_path)
            self.logger.info(f"Downloading {file_name} from S3...")
            self.fs.get(s3_full_path, str(local_file_path))
            
            # Get file size for confirmation
            file_size = local_file_path.stat().st_size / (1024 * 1024)  # MB
            
            self.logger.info(
                f"Successfully downloaded {local_file_path.name} "
                f"({file_size:.1f} MB) from s3://{s3_full_path}"
            )
            return True
            
        except Exception as e:
            file_name = os.path.basename(s3_path)
            self.logger.error(f"Failed to download {file_name}: {str(e)}")
            return False

    def list_files(self, s3_path: str = "renderformer/traindata/"):
        """List files in S3 bucket path."""
        s3_full_path = f"{self.bucket}/{s3_path}"
        
        try:
            self.logger.info(f"Listing files in s3://{s3_full_path}")
            
            files = self.fs.ls(s3_full_path, detail=True)
            
            if not files:
                self.logger.info("No files found.")
                return []
            
            print(f"\nFiles in s3://{s3_full_path}:")
            print("-" * 60)
            
            file_list = []
            for file_info in files:
                if file_info['type'] == 'file':
                    name = os.path.basename(file_info['name'])
                    size_mb = file_info['size'] / (1024 * 1024)
                    modified = file_info.get('LastModified', 'Unknown')
                    
                    print(f"{name:<40} {size_mb:>8.1f} MB  {modified}")
                    file_list.append(name)
            
            return file_list
            
        except Exception as e:
            self.logger.error(f"Failed to list files: {str(e)}")
            return []


def main():
    parser = argparse.ArgumentParser(
        description="Upload/Download zip files to/from S3 bucket"
    )
    
    parser.add_argument(
        'file_path',
        nargs='?',
        help='Path to the zip file (for upload) or S3 path (for download)'
    )
    
    parser.add_argument(
        '--s3-path',
        help='S3 destination path (for upload) or local path (for download)'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download mode: download file from S3'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List files in S3 bucket'
    )
    
    parser.add_argument(
        '--list-path',
        default='renderformer/traindata/',
        help='S3 path to list files from (default: renderformer/traindata/)'
    )
    
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test S3 connection and exit'
    )
    
    args = parser.parse_args()
    
    uploader = SimpleS3Uploader()
    
    # List files mode
    if args.list:
        uploader.list_files(args.list_path)
        sys.exit(0)
    
    # Download mode
    if args.download:
        if not args.s3_path:
            parser.error("S3 file path required for download mode")
        
        try:
            success = uploader.download_zip(args.s3_path, args.file_path)
            if not success:
                sys.exit(1)
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
        return
    
    # Upload mode (default)
    if not args.file_path:
        parser.error("zip_file required for upload (use --help for options)")
    
    try:
        success = uploader.upload_zip(args.file_path, args.s3_path)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 