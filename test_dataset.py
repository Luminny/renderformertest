#!/usr/bin/env python3
"""
Simple test script for RenderFormerDataset
"""

import os
import tempfile
import h5py
import numpy as np
import torch
from pathlib import Path

def test_dataset():
    """Test RenderFormerDataset"""
    print("Testing RenderFormerDataset...")
    
    # Import the dataset class
    try:
        from train_geo_raster import RenderFormerDataset
        print("✓ Successfully imported RenderFormerDataset")
    except ImportError as e:
        print(f"✗ Failed to import RenderFormerDataset: {e}")
        return False
    
    # Create temporary directory for test files
    # with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir = "F:/projects/renderformer/traindata/000-000" 
    print(f"Using temporary directory: {temp_dir}")
    
    # # Create multiple test H5 files
    # test_files = []
    # for i in range(3):
    #     test_file = os.path.join(temp_dir, f"test_{i}.h5")
    #     info = create_test_h5_file(test_file)
    #     test_files.append(info)
    #     print(f"  Created test file {i+1}/3")
    
    # Test dataset creation
    try:
        dataset = RenderFormerDataset(temp_dir, resolution=64)
        print(f"✓ Dataset created successfully")
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Expected length: 3")
        
        # if len(dataset) != 3:
        #     print("✗ Dataset length mismatch!")
        #     return False
            
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return False
    
    # Test data loading
    try:
        print("\nTesting data loading...")
        
        for i in range(min(2, len(dataset))):
            sample = dataset[i]
            print(f"  Sample {i+1}:")
            
            # Check required keys
            required_keys = ['triangles', 'c2w', 'fov', 'vn', 'file_path']
            for key in required_keys:
                if key in sample:
                    print(f"    ✓ {key}: {type(sample[key])}")
                else:
                    print(f"    ✗ Missing {key}")
                    return False

            # Check optional gt_img
            if 'gt_img' in sample:
                print(f"    ✓ gt_img: {type(sample['gt_img'])}")
            else:
                print(f"    - gt_img: None (optional)")

            # Check tensor shapes
            print(f"    triangles shape: {sample['triangles'].shape}")
            print(f"    c2w shape: {sample['c2w'].shape}")
            print(f"    fov shape: {sample['fov'].shape}")
            print(f"    vn shape: {sample['vn'].shape}")
            if sample['gt_img'] is not None:
                print(f"    gt_img shape: {sample['gt_img'].shape}")

            # Check data types
            for key in ['triangles', 'c2w', 'fov', 'vn']:
                if not isinstance(sample[key], torch.Tensor):
                    print(f"    ✗ {key} is not a torch.Tensor")
                    return False

            if sample['gt_img'] is not None:
                if not isinstance(sample['gt_img'], torch.Tensor):
                    print(f"    ✗ gt_img is not a torch.Tensor")
                    return False
            
            print()
        
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False
    
    # Test dataset iteration
    try:
        print("Testing dataset iteration...")
        count = 0
        for sample in dataset:
            count += 1
            if count > 2:  # Only test first 2 samples
                break
        
        print(f"✓ Successfully iterated through {count} samples")
        
    except Exception as e:
        print(f"✗ Failed to iterate dataset: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    return True

def main():
    """Run all tests"""
    print("RenderFormerDataset Test")
    print("=" * 30)
    
    # Test 1: Basic functionality
    if not test_dataset():
        print("✗ Basic dataset test failed!")
        return
    
    print("\n" + "=" * 30)
    print("✅ All tests passed!")
    print("\nThe RenderFormerDataset is working correctly.")

if __name__ == "__main__":
    main() 