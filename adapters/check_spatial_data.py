#!/usr/bin/env python3
"""
Check what spatial data is available in ITAX vs ITM vs ITMX files.
"""

import sys
sys.path.insert(0, 'pySPM_source')
import pySPM
from pathlib import Path

def check_spatial_capabilities(file_path, file_type):
    """Check spatial data capabilities for different file types."""
    print(f"\n{'='*50}")
    print(f"Checking {file_type.upper()}: {file_path.name}")
    print(f"{'='*50}")
    
    try:
        if file_type == 'itax':
            obj = pySPM.ITAX(str(file_path))
        elif file_type == 'itm':
            obj = pySPM.ITM(str(file_path))
        else:
            print(f"Unsupported file type: {file_type}")
            return
            
        print("✓ File loaded successfully")
        
        # Check basic properties
        if hasattr(obj, 'size'):
            print(f"Size info: {obj.size}")
            
        # Check for imaging capabilities
        spatial_methods = [
            'getImages',
            'getImage', 
            'get_images',
            'get_image'
        ]
        
        for method in spatial_methods:
            if hasattr(obj, method):
                print(f"✓ Has method: {method}")
                try:
                    if method in ['getImages', 'get_images']:
                        result = getattr(obj, method)()
                        print(f"  {method}() returned: {type(result)}, length: {len(result) if result else 0}")
                    elif method in ['getImage', 'get_image']:
                        # Try with a common mass
                        result = getattr(obj, method)(mass=27)  # Aluminum
                        if result is not None:
                            print(f"  {method}(mass=27) shape: {result.shape}")
                        else:
                            print(f"  {method}(mass=27) returned None")
                except Exception as e:
                    print(f"  {method} failed: {e}")
            else:
                print(f"✗ No method: {method}")
        
        # Check if this is an imaging file
        try:
            # Try to access imaging-related attributes
            img_attrs = ['pixels', 'real_size', 'fov']
            for attr in img_attrs:
                if hasattr(obj, attr):
                    value = getattr(obj, attr)
                    print(f"Imaging attribute {attr}: {value}")
        except Exception as e:
            print(f"Imaging attribute check failed: {e}")
            
        # For ITAX, check if there's any spatial data structure
        if file_type == 'itax':
            try:
                # Check the internal structure for spatial data
                spatial_paths = [
                    "ILTotalSpectrum",
                    "Images",
                    "ImageStack", 
                    "SpatialData"
                ]
                
                for path in spatial_paths:
                    try:
                        node = obj.root.goto(path)
                        print(f"✓ Found spatial structure: {path}")
                        # Try to explore this structure
                        children = node.get_list()
                        print(f"  Children: {len(children)} items")
                    except:
                        pass
                        
            except Exception as e:
                print(f"ITAX spatial structure check failed: {e}")
                
    except Exception as e:
        print(f"❌ Failed to load {file_type} file: {e}")

if __name__ == "__main__":
    base_path = Path("data/PositiveIonData")
    
    # Test the same sample in different formats
    sample_files = [
        (base_path / "P1_SQ1_01.itax", "itax"),
        (base_path / "P1_SQ1_01.itm", "itm"),
        (base_path / "P1_SQ1_01.itmx", "itmx")
    ]
    
    for file_path, file_type in sample_files:
        if file_path.exists():
            check_spatial_capabilities(file_path, file_type)
        else:
            print(f"\n❌ File not found: {file_path}")