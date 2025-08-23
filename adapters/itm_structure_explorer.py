#!/usr/bin/env python3
"""
Explore ITM file structure to understand what spatial data is available.
"""

import sys
sys.path.insert(0, '../pySPM_source')
import pySPM
from pathlib import Path

def explore_itm_structure(file_path):
    """Explore the structure of an ITM file to understand available data."""
    print(f"Exploring ITM structure: {file_path}")
    
    try:
        # Try basic ITM loading
        itm = pySPM.ITM(str(file_path))
        print("✓ ITM loaded successfully with standard method")
        
        # Check what attributes are available
        attrs = ['size', 'polarity', 'Nscan', 'spp', 'sf', 'k0', 'peaks']
        for attr in attrs:
            if hasattr(itm, attr):
                value = getattr(itm, attr)
                print(f"  {attr}: {value}")
        
        # Try to get imaging information
        if hasattr(itm, 'size'):
            print(f"  Image dimensions: {itm.size}")
        
        # Check if we can get images
        try:
            images = itm.getImages()
            print(f"  Available images: {len(images) if images else 0}")
        except Exception as e:
            print(f"  getImages failed: {e}")
        
        # Check for specific masses
        try:
            img = itm.getImage(mass=27)  # Try aluminum
            if img is not None:
                print(f"  Mass 27 image shape: {img.shape}")
        except Exception as e:
            print(f"  getImage(27) failed: {e}")
            
    except Exception as e:
        if "Missing block" in str(e):
            print(f"⚠ ITM standard loading failed: {e}")
            print("Trying manual structure exploration...")
            
            # Try manual exploration
            try:
                itm = object.__new__(pySPM.ITM)
                itm.filename = str(file_path)
                itm.f = open(str(file_path), 'rb')
                itm.Type = itm.f.read(8)
                
                if itm.Type == b'ITStrF01':
                    print("✓ Valid IONTOF file header")
                    
                    import pySPM.Block
                    itm.root = pySPM.Block.Block(itm.f)
                    
                    # Explore available blocks
                    print("Available blocks:")
                    try:
                        blocks = itm.root.get_list()
                        for i, block in enumerate(blocks[:10]):  # Show first 10
                            print(f"  {i}: {block}")
                    except Exception as e:
                        print(f"  Block exploration failed: {e}")
                else:
                    print("❌ Invalid file type")
            except Exception as e2:
                print(f"❌ Manual exploration failed: {e2}")

if __name__ == "__main__":
    # Test with one ITM file
    test_file = Path("../data/PositiveIonData/P1_SQ1_01.itm")
    if test_file.exists():
        explore_itm_structure(test_file)
    else:
        print(f"Test file not found: {test_file}")