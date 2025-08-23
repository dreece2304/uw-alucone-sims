#!/usr/bin/env python
"""
Alternative IONTOF reader that works around pySPM compatibility issues.
"""

import pySPM
import numpy as np
from pathlib import Path
import struct

def try_read_itax_alternative(file_path):
    """Try alternative ITAX reading approaches."""
    print(f"Trying alternative ITAX reading for {file_path}")
    
    try:
        # Try loading with different parameters
        itax = pySPM.ITAX(str(file_path))
        
        # Try getting spectrum with different approaches
        approaches = [
            lambda: itax.getSpectrum(),
            lambda: itax.getSpectrum(0),  # Try with index
            lambda: itax.getProfile(),    # Try profile instead
        ]
        
        for i, approach in enumerate(approaches):
            try:
                print(f"  Approach {i+1}...")
                result = approach()
                print(f"  ✓ Success with approach {i+1}: {type(result)}")
                
                if isinstance(result, tuple) and len(result) == 2:
                    masses, intensities = result
                    print(f"    Masses: {len(masses)} points")
                    print(f"    Intensities: {len(intensities)} points")
                    return masses, intensities
                elif hasattr(result, 'masses') and hasattr(result, 'intensities'):
                    return result.masses, result.intensities
                elif isinstance(result, dict):
                    if 'masses' in result and 'intensities' in result:
                        return result['masses'], result['intensities']
                
            except Exception as e:
                print(f"    Failed: {e}")
        
        # Try accessing raw data
        print("  Trying raw data access...")
        if hasattr(itax, 'root'):
            # Navigate the data structure
            root = itax.root
            print(f"    Root children: {root.get_list() if hasattr(root, 'get_list') else 'N/A'}")
            
    except Exception as e:
        print(f"  ITAX alternative failed: {e}")
        return None, None
    
    return None, None

def try_read_itm_alternative(file_path):
    """Try alternative ITM reading approaches."""
    print(f"Trying alternative ITM reading for {file_path}")
    
    try:
        # Try with more permissive loading
        itm = pySPM.ITM(str(file_path))
        
        # Try different spectrum extraction methods
        methods = ['get_spectrum', 'getSpectrum', 'sum_spectrum']
        
        for method_name in methods:
            if hasattr(itm, method_name):
                try:
                    print(f"  Trying {method_name}...")
                    method = getattr(itm, method_name)
                    result = method()
                    
                    if isinstance(result, tuple) and len(result) == 2:
                        masses, intensities = result
                        print(f"  ✓ Success: {len(masses)} masses, {len(intensities)} intensities")
                        return masses, intensities
                        
                except Exception as e:
                    print(f"    {method_name} failed: {e}")
        
    except Exception as e:
        if "Missing block" in str(e):
            print(f"  ITM file format incompatibility: {e}")
            # This specific error suggests the file structure doesn't match pySPM expectations
        else:
            print(f"  ITM alternative failed: {e}")
    
    return None, None

def try_manual_file_inspection(file_path):
    """Try to manually inspect the file structure."""
    print(f"Manual inspection of {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            # Read first 1024 bytes to look for headers/structure
            header = f.read(1024)
            
            # Look for text strings that might indicate data sections
            header_str = header.decode('utf-8', errors='ignore')
            print(f"  File header preview: {repr(header_str[:200])}")
            
            # Look for common IONTOF/ToF-SIMS markers
            markers = [b'IONTOF', b'TOF', b'spectrum', b'mass', b'intensity']
            found_markers = [marker for marker in markers if marker in header]
            if found_markers:
                print(f"  Found markers: {found_markers}")
            
    except Exception as e:
        print(f"  Manual inspection failed: {e}")

def test_alternative_reading():
    """Test alternative reading approaches."""
    test_files = [
        "data/PositiveIonData/P1_SQ1_01.itax",
        "data/PositiveIonData/P1_SQ1_01.itm", 
    ]
    
    for file_path in test_files:
        print(f"\n{'='*60}")
        print(f"Testing: {file_path}")
        print(f"{'='*60}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
        
        # Try manual inspection first
        try_manual_file_inspection(file_path)
        
        # Try specific approaches based on file type
        masses, intensities = None, None
        
        if file_path.suffix.lower() == '.itax':
            masses, intensities = try_read_itax_alternative(file_path)
        elif file_path.suffix.lower() == '.itm':
            masses, intensities = try_read_itm_alternative(file_path)
        
        if masses is not None and intensities is not None:
            print(f"✅ Successfully extracted spectrum!")
            print(f"   Masses: {len(masses)} points, range {min(masses):.3f} - {max(masses):.3f}")
            print(f"   Intensities: {len(intensities)} points, max {max(intensities):.3e}")
        else:
            print(f"❌ Could not extract spectrum")

if __name__ == "__main__":
    test_alternative_reading()