#!/usr/bin/env python
"""
Debug script to investigate pySPM compatibility issues with IONTOF files.
"""

import pySPM
from pathlib import Path
import traceback

def debug_iontof_file(file_path):
    """Debug a single IONTOF file with multiple approaches."""
    print(f"\n{'='*60}")
    print(f"Debugging: {file_path}")
    print(f"{'='*60}")
    
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"File size: {file_path.stat().st_size} bytes")
    print(f"File extension: {file_path.suffix}")
    
    # Try different pySPM loaders
    loaders = []
    if file_path.suffix.lower() == '.itm':
        loaders = [
            ('pySPM.ITM', lambda p: pySPM.ITM(str(p))),
            ('pySPM.Block', lambda p: pySPM.Block(str(p))),
        ]
    elif file_path.suffix.lower() == '.itax':
        loaders = [
            ('pySPM.ITAX', lambda p: pySPM.ITAX(str(p))),
            ('pySPM.Block', lambda p: pySPM.Block(str(p))),
        ]
    elif file_path.suffix.lower() == '.ita':
        loaders = [
            ('pySPM.ITA', lambda p: pySPM.ITA(str(p))),
            ('pySPM.Block', lambda p: pySPM.Block(str(p))),
        ]
    
    successful_obj = None
    
    for loader_name, loader_func in loaders:
        print(f"\n--- Trying {loader_name} ---")
        try:
            obj = loader_func(file_path)
            print(f"✓ Successfully loaded with {loader_name}")
            print(f"Object type: {type(obj)}")
            print(f"Available attributes: {[attr for attr in dir(obj) if not attr.startswith('_')][:10]}...")
            
            # Try to get spectrum
            print("Attempting to extract spectrum...")
            
            # List of spectrum methods to try
            spectrum_methods = [
                'get_spectrum',
                'getSpectrum', 
                'sum_spectrum',
                'get_sum_spectrum'
            ]
            
            for method_name in spectrum_methods:
                if hasattr(obj, method_name):
                    print(f"Found method: {method_name}")
                    try:
                        method = getattr(obj, method_name)
                        result = method()
                        print(f"✓ {method_name}() returned: {type(result)}")
                        if hasattr(result, '__len__'):
                            print(f"  Length: {len(result)}")
                        if isinstance(result, tuple) and len(result) == 2:
                            masses, intensities = result
                            print(f"  Masses: {len(masses)} points, range {min(masses):.3f} - {max(masses):.3f}")
                            print(f"  Intensities: {len(intensities)} points, range {min(intensities):.3e} - {max(intensities):.3e}")
                            successful_obj = obj
                            break
                    except Exception as e:
                        print(f"  {method_name}() failed: {e}")
            
            # Try attribute access
            if not successful_obj:
                print("Trying direct attribute access...")
                attribute_pairs = [
                    ('masses', 'spectrum'),
                    ('masses', 'intensities'),
                    ('mass', 'intensity'),
                    ('mz', 'intensity')
                ]
                
                for mass_attr, intensity_attr in attribute_pairs:
                    if hasattr(obj, mass_attr) and hasattr(obj, intensity_attr):
                        try:
                            masses = getattr(obj, mass_attr)
                            intensities = getattr(obj, intensity_attr)
                            print(f"✓ Found attributes: {mass_attr}, {intensity_attr}")
                            print(f"  Masses: {len(masses)} points")
                            print(f"  Intensities: {len(intensities)} points")
                            successful_obj = obj
                            break
                        except Exception as e:
                            print(f"  Attribute access {mass_attr}/{intensity_attr} failed: {e}")
            
            if successful_obj:
                break
                
        except Exception as e:
            print(f"✗ Failed with {loader_name}: {e}")
            # Print more detailed error for analysis
            print(f"  Error type: {type(e).__name__}")
            if "Missing block" in str(e):
                print("  This appears to be a file format compatibility issue")
            elif "tell" in str(e):
                print("  This appears to be a file handle issue")
    
    if not successful_obj:
        print(f"\n❌ No loader worked for {file_path}")
    else:
        print(f"\n✅ Successfully processed {file_path}")
    
    return successful_obj is not None

if __name__ == "__main__":
    # Test a few different file types from the user's data
    test_files = [
        "data/PositiveIonData/P1_SQ1_01.itm",
        "data/PositiveIonData/P1_SQ1_01.itax", 
        "data/PositiveIonData/P1_SQ1_01.itmx",
    ]
    
    success_count = 0
    for file_path in test_files:
        if debug_iontof_file(file_path):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(test_files)} files successfully processed")
    print(f"{'='*60}")