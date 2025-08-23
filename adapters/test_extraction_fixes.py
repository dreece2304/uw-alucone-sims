#!/usr/bin/env python3
"""
Test extraction fixes on a few sample files to verify they work before running full extraction.
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, 'pySPM_source')
sys.path.insert(0, '.')

import pySPM
from adapters.pyspm_fixes import open_iontof_with_fixes, validate_extracted_spectrum
from iontof_patcher import apply_pySPM_patches

def test_single_file(file_path):
    """Test extraction on a single file."""
    print(f"\n{'='*60}")
    print(f"TESTING: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Test with our fixes
        masses, intensities = open_iontof_with_fixes(file_path)
        
        if validate_extracted_spectrum(masses, intensities, file_path):
            print(f"✅ SUCCESS: Extracted {len(masses)} points")
            print(f"   Mass range: {masses.min():.3f} - {masses.max():.3f} u")
            print(f"   Intensity range: {intensities.min():.3e} - {intensities.max():.3e}")
            return True
        else:
            print(f"❌ FAILED: Invalid spectrum data")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    print("="*80)
    print("TESTING EXTRACTION FIXES")
    print("="*80)
    
    # Apply patches first
    apply_pySPM_patches()
    
    # Test files - one from each format and polarity
    test_files = [
        "01_raw_data/positive/P1_SQ1_01.itm",      # ITM positive
        "01_raw_data/positive/P1_SQ1_01.itax",     # ITAX positive  
        "01_raw_data/positive/P1_SQ1_01.itmx",     # ITMX positive
        "01_raw_data/negative/P1_SQ1_06.itm",      # ITM negative
        "01_raw_data/negative/P1_SQ1_06.itax",     # ITAX negative
        "01_raw_data/negative/P1_SQ1_06.itmx",     # ITMX negative
    ]
    
    results = {}
    for file_path in test_files:
        file_type = Path(file_path).suffix.upper()
        polarity = "positive" if "positive" in file_path else "negative"
        key = f"{file_type}_{polarity}"
        
        success = test_single_file(file_path)
        results[key] = success
    
    print(f"\n{'='*80}")
    print("EXTRACTION TEST SUMMARY")
    print(f"{'='*80}")
    
    for key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{key:15} {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {success_count}/{total_count} formats working")
    
    if success_count >= total_count * 0.5:  # At least 50% working
        print("✅ Extraction fixes are working - proceed with full extraction")
        return True
    else:
        print("❌ Too many extraction failures - need more fixes")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)