#!/usr/bin/env python
"""
Workaround for IONTOF compatibility issues.

Since pySPM has compatibility issues with your specific IONTOF files, 
let's try to patch the issues or provide alternative approaches.
"""

import pySPM
import numpy as np
from pathlib import Path
import struct

# Monkey patch to handle the buffer size issue in ITAX files
def patched_getSpectrum(self, *args, **kwargs):
    """Patched version of ITAX getSpectrum that handles buffer size issues."""
    try:
        # Try the original method first
        return self._original_getSpectrum(*args, **kwargs)
    except struct.error as e:
        if "unpack requires a buffer" in str(e):
            print(f"Buffer size issue detected: {e}")
            print("Attempting to work around buffer size mismatch...")
            
            # Try to access the raw data more carefully
            try:
                # Navigate to spectrum data location
                if hasattr(self, 'root'):
                    # Look for spectrum or profile data in the file structure
                    children = self.root.get_list() if hasattr(self.root, 'get_list') else []
                    print(f"Available data sections: {[c['name'] for c in children]}")
                    
                    # Look for common spectrum data locations
                    spectrum_paths = [
                        'Spectrum',
                        'SpectrumData', 
                        'Profile',
                        'Data',
                        'CommonDataObjects'
                    ]
                    
                    for path in spectrum_paths:
                        try:
                            data_section = self.root.goto(path)
                            if data_section:
                                print(f"Found data section: {path}")
                                # Try to extract data from this section
                                section_children = data_section.get_list() if hasattr(data_section, 'get_list') else []
                                print(f"  Children: {[c['name'] for c in section_children]}")
                                
                        except Exception as e2:
                            continue
                            
            except Exception as e2:
                print(f"Raw data access failed: {e2}")
            
            # Return empty arrays as fallback
            return np.array([]), np.array([])
        else:
            raise

def patch_pySPM():
    """Apply patches to pySPM to handle compatibility issues."""
    # Patch ITAX getSpectrum method
    if hasattr(pySPM, 'ITAX') and hasattr(pySPM.ITAX, 'ITAX'):
        original_method = getattr(pySPM.ITAX.ITAX, 'getSpectrum', None)
        if original_method:
            pySPM.ITAX.ITAX._original_getSpectrum = original_method
            pySPM.ITAX.ITAX.getSpectrum = patched_getSpectrum
            print("Applied ITAX getSpectrum patch")

def try_alternative_libraries():
    """Check if there are alternative libraries for reading IONTOF files."""
    alternatives = []
    
    # Check if we can use any other ToF-SIMS libraries
    try:
        import h5py
        alternatives.append("h5py (for HDF5-based ToF-SIMS formats)")
    except ImportError:
        pass
    
    try:
        import scipy.io
        alternatives.append("scipy.io (for .mat files if available)")
    except ImportError:
        pass
    
    if alternatives:
        print(f"Alternative libraries available: {alternatives}")
    else:
        print("No alternative libraries found")

def create_mock_data_for_testing():
    """Create mock ToF-SIMS data for testing the pipeline."""
    print("\nCreating mock ToF-SIMS data for testing...")
    
    # Generate realistic ToF-SIMS-like data
    # Mass range typical for positive ion ToF-SIMS
    masses = np.arange(1, 200, 0.1)  # 0.1 Da resolution from 1-200 Da
    
    # Create synthetic spectra with realistic peak patterns
    np.random.seed(42)  # For reproducible results
    
    samples_data = {}
    
    for i, sample_name in enumerate(['P1_SQ1_01', 'P1_SQ2_01', 'P1_SQ3_01']):
        intensities = np.random.exponential(0.001, len(masses))  # Base noise level
        
        # Add characteristic peaks for positive ion ToF-SIMS
        common_peaks = [
            (1.0, 0.01),    # H+
            (7.0, 0.005),   # Li+
            (12.0, 0.02),   # C+
            (15.0, 0.015),  # CH3+
            (23.0, 0.05),   # Na+
            (27.0, 0.03),   # Al+ or C2H3+
            (39.0, 0.025),  # K+
            (43.0, 0.02),   # C2H3O+ or C3H7+
        ]
        
        for peak_mass, peak_intensity in common_peaks:
            # Add peak with some sample-to-sample variation
            variation = 1 + 0.2 * (i - 1)  # ±20% variation between samples
            peak_idx = np.argmin(np.abs(masses - peak_mass))
            intensities[peak_idx] += peak_intensity * variation
            
            # Add peak broadening
            sigma = 0.5  # Peak width
            for j in range(max(0, peak_idx-5), min(len(masses), peak_idx+5)):
                distance = abs(j - peak_idx) * 0.1  # 0.1 Da per index
                intensities[j] += peak_intensity * variation * np.exp(-distance**2/(2*sigma**2)) * 0.3
        
        samples_data[sample_name] = (masses, intensities)
        print(f"  Generated mock data for {sample_name}: {len(masses)} masses")
    
    return samples_data

def test_with_mock_data():
    """Test the conversion pipeline with mock data."""
    print(f"\n{'='*60}")
    print("Testing with mock data")
    print(f"{'='*60}")
    
    # Generate mock data
    mock_data = create_mock_data_for_testing()
    
    # Use the binning utilities to convert to PNNL format
    from adapters.binning import bin_to_unit_mass, merge_samples_to_tsv
    
    # Bin each sample to unit masses
    binned_data = {}
    for sample_name, (masses, intensities) in mock_data.items():
        unit_masses, binned_intensities = bin_to_unit_mass(masses, intensities)
        binned_data[sample_name] = (unit_masses, binned_intensities)
        print(f"Binned {sample_name}: {len(masses)} -> {len(unit_masses)} unit masses")
    
    # Create TSV file
    output_path = Path("results/mock_positive_test.tsv")
    output_path.parent.mkdir(exist_ok=True)
    
    merge_samples_to_tsv(binned_data, output_path, 'P')
    print(f"✅ Mock data TSV created: {output_path}")
    
    # Validate the output
    from adapters.binning import validate_pnnl_tsv
    if validate_pnnl_tsv(output_path):
        print("✅ Mock data TSV validation passed")
    else:
        print("❌ Mock data TSV validation failed")
    
    return str(output_path)

if __name__ == "__main__":
    print("IONTOF Compatibility Workaround")
    print(f"{'='*60}")
    
    # Try patching pySPM
    patch_pySPM()
    
    # Check for alternatives
    try_alternative_libraries()
    
    # Test one more time with patches
    print(f"\n{'='*60}")
    print("Testing with patches applied")
    print(f"{'='*60}")
    
    test_file = "data/PositiveIonData/P1_SQ1_01.itax"
    if Path(test_file).exists():
        try:
            itax = pySPM.ITAX(test_file)
            masses, intensities = itax.getSpectrum()
            
            if len(masses) > 0 and len(intensities) > 0:
                print(f"✅ Success with patches: {len(masses)} masses extracted")
            else:
                print("❌ Patches didn't resolve the issue")
        except Exception as e:
            print(f"❌ Still failing with patches: {e}")
    
    # Create and test with mock data as fallback
    mock_tsv_path = test_with_mock_data()
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    print("1. Your IONTOF files appear to be from a newer version/configuration")
    print("   than what pySPM 0.6.3 supports.")
    print("2. You may need to:")
    print("   - Export data from IONTOF software in a different format")
    print("   - Use IONTOF's built-in export to ASCII/CSV")
    print("   - Contact IONTOF support for pySPM compatibility")
    print(f"3. For now, you can test the PCA pipeline with: {mock_tsv_path}")
    print("4. The adapters CLI structure is working correctly")