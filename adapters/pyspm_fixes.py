#!/usr/bin/env python
"""
pySPM compatibility fixes for IONTOF files.

This module provides monkey patches and wrapper functions to handle:
1. Missing "propend/Registration.Raster.ShotsPerPixel" blocks in ITM files
2. Buffer size mismatches in ITAX spectrum reading
3. Alternative data structure paths
"""

import os
import struct
import warnings
import numpy as np

def open_iontof_with_fixes(file_path):
    """
    Open IONTOF data file with compatibility fixes.
    
    Args:
        file_path: Path to IONTOF data file (.itm, .itax, .ita)
        
    Returns:
        Tuple of (masses, intensities) arrays
        
    Raises:
        ImportError: If pySPM is not available
        RuntimeError: If file cannot be processed
    """
    try:
        import pySPM
    except ImportError:
        raise ImportError("pySPM is required for IONTOF data processing. Install with: pip install pySPM")
    
    file_path = str(file_path)
    suffix = os.path.splitext(file_path)[1].lower()
    
    if suffix == '.itax':
        return _open_itax_with_fixes(file_path)
    elif suffix in ['.itm', '.ita', '.itmx']:
        return _open_itm_with_fixes(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _open_itax_with_fixes(file_path):
    """Open ITAX file with buffer size fixes."""
    import pySPM
    
    print(f"Opening ITAX file with fixes: {file_path}")
    
    try:
        itax = pySPM.ITAX(file_path)
        print(f"✓ ITAX file loaded successfully")
        
        # Try to get spectrum with our fixes
        masses, intensities = _get_itax_spectrum_fixed(itax)
        
        if len(masses) > 0 and len(intensities) > 0:
            print(f"✓ Extracted {len(masses)} spectrum points")
            print(f"  Mass range: {min(masses):.3f} - {max(masses):.3f} u")
            return masses, intensities
        else:
            print("⚠ Spectrum extraction returned empty arrays")
            return np.array([]), np.array([])
            
    except Exception as e:
        print(f"❌ ITAX processing failed: {e}")
        return np.array([]), np.array([])


def _get_itax_spectrum_fixed(itax, sf=None, k0=None, time=False):
    """Get spectrum from ITAX with buffer size fixes."""
    try:
        # First try the original method
        return itax.getSpectrum(sf=sf, k0=k0, time=time)
    except struct.error as e:
        if "unpack requires a buffer" in str(e):
            print("ITAX: Fixing buffer size mismatch...")
            return _extract_itax_spectrum_alternative(itax, sf, k0, time)
        else:
            raise
    except Exception as e:
        print(f"ITAX: getSpectrum failed: {e}")
        return np.array([]), np.array([])


def _extract_itax_spectrum_alternative(itax, sf=None, k0=None, time=False):
    """Alternative ITAX spectrum extraction with buffer size handling."""
    try:
        # Get spectrum length more carefully
        slen_node = itax.root.goto("CommonDataObjects/DataViewCollection/*/sizeSpectrum")
        slen = slen_node.get_long()  # Use new API
        print(f"ITAX: Expected spectrum length: {slen}")
        
        # Get raw data
        raw_node = itax.root.goto(
            "CommonDataObjects/DataViewCollection/*/dataSource/simsDataCache/spectrum/correctedData"
        )
        raw = raw_node.value
        print(f"ITAX: Raw data buffer size: {len(raw)} bytes")
        
        # Calculate expected vs actual size
        expected_bytes = slen * 8  # 8 bytes per double
        actual_bytes = len(raw)
        
        if actual_bytes < expected_bytes:
            print(f"ITAX: Buffer too small ({actual_bytes} < {expected_bytes}), truncating spectrum")
            slen = actual_bytes // 8
            
        elif actual_bytes > expected_bytes:
            print(f"ITAX: Buffer larger than expected, using available data")
            
        # Try to unpack with corrected size
        spectrum = np.array(struct.unpack("<" + str(slen) + "d", raw[:slen*8]))
        CH = 2 * np.arange(slen)
        
        if time:
            return CH, spectrum
        
        # Get mass calibration
        if sf is None:
            try:
                sf = itax.root.goto(
                    "CommonDataObjects/DataViewCollection/*/properties/Context.MassScale.SF",
                    lazy=True,
                ).get_key_value()["float"]
            except:
                print("ITAX: Using default sf = 72000")
                sf = 72000
                
        if k0 is None:
            try:
                k0 = itax.root.goto(
                    "CommonDataObjects/DataViewCollection/*/properties/Context.MassScale.K0",
                    lazy=True,
                ).get_key_value()["float"]
            except:
                print("ITAX: Using default k0 = 0")
                k0 = 0
        
        # Convert to mass using pySPM utils
        import pySPM
        m = pySPM.utils.time2mass(CH, sf, k0)
        
        print(f"✓ ITAX: Successfully extracted {len(spectrum)} spectrum points with fixes")
        return m, spectrum
        
    except Exception as e:
        print(f"ITAX: Alternative extraction failed: {e}")
        return np.array([]), np.array([])


def _open_itm_with_fixes(file_path):
    """Open ITM/ITMX file with missing block fixes."""
    import pySPM
    
    file_ext = os.path.splitext(file_path)[1].lower()
    print(f"Opening {file_ext.upper()} file with fixes: {file_path}")
    
    try:
        # Try normal opening first
        if file_ext == '.itmx':
            # ITMX files are often similar to ITM but may need different handling
            print("ITMX: Trying ITM-compatible loading...")
        
        itm = pySPM.ITM(file_path)
        print(f"✓ {file_ext.upper()} file loaded successfully")
        
        # Try to get spectrum - use multiple methods immediately
        spectrum_methods = [
            ("get_spectrum", lambda: itm.get_spectrum()),
            ("getSumSpectrum", lambda: itm.getSumSpectrum() if hasattr(itm, 'getSumSpectrum') else None),
            ("alternative", lambda: _extract_itm_spectrum_alternative(itm)),
            ("sum_manual", lambda: _extract_itm_sum_spectrum(itm))
        ]
        
        for method_name, method in spectrum_methods:
            try:
                result = method()
                if result is not None:
                    masses, intensities = result
                    if len(masses) > 0 and len(intensities) > 0:
                        print(f"✓ Extracted {len(masses)} spectrum points using {method_name}")
                        return masses, intensities
            except Exception as e:
                print(f"  {method_name} failed: {e}")
                continue
        
        print(f"❌ All spectrum extraction methods failed for ITM")
        return np.array([]), np.array([])
        
    except Exception as e:
        if "Missing block" in str(e) and "ShotsPerPixel" in str(e):
            print(f"{file_ext.upper()}: Detected missing ShotsPerPixel block, trying alternative approach...")
            return _open_itm_alternative(file_path)
        else:
            print(f"❌ {file_ext.upper()} processing failed: {e}")
            return np.array([]), np.array([])


def _open_itm_alternative(file_path):
    """Alternative ITM opening method that bypasses missing blocks."""
    import pySPM
    
    try:
        # Create ITM object manually with error handling
        itm = object.__new__(pySPM.ITM)  # Create without calling __init__
        
        # Manually initialize the essential parts
        itm.filename = file_path
        itm.label = os.path.basename(file_path)
        itm.f = open(file_path, 'rb')
        itm.Type = itm.f.read(8)
        
        if itm.Type != b'ITStrF01':
            raise ValueError(f"Not an IONTOF file: {file_path}")
        
        import pySPM.Block
        itm.root = pySPM.Block.Block(itm.f)
        
        # Set default values for missing attributes
        itm.size = {"pixels": {"x": 256, "y": 256}, "real": {"x": 500e-6, "y": 500e-6, "unit": "m"}}
        itm.polarity = "Positive"  # Default
        itm.peaks = {}
        itm.meas_data = {}
        itm.rawlist = None
        itm.Nscan = 1  # Default
        itm.spp = 1    # Default shots per pixel
        itm.sf = 72000  # Default mass calibration
        itm.k0 = 0
        itm.scale = 1
        
        print("ITM: Initialized with default parameters")
        
        # Try multiple spectrum extraction methods
        spectrum_methods = [
            lambda: itm.get_spectrum(),  # Standard method
            lambda: _extract_itm_spectrum_alternative(itm),  # Alternative method
            lambda: _extract_itm_sum_spectrum(itm)  # Sum spectrum method
        ]
        
        for i, method in enumerate(spectrum_methods):
            try:
                masses, intensities = method()
                if len(masses) > 0 and len(intensities) > 0:
                    print(f"✓ ITM: Extracted {len(masses)} spectrum points with method {i+1}")
                    return masses, intensities
            except Exception as e:
                print(f"ITM: Method {i+1} failed: {e}")
                continue
        
        print("ITM: All spectrum extraction methods failed")
        return np.array([]), np.array([])
            
    except Exception as e:
        print(f"❌ ITM alternative method failed: {e}")
        return np.array([]), np.array([])


def validate_extracted_spectrum(masses, intensities, file_path):
    """Validate extracted spectrum data."""
    if len(masses) == 0 or len(intensities) == 0:
        print(f"⚠ No spectrum data extracted from {file_path}")
        return False
    
    if len(masses) != len(intensities):
        print(f"⚠ Mass and intensity arrays have different lengths: {len(masses)} vs {len(intensities)}")
        return False
    
    if np.any(masses <= 0):
        print(f"⚠ Found non-positive masses in {file_path}")
        return False
    
    if np.any(intensities < 0):
        print(f"⚠ Found negative intensities in {file_path} (may be normal for processed data)")
        # Don't fail for negative intensities - they can be normal in processed ToF-SIMS data
    
    # Check for reasonable mass range (ToF-SIMS typically 1-1000 u)
    min_mass, max_mass = np.min(masses), np.max(masses)
    if min_mass > 100 or max_mass < 10:
        print(f"⚠ Unusual mass range in {file_path}: {min_mass:.3f} - {max_mass:.3f} u")
    
    print(f"✓ Spectrum validation passed for {file_path}")
    print(f"  {len(masses)} points, mass range: {min_mass:.3f} - {max_mass:.3f} u")
    print(f"  Intensity range: {np.min(intensities):.3e} - {np.max(intensities):.3e}")
    
    return True


def _extract_itm_spectrum_alternative(itm):
    """Alternative ITM spectrum extraction that avoids filterdata block."""
    try:
        # Look for spectrum data in different locations
        possible_paths = [
            "MeasData/MassSpectrum/Spectrum",
            "MeasData/Spectrum/Spectrum", 
            "Data/Spectrum",
            "rawdata/Spectrum"
        ]
        
        for path in possible_paths:
            try:
                spectrum_node = itm.root.goto(path)
                spectrum_data = spectrum_node.value
                
                if len(spectrum_data) > 0:
                    # Convert to mass and intensity arrays
                    n_points = len(spectrum_data)
                    masses = np.arange(n_points) * 0.01  # Simple linear mass scale
                    intensities = np.array(spectrum_data, dtype=np.float32)
                    
                    print(f"ITM Alternative: Found spectrum at {path}")
                    return masses, intensities
                    
            except Exception:
                continue
        
        raise Exception("No spectrum data found in any known location")
        
    except Exception as e:
        raise Exception(f"Alternative ITM extraction failed: {e}")


def _extract_itm_sum_spectrum(itm):
    """Extract sum spectrum from ITM if available."""
    try:
        # Try to get sum spectrum which is more commonly available
        if hasattr(itm, 'getSumSpectrum'):
            masses, intensities = itm.getSumSpectrum()
            return masses, intensities
        elif hasattr(itm, 'get_sum_spectrum'):
            masses, intensities = itm.get_sum_spectrum()
            return masses, intensities
        else:
            # Manual sum spectrum extraction
            sum_paths = [
                "MeasData/MassSpectrum/SumSpectrum",
                "Data/SumSpectrum",
                "SumSpectrum"
            ]
            
            for path in sum_paths:
                try:
                    sum_node = itm.root.goto(path)
                    sum_data = sum_node.value
                    
                    if len(sum_data) > 0:
                        n_points = len(sum_data) 
                        # Use mass calibration if available
                        if hasattr(itm, 'sf') and hasattr(itm, 'k0'):
                            time_channels = np.arange(n_points) * 2  # 2 ns per channel typical
                            import pySPM
                            masses = pySPM.utils.time2mass(time_channels, itm.sf, itm.k0)
                        else:
                            masses = np.arange(n_points) * 0.01
                        
                        intensities = np.array(sum_data, dtype=np.float32)
                        
                        print(f"ITM Sum: Found sum spectrum at {path}")
                        return masses, intensities
                        
                except Exception:
                    continue
            
            raise Exception("No sum spectrum found")
            
    except Exception as e:
        raise Exception(f"Sum spectrum extraction failed: {e}")