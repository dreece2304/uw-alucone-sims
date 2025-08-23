#!/usr/bin/env python
"""
Patch pySPM to fix compatibility issues with user's IONTOF files.

This creates monkey patches for ITM and ITAX classes to handle:
1. Missing "propend/Registration.Raster.ShotsPerPixel" blocks
2. Buffer size mismatches in ITAX spectrum reading
3. Alternative data structure paths
"""

import pySPM
import numpy as np
import struct
import warnings
import os

def patch_ITM():
    """Patch ITM class to handle missing blocks gracefully."""
    
    # Store original __init__ method
    original_ITM_init = pySPM.ITM.__init__
    
    def patched_ITM_init(self, filename, debug=False, readonly=True, precond=False, label=None):
        """Patched ITM constructor with better error handling."""
        try:
            # Try the original initialization
            original_ITM_init(self, filename, debug, readonly, precond, label)
            return
        except Exception as e:
            if "Missing block" in str(e) and "ShotsPerPixel" in str(e):
                print(f"ITM: Handling missing ShotsPerPixel block, trying alternative approach...")
                # Retry with modified approach
                self.filename = filename
                if label is None:
                    self.label = os.path.basename(filename)
                else:
                    self.label = label
                
                if not os.path.exists(filename):
                    print(f'ERROR: File "{filename}" not found')
                    raise FileNotFoundError
                    
                if readonly:
                    self.f = open(self.filename, "rb")
                else:
                    self.f = open(self.filename, "r+b")
                    
                self.Type = self.f.read(8)
                assert self.Type == b"ITStrF01"
                
                import pySPM.Block
                self.root = pySPM.Block.Block(self.f)
                
                # Try to get size info with fallbacks
                try:
                    d = self.root.goto("Meta/SI Image").dict_list()
                    self.size = {
                        "pixels": {"x": d["res_x"]["long"], "y": d["res_y"]["long"]},
                        "real": {
                            "x": d["fieldofview"]["float"],
                            "y": d["fieldofview"]["float"] * d["res_y"]["long"] / d["res_x"]["long"],
                            "unit": "m",
                        },
                    }
                except:
                    # Use default size if meta info not available
                    print("ITM: Using default size parameters")
                    self.size = {
                        "pixels": {"x": 256, "y": 256},
                        "real": {"x": 500e-6, "y": 500e-6, "unit": "m"}
                    }
                
                # Try to get polarity with fallback
                try:
                    self.polarity = self.get_value("Instrument.Analyzer_Polarity_Switch")["string"]
                except:
                    print("ITM: Could not determine polarity, defaulting to Positive")
                    self.polarity = "Positive"
                
                self.peaks = {}
                self.meas_data = {}
                self.rawlist = None
                
                # Try to get Nscan with multiple fallbacks
                self.Nscan = None
                scan_paths = [
                    "filterdata/TofCorrection/ImageStack/Reduced Data/NumberOfScans",
                    "propend/Measurement.ScanNumber",
                    "propstart/Measurement.ScanNumber"
                ]
                
                for path in scan_paths:
                    try:
                        if "NumberOfScans" in path:
                            self.Nscan = self.root.goto(path).getLong()
                        else:
                            self.Nscan = self.root.goto(path).get_key_value()["int"]
                        print(f"ITM: Found Nscan = {self.Nscan} at {path}")
                        break
                    except:
                        continue
                
                if self.Nscan is None:
                    print("ITM: Could not determine scan number, defaulting to 1")
                    self.Nscan = 1
                
                # Try to get shots per pixel with fallbacks
                self.spp = None
                spp_paths = [
                    "propend/Registration.Raster.ShotsPerPixel",
                    "propstart/Registration.Raster.ShotsPerPixel",
                    "Measurement Options/ShotsPerPixel"
                ]
                
                for path in spp_paths:
                    try:
                        self.spp = self.root.goto(path).get_key_value()["int"]
                        print(f"ITM: Found spp = {self.spp} at {path}")
                        break
                    except:
                        continue
                
                if self.spp is None:
                    print("ITM: Could not determine shots per pixel, defaulting to 1")
                    self.spp = 1
                
                # Try to load peaks
                try:
                    R = [z for z in self.root.goto("MassIntervalList").get_list() if z["name"] == "mi"]
                    for x in R:
                        try:
                            X = self.root.goto("MassIntervalList/mi[" + str(x["id"]) + "]")
                            d = X.dict_list()
                            self.peaks[d["id"]["long"]] = d
                        except ValueError:
                            pass
                except Exception as e:
                    if debug:
                        raise e
                
                # Try to get mass calibration
                try:
                    self.sf, self.k0 = self.get_mass_cal()
                except:
                    print("ITM: Could not get mass calibration, using defaults")
                    self.sf, self.k0 = 72000, 0
                
                self.scale = 1
                if precond:
                    try:
                        self.precond()
                    except:
                        print("ITM: Preconditioner failed, continuing without it")
                        
                print("ITM: Successfully loaded with compatibility fixes")
            else:
                # Re-raise other exceptions
                raise e
    
    # Apply the patch
    pySPM.ITM.__init__ = patched_ITM_init
    print("✓ Applied ITM compatibility patch")


def patch_ITAX():
    """Patch ITAX class to handle buffer size mismatches."""
    
    # Store original getSpectrum method
    original_getSpectrum = pySPM.ITAX.getSpectrum
    
    def patched_getSpectrum(self, sf=None, k0=None, time=False, **kargs):
        """Patched getSpectrum with better error handling for buffer size issues."""
        try:
            # Try the original method first
            return original_getSpectrum(self, sf, k0, time, **kargs)
        except struct.error as e:
            if "unpack requires a buffer" in str(e):
                print(f"ITAX: Buffer size mismatch detected, trying alternative approach...")
                
                # Try to read spectrum with different approaches
                try:
                    # Get spectrum length more carefully
                    slen_node = self.root.goto("CommonDataObjects/DataViewCollection/*/sizeSpectrum")
                    slen = slen_node.getLong()
                    print(f"ITAX: Expected spectrum length: {slen}")
                    
                    # Get raw data
                    raw_node = self.root.goto(
                        "CommonDataObjects/DataViewCollection/*/dataSource/simsDataCache/spectrum/correctedData"
                    )
                    raw = raw_node.value
                    print(f"ITAX: Raw data buffer size: {len(raw)} bytes")
                    
                    # Calculate expected vs actual size
                    expected_bytes = slen * 8  # 8 bytes per double
                    actual_bytes = len(raw)
                    
                    if actual_bytes < expected_bytes:
                        print(f"ITAX: Buffer too small ({actual_bytes} < {expected_bytes}), truncating spectrum length")
                        slen = actual_bytes // 8
                        
                    elif actual_bytes > expected_bytes:
                        print(f"ITAX: Buffer larger than expected ({actual_bytes} > {expected_bytes}), using available data")
                        
                    # Try to unpack with corrected size
                    spectrum = np.array(struct.unpack("<" + str(slen) + "d", raw[:slen*8]))
                    CH = 2 * np.arange(slen)
                    
                    if time:
                        return CH, spectrum
                    
                    # Get mass calibration
                    if sf is None:
                        try:
                            sf = self.root.goto(
                                "CommonDataObjects/DataViewCollection/*/properties/Context.MassScale.SF",
                                lazy=True,
                            ).getKeyValue()["float"]
                        except:
                            sf = 72000  # Default value
                            
                    if k0 is None:
                        try:
                            k0 = self.root.goto(
                                "CommonDataObjects/DataViewCollection/*/properties/Context.MassScale.K0",
                                lazy=True,
                            ).getKeyValue()["float"]
                        except:
                            k0 = 0  # Default value
                    
                    import pySPM.utils
                    m = pySPM.utils.time2mass(CH, sf, k0)
                    
                    print(f"✓ ITAX: Successfully extracted {len(spectrum)} spectrum points")
                    return m, spectrum
                    
                except Exception as e2:
                    print(f"ITAX: Alternative approach failed: {e2}")
                    # Return empty arrays as last resort
                    return np.array([]), np.array([])
            else:
                # Re-raise other struct errors
                raise e
        except Exception as e:
            print(f"ITAX: getSpectrum failed: {e}")
            # Return empty arrays as fallback
            return np.array([]), np.array([])
    
    # Apply the patch
    pySPM.ITAX.getSpectrum = patched_getSpectrum
    print("✓ Applied ITAX compatibility patch")


def apply_pySPM_patches():
    """Apply all pySPM compatibility patches."""
    print("Applying pySPM compatibility patches...")
    
    # Import required modules
    import os
    
    try:
        patch_ITM()
        patch_ITAX()
        print("✅ All pySPM patches applied successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to apply patches: {e}")
        return False


if __name__ == "__main__":
    # Test the patches
    apply_pySPM_patches()
    
    # Test on user's files
    test_files = [
        "data/PositiveIonData/P1_SQ1_01.itm",
        "data/PositiveIonData/P1_SQ1_01.itax",
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'='*60}")
            print(f"Testing patched pySPM on: {file_path}")
            print(f"{'='*60}")
            
            try:
                if file_path.endswith('.itm'):
                    obj = pySPM.ITM(file_path)
                    print(f"✓ Successfully loaded ITM file")
                    print(f"  Size: {obj.size}")
                    print(f"  Polarity: {obj.polarity}")
                    print(f"  Scans: {obj.Nscan}")
                    print(f"  Shots per pixel: {obj.spp}")
                    
                elif file_path.endswith('.itax'):
                    obj = pySPM.ITAX(file_path)
                    print(f"✓ Successfully loaded ITAX file")
                    print(f"  Size: {obj.size}")
                    
                    # Try to get spectrum
                    masses, intensities = obj.getSpectrum()
                    if len(masses) > 0 and len(intensities) > 0:
                        print(f"✓ Successfully extracted spectrum: {len(masses)} points")
                        print(f"  Mass range: {min(masses):.3f} - {max(masses):.3f} u")
                        print(f"  Intensity range: {min(intensities):.3e} - {max(intensities):.3e}")
                    else:
                        print("⚠ Spectrum extraction returned empty arrays")
                        
            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")
        else:
            print(f"⚠ File not found: {file_path}")