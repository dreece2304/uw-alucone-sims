#!/usr/bin/env python3
"""
ITM Spatial Data Reader
======================

Extract spatial imaging data from ITM files with missing metadata blocks.
Based on pySPM ITM source code analysis.
"""

import sys
sys.path.insert(0, 'pySPM_source')
import pySPM
import numpy as np
import struct
from pathlib import Path

class ITMSpatialReader:
    """Read spatial data from ITM files with missing metadata blocks."""
    
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.itm = None
        self.spatial_data = None
        
    def load_with_fallbacks(self):
        """Load ITM with manual fallbacks for missing blocks."""
        try:
            # Try normal loading first
            self.itm = pySPM.ITM(str(self.file_path))
            print("✓ ITM loaded with standard method")
            return True
            
        except Exception as e:
            if "Missing block" in str(e) and "ShotsPerPixel" in str(e):
                print("⚠ Missing ShotsPerPixel block, trying manual initialization...")
                return self._manual_init()
            else:
                print(f"❌ ITM loading failed: {e}")
                return False
    
    def _manual_init(self):
        """Manually initialize ITM object with fallbacks."""
        try:
            # Create ITM object without calling __init__
            self.itm = object.__new__(pySPM.ITM)
            
            # Basic file setup
            self.itm.filename = str(self.file_path)
            self.itm.label = self.file_path.name
            self.itm.f = open(str(self.file_path), 'rb')
            self.itm.Type = self.itm.f.read(8)
            
            if self.itm.Type != b'ITStrF01':
                raise ValueError(f"Invalid ITM file: {self.file_path}")
            
            # Initialize block structure
            from pySPM.Block import Block
            self.itm.root = Block(self.itm.f)
            
            # Set defaults for missing attributes (from source code)
            self.itm.size = {
                "pixels": {"x": 256, "y": 256}, 
                "real": {"x": 500e-6, "y": 500e-6, "unit": "m"}
            }
            self.itm.polarity = "Positive"  # Default
            self.itm.peaks = {}
            self.itm.meas_data = {}
            self.itm.rawlist = None
            self.itm.scale = 1
            
            # Try to get actual values where possible
            try:
                # Try to get size from Meta/SI Image
                d = self.itm.root.goto("Meta/SI Image").dict_list()
                self.itm.size = {
                    "pixels": {"x": d["res_x"]["long"], "y": d["res_y"]["long"]},
                    "real": {
                        "x": d["fieldofview"]["float"],
                        "y": d["fieldofview"]["float"] * d["res_y"]["long"] / d["res_x"]["long"],
                        "unit": "m",
                    },
                }
                print(f"✓ Got size info: {self.itm.size}")
            except:
                # Try alternative size methods
                try:
                    s = self.itm.get_value("Registration.Raster.Resolution")["int"]
                    fov = self.itm.get_value("Registration.Raster.FieldOfView")["float"]
                    self.itm.size = {
                        "pixels": {"x": s, "y": s},
                        "real": {"x": fov, "y": fov, "unit": "m"},
                    }
                    print(f"✓ Got size from Registration: {self.itm.size}")
                except:
                    print("⚠ Using default size parameters")
            
            # Get polarity
            try:
                self.itm.polarity = self.itm.get_value("Instrument.Analyzer_Polarity_Switch")["string"]
            except:
                print("⚠ Using default polarity: Positive")
            
            # Get number of scans with fallbacks (from source code)
            self.itm.Nscan = None
            scan_paths = [
                "filterdata/TofCorrection/ImageStack/Reduced Data/NumberOfScans",
                "propend/Measurement.ScanNumber",
                "propstart/Measurement.ScanNumber"
            ]
            
            for path in scan_paths:
                try:
                    if "NumberOfScans" in path:
                        self.itm.Nscan = self.itm.root.goto(path).get_long()
                    else:
                        self.itm.Nscan = self.itm.root.goto(path).get_key_value()["int"]
                    print(f"✓ Found Nscan = {self.itm.Nscan} at {path}")
                    break
                except:
                    continue
            
            if self.itm.Nscan is None:
                self.itm.Nscan = 1
                print("⚠ Using default Nscan = 1")
            
            # Shots per pixel - the problematic parameter
            self.itm.spp = 1  # Default fallback
            print("⚠ Using default spp = 1")
            
            # Mass calibration
            try:
                self.itm.sf, self.itm.k0 = self.itm.get_mass_cal()
                print(f"✓ Got mass calibration: sf={self.itm.sf}, k0={self.itm.k0}")
            except:
                self.itm.sf, self.itm.k0 = 72000, 0
                print("⚠ Using default mass calibration")
            
            print("✓ ITM manually initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Manual ITM initialization failed: {e}")
            return False
    
    def extract_spatial_data(self, mass_ranges=None, scan=0):
        """Extract spatial data for specific mass ranges."""
        if self.itm is None:
            print("❌ ITM not loaded")
            return None
        
        try:
            print(f"Extracting spatial data from scan {scan}...")
            
            # Get pixel-by-pixel raw data
            pixel_data = self.itm.get_raw_data(scan)
            print(f"✓ Got pixel data for {len(pixel_data)} pixels")
            
            # Get image dimensions
            nx, ny = self.itm.size["pixels"]["x"], self.itm.size["pixels"]["y"]
            print(f"Image dimensions: {nx} × {ny}")
            
            if mass_ranges is None:
                # Default ranges for common elements
                mass_ranges = [
                    (26.5, 27.5),  # Al
                    (11.5, 12.5),  # C
                    (0.8, 1.2),    # H
                    (15.5, 16.5),  # O
                ]
            
            spatial_images = {}
            
            for i, (low_mass, high_mass) in enumerate(mass_ranges):
                print(f"Processing mass range {low_mass:.1f}-{high_mass:.1f} u...")
                
                # Convert masses to time channels
                from pySPM.utils import mass2time
                low_time = mass2time(low_mass, sf=self.itm.sf, k0=self.itm.k0)
                high_time = mass2time(high_mass, sf=self.itm.sf, k0=self.itm.k0)
                
                # Create spatial image
                image = np.zeros((ny, nx), dtype=np.float32)
                
                for (x, y), times in pixel_data.items():
                    if 0 <= x < nx and 0 <= y < ny:
                        # Count ions in this mass range
                        count = np.sum((np.array(times) >= low_time) & (np.array(times) <= high_time))
                        image[y, x] = count
                
                spatial_images[f"{low_mass:.1f}-{high_mass:.1f}u"] = {
                    'image': image,
                    'mass_range': (low_mass, high_mass),
                    'total_counts': np.sum(image),
                    'max_counts': np.max(image)
                }
                
                print(f"  Total counts: {np.sum(image):.0f}, Max: {np.max(image):.0f}")
            
            self.spatial_data = spatial_images
            return spatial_images
            
        except Exception as e:
            print(f"❌ Spatial data extraction failed: {e}")
            return None
    
    def get_total_ion_image(self, scan=0):
        """Get total ion count image."""
        if self.itm is None:
            return None
            
        try:
            pixel_data = self.itm.get_raw_data(scan)
            nx, ny = self.itm.size["pixels"]["x"], self.itm.size["pixels"]["y"]
            
            image = np.zeros((ny, nx), dtype=np.float32)
            
            for (x, y), times in pixel_data.items():
                if 0 <= x < nx and 0 <= y < ny:
                    image[y, x] = len(times)  # Total ion count
            
            return image
            
        except Exception as e:
            print(f"❌ Total ion image extraction failed: {e}")
            return None
    
    def save_spatial_data(self, output_dir):
        """Save spatial images as numpy arrays."""
        if self.spatial_data is None:
            print("❌ No spatial data to save")
            return False
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        sample_name = self.file_path.stem
        
        for mass_range, data in self.spatial_data.items():
            filename = f"{sample_name}_{mass_range.replace('.', 'p').replace('-', '_')}.npz"
            filepath = output_dir / filename
            
            np.savez_compressed(
                filepath,
                image=data['image'],
                mass_range=data['mass_range'],
                sample_info={
                    'file': str(self.file_path),
                    'size': self.itm.size,
                    'polarity': self.itm.polarity,
                    'scans': self.itm.Nscan
                }
            )
            
            print(f"✓ Saved spatial data: {filepath}")
        
        return True

def test_itm_spatial_reader():
    """Test the ITM spatial reader on a sample file."""
    test_file = Path("data/PositiveIonData/P1_SQ1_01.itm")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"Testing ITM spatial reader on: {test_file}")
    print("="*60)
    
    reader = ITMSpatialReader(test_file)
    
    # Load ITM file
    if not reader.load_with_fallbacks():
        return False
    
    # Extract spatial data
    spatial_data = reader.extract_spatial_data()
    
    if spatial_data:
        print("\n✅ SUCCESS! Spatial data extracted")
        print(f"Mass ranges processed: {list(spatial_data.keys())}")
        
        # Save data
        reader.save_spatial_data("test_spatial_output")
        return True
    else:
        print("\n❌ FAILED to extract spatial data")
        return False

if __name__ == "__main__":
    success = test_itm_spatial_reader()
    exit(0 if success else 1)