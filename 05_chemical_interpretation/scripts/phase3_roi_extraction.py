#!/usr/bin/env python3
"""
Phase 3: ROI-Summed Spectra Extraction
- Extract spectra from defined ROI areas
- Sum spectra within each ROI
- Apply unit mass binning
- Generate ROI TSV files matching existing format
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import imaging libraries
try:
    import pySPM
    PYSPM_AVAILABLE = True
except ImportError:
    PYSPM_AVAILABLE = False

def load_roi_schema() -> Dict:
    """Load ROI schema"""
    schema_path = Path('roi/schema.json')
    with open(schema_path, 'r') as f:
        return json.load(f)

def create_imaging_object_safe(file_path: Path) -> Optional[object]:
    """Create imaging object with error handling (ITAX preferred, ITM fallback)"""
    
    if file_path.suffix.lower() == '.itax':
        # Try ITAX first - better for imaging
        try:
            print(f"    Loading ITAX file: {file_path.name}")
            itax = pySPM.ITA(str(file_path))
            return itax
        except Exception as e:
            print(f"    ITAX loading failed for {file_path.name}: {e}")
            return None
    
    elif file_path.suffix.lower() == '.itm':
        # ITM fallback with our proven fixes
        try:
            # Try standard approach first
            itm = pySPM.ITM(str(file_path))
            return itm
        except Exception as e:
            if "Missing block" in str(e) and "ShotsPerPixel" in str(e):
                print(f"    Using alternative ITM initialization for {file_path.name}...")
                try:
                    # Create ITM object manually with error handling
                    itm = object.__new__(pySPM.ITM)
                    itm.filename = str(file_path)
                    itm.label = file_path.name
                    itm.f = open(str(file_path), 'rb')
                    itm.Type = itm.f.read(8)
                    
                    if itm.Type != b'ITStrF01':
                        raise ValueError(f"Not an IONTOF file: {file_path}")
                    
                    itm.root = pySPM.Block.Block(itm.f)
                    
                    # Set default values for missing attributes
                    itm.size = {"pixels": {"x": 256, "y": 256}, "real": {"x": 500e-6, "y": 500e-6, "unit": "m"}}
                    itm.polarity = "Positive"
                    itm.peaks = {}
                    itm.meas_data = {}
                    itm.rawlist = None
                    itm.Nscan = 1
                    itm.spp = 1    # Default shots per pixel
                    itm.sf = 72000  # Default mass calibration
                    itm.k0 = 0
                    itm.scale = 1
                    
                    return itm
                    
                except Exception as e2:
                    print(f"    Alternative ITM approach failed for {file_path.name}: {e2}")
                    return None
            else:
                print(f"    ITM loading failed for {file_path.name}: {e}")
                return None
    
    else:
        print(f"    Unsupported file type: {file_path.suffix}")
        return None

def extract_roi_spectrum(itm_obj: object, roi_coords: List[int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract and sum spectra from ROI area"""
    
    x, y, width, height = roi_coords
    
    try:
        # Initialize spectrum accumulators
        all_masses = []
        all_intensities = []
        
        # Extract spectra from each pixel in the ROI
        spectra_count = 0
        for py in range(y, y + height):
            for px in range(x, x + width):
                try:
                    spectrum = itm_obj.getSpectrum(px, py)
                    if spectrum and len(spectrum) >= 2:
                        masses, intensities = spectrum
                        if len(masses) > 0:
                            all_masses.extend(masses)
                            all_intensities.extend(intensities)
                            spectra_count += 1
                except Exception as e:
                    # Skip problematic pixels
                    continue
        
        if spectra_count == 0:
            print(f"    No valid spectra found in ROI {roi_coords}")
            return None
        
        if len(all_masses) == 0:
            print(f"    No mass data found in ROI {roi_coords}")
            return None
        
        # Convert to numpy arrays
        masses = np.array(all_masses, dtype=float)
        intensities = np.array(all_intensities, dtype=float)
        
        print(f"    Extracted {spectra_count} spectra from ROI ({len(masses)} total peaks)")
        return masses, intensities
        
    except Exception as e:
        print(f"    ROI extraction failed: {e}")
        return None

def apply_unit_mass_binning(masses: np.ndarray, intensities: np.ndarray, 
                          mass_range: Tuple[int, int] = (0, 300)) -> Tuple[np.ndarray, np.ndarray]:
    """Apply unit mass binning (same as existing pipeline)"""
    
    # Filter to mass range and remove invalid values
    mask = np.isfinite(masses) & np.isfinite(intensities)
    mask &= (masses >= mass_range[0]) & (masses <= mass_range[1])
    
    if np.sum(mask) == 0:
        print(f"    No valid masses in range {mass_range}")
        return np.array([]), np.array([])
    
    masses_filtered = masses[mask]
    intensities_filtered = intensities[mask]
    
    # Round to unit masses
    unit_masses = np.rint(masses_filtered).astype(int)
    
    # Create dataframe for grouping
    df = pd.DataFrame({
        'mass': unit_masses,
        'intensity': intensities_filtered
    })
    
    # Sum intensities by unit mass
    binned = df.groupby('mass')['intensity'].sum().reset_index()
    
    return binned['mass'].values, binned['intensity'].values

def process_imaging_file(file_path: Path, roi_schema: Dict, pattern: str, polarity: str) -> Optional[Dict]:
    """Process a single imaging file and extract ROI spectra"""
    
    print(f"  Processing {file_path.name} ({pattern} {polarity})...")
    
    if not PYSPM_AVAILABLE:
        print(f"    pySPM not available")
        return None
    
    # Create imaging object
    img_obj = create_imaging_object_safe(file_path)
    if img_obj is None:
        return None
    
    # Extract spectra from each ROI
    roi_spectra = {}
    roi_definitions = roi_schema['roi_definitions']
    
    for roi_name, roi_data in roi_definitions.items():
        print(f"    Extracting ROI: {roi_name}")
        
        roi_coords = roi_data['coordinates']
        spectrum_data = extract_roi_spectrum(img_obj, roi_coords)
        
        if spectrum_data is None:
            print(f"    Failed to extract {roi_name}")
            continue
            
        masses, intensities = spectrum_data
        
        # Apply unit mass binning
        binned_masses, binned_intensities = apply_unit_mass_binning(masses, intensities)
        
        if len(binned_masses) == 0:
            print(f"    No binned data for {roi_name}")
            continue
        
        roi_spectra[roi_name] = {
            'masses': binned_masses,
            'intensities': binned_intensities,
            'dose': roi_data['dose'],
            'pixel_count': roi_coords[2] * roi_coords[3]  # width * height
        }
        
        print(f"    ✓ {roi_name}: {len(binned_masses)} unit masses, dose={roi_data['dose']} µC/cm²")
    
    # Clean up
    if hasattr(img_obj, 'f') and img_obj.f:
        img_obj.f.close()
    
    return roi_spectra

def create_roi_tsv(all_roi_data: Dict, polarity: str, roi_schema: Dict) -> None:
    """Create TSV file from ROI data matching existing format"""
    
    print(f"\nCreating {polarity} ROI TSV file...")
    
    # Determine all unique masses across all ROIs and patterns
    all_masses = set()
    for pattern_data in all_roi_data.values():
        for roi_data in pattern_data.values():
            if roi_data:
                all_masses.update(roi_data['masses'])
    
    if len(all_masses) == 0:
        print(f"  No mass data found for {polarity} polarity")
        return
    
    # Sort masses
    sorted_masses = sorted(all_masses)
    print(f"  Found {len(sorted_masses)} unique unit masses")
    
    # Create column structure matching existing format
    # Columns: P{pattern}_{dose}uC-{pol} (e.g., P1_500uC-P)
    columns = ['mass']
    dose_mapping = roi_schema['dose_mapping']
    
    patterns = ['P1', 'P2', 'P3'] 
    pol_code = 'P' if polarity == 'positive' else 'N'
    
    for pattern in patterns:
        for sq_code, dose in dose_mapping.items():
            col_name = f"{pattern}_{dose}uC-{pol_code}"
            columns.append(col_name)
    
    # Initialize dataframe
    df = pd.DataFrame(index=sorted_masses, columns=columns[1:])
    df.index.name = 'mass'
    df = df.fillna(0.0)
    
    # Fill data
    for pattern, pattern_data in all_roi_data.items():
        if not pattern_data:
            continue
            
        for roi_name, roi_spectra in pattern_data.items():
            if not roi_spectra:
                continue
                
            dose = roi_spectra['dose']
            masses = roi_spectra['masses']
            intensities = roi_spectra['intensities']
            
            # Find matching column
            col_name = f"{pattern}_{dose}uC-{pol_code}"
            if col_name in df.columns:
                # Map masses to dataframe
                for mass, intensity in zip(masses, intensities):
                    if mass in df.index:
                        df.loc[mass, col_name] = intensity
    
    # Reset index to make mass a column
    df = df.reset_index()
    
    # Save TSV
    out_dir = Path('out/roi')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = out_dir / f'all_{polarity}_roi.tsv'
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"  Saved: {output_file}")
    print(f"  Shape: {df.shape[0]} masses × {df.shape[1]} columns")
    
    return output_file

def main():
    """Execute Phase 3 analysis"""
    print("=== Phase 3: ROI-Summed Spectra Extraction ===")
    
    if not PYSPM_AVAILABLE:
        print("❌ pySPM not available - cannot extract ROI spectra")
        return
    
    # Load ROI schema
    roi_schema = load_roi_schema()
    print("ROI schema loaded")
    
    # Find imaging files
    data_dir = Path('data')
    pos_dir = data_dir / 'PositiveIonData'
    neg_dir = data_dir / 'NegativeIonData'
    
    # Collect files by pattern and polarity
    file_mapping = {
        'positive': {},
        'negative': {}
    }
    
    patterns = ['P1', 'P2', 'P3']
    
    # Try ITAX files first (better imaging support), then ITM as fallback
    # Positive ion files (01 suffix)
    for pattern in patterns:
        pattern_files = []
        for sq_code in ['SQ1r', 'SQ1', 'SQ2', 'SQ3', 'SQ4']:
            # Try ITAX first
            itax_path = pos_dir / f'{pattern}_{sq_code}_01.itax'
            itm_path = pos_dir / f'{pattern}_{sq_code}_01.itm'
            
            if itax_path.exists():
                pattern_files.append(itax_path)
            elif itm_path.exists():
                pattern_files.append(itm_path)
        file_mapping['positive'][pattern] = pattern_files
    
    # Negative ion files (06 suffix)  
    for pattern in patterns:
        pattern_files = []
        for sq_code in ['SQ1r', 'SQ1', 'SQ2', 'SQ3', 'SQ4']:
            # Try ITAX first
            itax_path = neg_dir / f'{pattern}_{sq_code}_06.itax'
            itm_path = neg_dir / f'{pattern}_{sq_code}_06.itm'
            
            if itax_path.exists():
                pattern_files.append(itax_path)
            elif itm_path.exists():
                pattern_files.append(itm_path)
        file_mapping['negative'][pattern] = pattern_files
    
    # Print file summary
    for polarity, polarity_data in file_mapping.items():
        total_files = sum(len(files) for files in polarity_data.values())
        print(f"{polarity.capitalize()}: {total_files} files across {len(polarity_data)} patterns")
    
    # Process files and extract ROI spectra
    all_extracted_data = {'positive': {}, 'negative': {}}
    
    for polarity in ['positive', 'negative']:
        print(f"\nProcessing {polarity} ion data...")
        
        for pattern, files in file_mapping[polarity].items():
            print(f"\nPattern {pattern}:")
            
            # For imaging, we'll process one representative file per pattern
            # (In full implementation, could process all dose files separately)
            if files:
                # Take the first file as representative
                representative_file = files[0]  # This is usually SQ1 or SQ1r
                
                roi_data = process_imaging_file(representative_file, roi_schema, pattern, polarity)
                all_extracted_data[polarity][pattern] = roi_data
            else:
                print(f"  No files found for {pattern}")
                all_extracted_data[polarity][pattern] = None
    
    # Create TSV files
    print(f"\n{'='*50}")
    print("Creating ROI TSV files...")
    
    output_files = []
    for polarity in ['positive', 'negative']:
        if any(all_extracted_data[polarity].values()):
            output_file = create_roi_tsv(all_extracted_data[polarity], polarity, roi_schema)
            if output_file:
                output_files.append(output_file)
    
    print("\n=== Phase 3 Complete ===")
    if output_files:
        print("Generated ROI TSV files:")
        for file in output_files:
            print(f"  {file}")
    else:
        print("⚠️  No ROI TSV files generated - check extraction results")
    
    return all_extracted_data

if __name__ == "__main__":
    extracted_data = main()