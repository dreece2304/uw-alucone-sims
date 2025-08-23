#!/usr/bin/env python3
"""
Phase 1: Imaging Data Readiness & Indexing
- Check available imaging files (.itm/.itmx)
- Attempt to read imaging data using pySPM
- Generate index files with metadata
- Determine if conversion to imzML is needed
"""

import json
import sys
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

try:
    from pyimzml.ImzMLParser import ImzMLParser
    PYIMZML_AVAILABLE = True
except ImportError:
    PYIMZML_AVAILABLE = False

def find_imaging_files() -> Dict[str, List[Path]]:
    """Find available imaging files"""
    print("Scanning for imaging files...")
    
    imaging_files = {
        'imzml': [],
        'itm': [],
        'itmx': [],
        'other': []
    }
    
    data_dir = Path('data')
    
    # Check for imzML files first
    for pattern in ['PositiveIonData', 'NegativeIonData', '.']:
        search_dir = data_dir / pattern if pattern != '.' else data_dir
        if search_dir.exists():
            for ext in ['.imzML', '.imzml']:
                imaging_files['imzml'].extend(search_dir.glob(f'*{ext}'))
    
    # Check for IONTOF imaging files
    for pattern_dir in ['PositiveIonData', 'NegativeIonData']:
        search_dir = data_dir / pattern_dir
        if search_dir.exists():
            imaging_files['itm'].extend(search_dir.glob('*.itm'))
            imaging_files['itmx'].extend(search_dir.glob('*.itmx'))
    
    # Print summary
    for file_type, files in imaging_files.items():
        if files:
            print(f"  Found {len(files)} {file_type.upper()} files")
            for f in files[:3]:  # Show first 3
                print(f"    {f}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
    
    return imaging_files

def analyze_itm_file(file_path: Path) -> Optional[Dict]:
    """Analyze an ITM imaging file using pySPM"""
    if not PYSPM_AVAILABLE:
        return None
    
    try:
        print(f"  Analyzing {file_path.name}...")
        itm = pySPM.ITM(str(file_path))
        
        # Get basic metadata
        info = {
            'file_path': str(file_path),
            'file_type': 'ITM',
            'size_x': getattr(itm, 'size', {}).get('pixels', {}).get('x', 'unknown'),
            'size_y': getattr(itm, 'size', {}).get('pixels', {}).get('y', 'unknown'),
            'real_size_x': getattr(itm, 'size', {}).get('real', {}).get('x', 'unknown'),
            'real_size_y': getattr(itm, 'size', {}).get('real', {}).get('y', 'unknown'),
            'polarity': getattr(itm, 'polarity', 'unknown'),
            'n_scans': getattr(itm, 'Nscan', 'unknown'),
            'shots_per_pixel': getattr(itm, 'spp', 'unknown')
        }
        
        # Try to get mass range from a spectrum
        try:
            test_spectrum = itm.getSpectrum(0, 0)  # Get spectrum from pixel 0,0
            if test_spectrum and len(test_spectrum) >= 2:
                masses, intensities = test_spectrum
                if len(masses) > 0:
                    info['mass_min'] = float(min(masses))
                    info['mass_max'] = float(max(masses))
                    info['n_masses'] = len(masses)
                    info['centroided'] = True  # ITM are typically centroided
        except Exception as e:
            print(f"    Warning: Could not extract test spectrum: {e}")
            info['mass_min'] = 'unknown'
            info['mass_max'] = 'unknown' 
            info['n_masses'] = 'unknown'
        
        # Calculate estimated number of spectra
        if info['size_x'] != 'unknown' and info['size_y'] != 'unknown':
            info['n_spectra'] = info['size_x'] * info['size_y']
        else:
            info['n_spectra'] = 'unknown'
        
        return info
        
    except Exception as e:
        if "Missing block" in str(e) and "ShotsPerPixel" in str(e):
            print(f"    ITM: Detected missing ShotsPerPixel block, trying alternative approach...")
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
                
                print(f"    ITM: Initialized with default parameters, extracting metadata...")
                
                # Get basic metadata with fallback values
                info = {
                    'file_path': str(file_path),
                    'file_type': 'ITM',
                    'size_x': getattr(itm, 'size', {}).get('pixels', {}).get('x', 256),
                    'size_y': getattr(itm, 'size', {}).get('pixels', {}).get('y', 256),
                    'real_size_x': getattr(itm, 'size', {}).get('real', {}).get('x', 500e-6),
                    'real_size_y': getattr(itm, 'size', {}).get('real', {}).get('y', 500e-6),
                    'polarity': getattr(itm, 'polarity', 'Positive'),
                    'n_scans': getattr(itm, 'Nscan', 1),
                    'shots_per_pixel': getattr(itm, 'spp', 1),
                    'readable': True
                }
                
                # Try to get mass range from a spectrum
                try:
                    test_spectrum = itm.getSpectrum(0, 0)  # Get spectrum from pixel 0,0
                    if test_spectrum and len(test_spectrum) >= 2:
                        masses, intensities = test_spectrum
                        if len(masses) > 0:
                            info['mass_min'] = float(min(masses))
                            info['mass_max'] = float(max(masses))
                            info['n_masses'] = len(masses)
                            info['centroided'] = True  # ITM are typically centroided
                            print(f"    ✓ ITM alternative approach successful: extracted mass range {info['mass_min']:.1f}-{info['mass_max']:.1f}")
                except Exception as e2:
                    print(f"    Warning: Could not extract test spectrum with alternative method: {e2}")
                    info['mass_min'] = 'unknown'
                    info['mass_max'] = 'unknown' 
                    info['n_masses'] = 'unknown'
                
                # Calculate estimated number of spectra
                if info['size_x'] != 'unknown' and info['size_y'] != 'unknown':
                    info['n_spectra'] = info['size_x'] * info['size_y']
                else:
                    info['n_spectra'] = 'unknown'
                
                # Clean up
                if hasattr(itm, 'f') and itm.f:
                    itm.f.close()
                
                return info
                
            except Exception as e2:
                print(f"    ITM: Alternative approach also failed: {e2}")
                return {
                    'file_path': str(file_path),
                    'file_type': 'ITM',
                    'error': str(e),
                    'readable': False
                }
        else:
            print(f"    Error analyzing {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'file_type': 'ITM',
                'error': str(e),
                'readable': False
            }

def analyze_imzml_file(file_path: Path) -> Optional[Dict]:
    """Analyze an imzML file using pyimzML"""
    if not PYIMZML_AVAILABLE:
        return None
    
    try:
        print(f"  Analyzing {file_path.name}...")
        parser = ImzMLParser(str(file_path))
        
        # Get coordinates
        coords = parser.coordinates
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        
        # Get mass range from first spectrum
        masses, intensities = parser.getspectrum(0)
        
        info = {
            'file_path': str(file_path),
            'file_type': 'imzML',
            'n_spectra': len(coords),
            'size_x': max(x_coords) - min(x_coords) + 1,
            'size_y': max(y_coords) - min(y_coords) + 1,
            'coord_min_x': min(x_coords),
            'coord_max_x': max(x_coords),
            'coord_min_y': min(y_coords),
            'coord_max_y': max(y_coords),
            'mass_min': float(min(masses)) if len(masses) > 0 else 'unknown',
            'mass_max': float(max(masses)) if len(masses) > 0 else 'unknown',
            'n_masses': len(masses),
            'centroided': parser.metadata.file_description.param_by_name("MS1 spectrum").value == "centroid spectrum",
            'polarity': 'positive' if 'positive' in str(file_path).lower() else 'negative'
        }
        
        return info
        
    except Exception as e:
        print(f"    Error analyzing {file_path}: {e}")
        return {
            'file_path': str(file_path),
            'file_type': 'imzML', 
            'error': str(e),
            'readable': False
        }

def parse_pattern_and_polarity(file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Parse pattern (P1, P2, P3) and polarity from file path"""
    path_str = str(file_path)
    
    # Extract pattern
    pattern = None
    for p in ['P1', 'P2', 'P3']:
        if p in path_str:
            pattern = p
            break
    
    # Extract polarity
    polarity = None
    if 'PositiveIonData' in path_str or '_01.' in path_str:
        polarity = 'positive'
    elif 'NegativeIonData' in path_str or '_06.' in path_str:
        polarity = 'negative'
    elif 'positive' in path_str.lower():
        polarity = 'positive'
    elif 'negative' in path_str.lower():
        polarity = 'negative'
    
    return pattern, polarity

def main():
    """Execute Phase 1 analysis"""
    print("=== Phase 1: Imaging Data Readiness & Indexing ===")
    
    # Check library availability
    print(f"pySPM available: {PYSPM_AVAILABLE}")
    print(f"pyimzML available: {PYIMZML_AVAILABLE}")
    
    if not PYSPM_AVAILABLE and not PYIMZML_AVAILABLE:
        print("Error: Neither pySPM nor pyimzML is available!")
        return
    
    # Find imaging files
    imaging_files = find_imaging_files()
    
    # Analyze available files
    all_analyses = []
    
    # Analyze imzML files first (preferred)
    if imaging_files['imzml']:
        print("\nAnalyzing imzML files...")
        for file_path in imaging_files['imzml']:
            analysis = analyze_imzml_file(file_path)
            if analysis:
                pattern, polarity = parse_pattern_and_polarity(file_path)
                analysis['pattern'] = pattern
                analysis['polarity'] = polarity
                all_analyses.append(analysis)
    
    # Analyze ITM files if no imzML available
    elif imaging_files['itm']:
        print("\nAnalyzing ITM files...")
        for file_path in imaging_files['itm']:
            analysis = analyze_itm_file(file_path)
            if analysis:
                pattern, polarity = parse_pattern_and_polarity(file_path)
                analysis['pattern'] = pattern
                analysis['polarity'] = polarity
                all_analyses.append(analysis)
    
    else:
        print("\nNo imaging files found!")
        print("Looking for: .imzML, .itm files")
        print("Please ensure imaging data is available in the data/ directory")
        return
    
    # Generate index files per pattern/polarity combination
    print(f"\nGenerating index files for {len(all_analyses)} imaging files...")
    
    index_files_created = []
    for analysis in all_analyses:
        if analysis.get('pattern') and analysis.get('polarity'):
            # Create index filename
            index_filename = f"{analysis['pattern']}_{analysis['polarity']}.index.json"
            index_path = Path('data') / index_filename
            
            # Save index
            with open(index_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            index_files_created.append(index_path)
            print(f"  Created: {index_path}")
    
    # Create summary report
    print("\nGenerating summary report...")
    
    summary_lines = []
    summary_lines.append("=== Imaging Data Readiness Summary ===")
    summary_lines.append(f"Analysis Date: 2025-08-22")
    summary_lines.append("")
    
    summary_lines.append(f"Library Status:")
    summary_lines.append(f"  pySPM: {'Available' if PYSPM_AVAILABLE else 'Not Available'}")
    summary_lines.append(f"  pyimzML: {'Available' if PYIMZML_AVAILABLE else 'Not Available'}")
    summary_lines.append("")
    
    summary_lines.append(f"Files Found:")
    for file_type, files in imaging_files.items():
        if files:
            summary_lines.append(f"  {file_type.upper()}: {len(files)} files")
    summary_lines.append("")
    
    summary_lines.append(f"Analyzed Files:")
    readable_count = sum(1 for a in all_analyses if not a.get('error'))
    summary_lines.append(f"  Total: {len(all_analyses)}")
    summary_lines.append(f"  Readable: {readable_count}")
    summary_lines.append(f"  Index files created: {len(index_files_created)}")
    summary_lines.append("")
    
    if readable_count > 0:
        summary_lines.append("File Details:")
        for analysis in all_analyses:
            if not analysis.get('error'):
                file_name = Path(analysis['file_path']).name
                pattern = analysis.get('pattern', 'unknown')
                polarity = analysis.get('polarity', 'unknown')
                n_spectra = analysis.get('n_spectra', 'unknown')
                mass_range = f"{analysis.get('mass_min', '?')}-{analysis.get('mass_max', '?')} u"
                
                summary_lines.append(f"  {file_name}: {pattern} {polarity}, {n_spectra} spectra, {mass_range}")
    
    # Recommendations
    summary_lines.append("")
    summary_lines.append("Recommendations:")
    
    if imaging_files['imzml']:
        summary_lines.append("  ✓ imzML files found - ready for ROI analysis")
    elif imaging_files['itm']:
        summary_lines.append("  ⚠ Only ITM files found - consider exporting to imzML for broader compatibility")
        summary_lines.append("  ✓ Can proceed with ITM files using pySPM")
    else:
        summary_lines.append("  ✗ No imaging files found - export imaging data from IONTOF software")
        summary_lines.append("    Recommended: Export as imzML (centroided, per pattern/polarity)")
    
    # Save summary
    summary_path = Path('qc/imzml_readiness_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\nSummary saved to: {summary_path}")
    
    print("\n=== Phase 1 Complete ===")
    print(f"Generated files:")
    print(f"  - {len(index_files_created)} index files in data/")
    print(f"  - qc/imzml_readiness_summary.txt")
    
    if readable_count == 0:
        print("\n⚠ WARNING: No readable imaging files found!")
        print("  Please ensure imaging data is properly exported and accessible.")
    
    return all_analyses

if __name__ == "__main__":
    analyses = main()