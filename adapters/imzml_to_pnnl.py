"""
imzML data format converter to PNNL SIMS-PCA format.

Handles imzML imaging mass spectrometry files using pyimzML and converts them 
to the tab-delimited format expected by PNNL's ATOFSIMSCLASS.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

# pyimzML is imported lazily in functions to avoid import errors when module is not available

from .binning import bin_to_unit_mass, merge_samples_to_tsv


def extract_sum_spectrum_from_imzml(imzml_path: Union[str, Path], 
                                   subset_pixels: Optional[int] = None,
                                   random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract summed spectrum from imzML file by accumulating all pixels.
    
    For imaging datasets, this creates a global sum spectrum representing
    the entire sample by summing intensities across all pixels/spectra.
    
    Args:
        imzml_path: Path to imzML file
        subset_pixels: Optional number of pixels to randomly sample for speed
        random_seed: Random seed for reproducible pixel sampling
        
    Returns:
        Tuple of (masses, intensities) for the summed spectrum
        
    Raises:
        ImportError: If pyimzML is not available
    """
    try:
        from pyimzml.ImzMLParser import ImzMLParser
    except ImportError:
        raise ImportError("pyimzML is required for imzML processing. Install with: pip install pyimzML")
    
    imzml_path = Path(imzml_path)
    
    if not imzml_path.exists():
        raise FileNotFoundError(f"imzML file not found: {imzml_path}")
    
    print(f"Reading imzML file: {imzml_path}")
    
    try:
        parser = ImzMLParser(str(imzml_path))
    except Exception as e:
        raise RuntimeError(f"Failed to parse imzML file: {e}")
    
    # Get total number of spectra
    total_pixels = len(parser.coordinates)
    
    print(f"Total pixels in dataset: {total_pixels}")
    
    # Determine which pixel indices to process
    if subset_pixels and subset_pixels < total_pixels:
        if random_seed is not None:
            random.seed(random_seed)
        
        # Randomly sample subset of pixel indices
        indices_to_process = random.sample(range(total_pixels), subset_pixels)
        print(f"Randomly sampling {len(indices_to_process)} pixels")
    else:
        indices_to_process = list(range(total_pixels))
        print(f"Processing all {len(indices_to_process)} pixels")
    
    # Accumulate spectra across pixels using index-based access
    mass_intensity_dict = {}
    processed_count = 0
    
    for pixel_index in indices_to_process:
        try:
            # Get spectrum for this pixel using index-based access
            masses, intensities = parser.getspectrum(pixel_index)
            
            # Accumulate intensities for each mass
            for mass, intensity in zip(masses, intensities):
                if mass in mass_intensity_dict:
                    mass_intensity_dict[mass] += intensity
                else:
                    mass_intensity_dict[mass] = intensity
            
            processed_count += 1
            
            # Progress update for large datasets
            if processed_count % 1000 == 0:
                print(f"  Processed {processed_count}/{len(indices_to_process)} pixels")
                
        except Exception as e:
            print(f"Warning: Failed to process pixel index {pixel_index}: {e}")
            continue
    
    if not mass_intensity_dict:
        raise RuntimeError("No spectra could be extracted from imzML file")
    
    # Convert to arrays
    masses = np.array(list(mass_intensity_dict.keys()))
    intensities = np.array(list(mass_intensity_dict.values()))
    
    # Sort by mass
    sort_indices = np.argsort(masses)
    masses = masses[sort_indices]
    intensities = intensities[sort_indices]
    
    print(f"Extracted {len(masses)} unique masses from {processed_count} pixels")
    
    return masses, intensities


def process_imzml_files(file_paths: List[Union[str, Path]], 
                       polarity: str = 'P',
                       sample_names: Optional[List[str]] = None,
                       subset_pixels: Optional[int] = None) -> dict:
    """
    Process multiple imzML files and bin to unit masses.
    
    Each imzML file is treated as one sample, with all pixels summed
    to create a representative spectrum for that sample.
    
    Args:
        file_paths: List of paths to imzML files
        polarity: Polarity indicator ('P' or 'N')
        sample_names: Optional list of custom sample names (default: use filenames)
        subset_pixels: Optional number of pixels to randomly sample per file
        
    Returns:
        Dictionary mapping sample names to (masses, intensities) tuples
    """
    if sample_names and len(sample_names) != len(file_paths):
        raise ValueError("Number of sample names must match number of files")
    
    sample_data = {}
    
    for i, file_path in enumerate(file_paths):
        file_path = Path(file_path)
        
        # Determine sample name
        if sample_names:
            sample_name = sample_names[i]
        else:
            sample_name = file_path.stem
        
        try:
            print(f"Processing {file_path}...")
            
            # Extract summed spectrum from imzML
            masses, intensities = extract_sum_spectrum_from_imzml(
                file_path, subset_pixels=subset_pixels
            )
            
            # Bin to unit masses
            unit_masses, binned_intensities = bin_to_unit_mass(masses, intensities)
            
            sample_data[sample_name] = (unit_masses, binned_intensities)
            
            print(f"  Binned {len(masses)} masses -> {len(unit_masses)} unit masses")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Continue with other files
            continue
    
    return sample_data


def imzml_to_pnnl_tsv(file_paths: List[Union[str, Path]],
                     output_tsv: Union[str, Path],
                     polarity: str = 'P',
                     sample_names: Optional[List[str]] = None,
                     subset_pixels: Optional[int] = None) -> None:
    """
    Convert imzML files to PNNL SIMS-PCA TSV format.
    
    Creates a tab-delimited file with:
    - First column: "Mass (u)" (integer unit masses)
    - Sample columns: "1-P", "2-P", etc. (numbered in order received)
    
    Each imzML file represents one sample, with all imaging pixels summed
    to create a representative spectrum.
    
    Args:
        file_paths: List of imzML file paths
        output_tsv: Output TSV file path
        polarity: Polarity indicator ('P' for positive, 'N' for negative)
        sample_names: Optional custom sample names (default: use file order)
        subset_pixels: Optional number of pixels to randomly sample per file for speed
    """
    if polarity not in ['P', 'N']:
        raise ValueError("Polarity must be 'P' (positive) or 'N' (negative)")
    
    print(f"Converting {len(file_paths)} imzML files to PNNL format...")
    if subset_pixels:
        print(f"Using random subset of {subset_pixels} pixels per file")
    
    # Process all files
    sample_data = process_imzml_files(file_paths, polarity, sample_names, subset_pixels)
    
    if not sample_data:
        raise RuntimeError("No files were successfully processed")
    
    # Merge into TSV format
    merge_samples_to_tsv(sample_data, Path(output_tsv), polarity)
    
    print(f"Successfully converted {len(sample_data)} samples")
    print(f"Output saved to: {output_tsv}")


def validate_imzml_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file can be opened as imzML data.
    
    Args:
        file_path: Path to file to validate
        
    Returns:
        True if file can be opened, False otherwise
    """
    try:
        from pyimzml.ImzMLParser import ImzMLParser
    except ImportError:
        return False
    
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return False
        
        parser = ImzMLParser(str(file_path))
        total_pixels = len(parser.coordinates)
        
        # Try to get first spectrum using index-based access
        if total_pixels > 0:
            masses, intensities = parser.getspectrum(0)
            return len(masses) > 0 and len(intensities) > 0
        
        return False
    except Exception:
        return False


def get_imzml_info(file_path: Union[str, Path]) -> dict:
    """
    Get basic information about an imzML file.
    
    Args:
        file_path: Path to imzML file
        
    Returns:
        Dictionary with file information
    """
    try:
        from pyimzml.ImzMLParser import ImzMLParser
    except ImportError:
        return {'file_path': str(file_path), 'error': 'pyimzML not available'}
    
    try:
        parser = ImzMLParser(str(file_path))
        total_pixels = len(parser.coordinates)
        
        # Get first spectrum to determine mass range using index-based access
        if total_pixels > 0:
            masses, _ = parser.getspectrum(0)
            mass_range = (float(np.min(masses)), float(np.max(masses)))
        else:
            mass_range = (0, 0)
        
        info = {
            'file_path': str(file_path),
            'total_pixels': total_pixels,
            'mass_range': mass_range,
            'polarity': getattr(parser, 'polarity', 'unknown'),
        }
        
        return info
    except Exception as e:
        return {'file_path': str(file_path), 'error': str(e)}