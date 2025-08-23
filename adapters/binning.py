"""
Unit-mass binning utilities for ToF-SIMS data.

Provides functions to bin high-resolution mass spectra to integer unit masses
as required by PNNL's SIMS-PCA analysis workflow.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from pathlib import Path


def make_unit_mass_axis(mz_min: float, mz_max: float) -> np.ndarray:
    """
    Create a unit-mass axis from min to max m/z values.
    
    Args:
        mz_min: Minimum m/z value
        mz_max: Maximum m/z value
        
    Returns:
        Array of integer unit masses from floor(mz_min) to ceil(mz_max)
    """
    return np.arange(int(np.floor(mz_min)), int(np.ceil(mz_max)) + 1)


def bin_to_unit_mass(mz: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin high-resolution mass spectrum to unit masses.
    
    Rounds m/z values to nearest integer and sums intensities for each unit mass.
    
    Args:
        mz: Array of m/z values
        intensity: Array of corresponding intensities
        
    Returns:
        Tuple of (unit_masses, binned_intensities)
    """
    # Round m/z to nearest integer
    unit_masses = np.rint(mz).astype(int)
    
    # Create DataFrame for groupby operation
    df = pd.DataFrame({'mass': unit_masses, 'intensity': intensity})
    
    # Sum intensities for each unit mass
    binned = df.groupby('mass')['intensity'].sum().reset_index()
    
    return binned['mass'].values, binned['intensity'].values


def merge_samples_to_tsv(sample_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                        output_path: Path,
                        polarity: str = 'P') -> None:
    """
    Merge multiple samples into a single TSV file compatible with PNNL SIMS-PCA.
    
    The output format follows PNNL's expected schema:
    - First column: "Mass (u)" with integer unit masses
    - Subsequent columns: "{sample_num}-{polarity}" (e.g., "1-P", "2-P")
    
    Based on PNNL's pca_sims.py which uses re.split('-[PN]', label)[0] 
    to extract sample numbers from column headers.
    
    Args:
        sample_data: Dict mapping sample names to (masses, intensities) tuples
        output_path: Path where TSV file will be written
        polarity: Polarity indicator ('P' or 'N')
    """
    # Find global mass range
    all_masses = []
    for masses, _ in sample_data.values():
        all_masses.extend(masses)
    
    if not all_masses:
        raise ValueError("No mass data found in any samples")
    
    global_mass_axis = make_unit_mass_axis(min(all_masses), max(all_masses))
    
    # Create DataFrame with mass axis
    result_df = pd.DataFrame({'Mass (u)': global_mass_axis})
    
    # Add each sample as a column
    for sample_idx, (sample_name, (masses, intensities)) in enumerate(sample_data.items(), 1):
        # Create series with zeros for all masses
        sample_series = pd.Series(0.0, index=global_mass_axis, dtype=float)
        
        # Fill in actual intensities
        for mass, intensity in zip(masses, intensities):
            if mass in sample_series.index:
                sample_series[mass] += intensity
        
        # Column name format expected by PNNL: "{number}-{polarity}"
        column_name = f"{sample_idx}-{polarity}"
        result_df[column_name] = sample_series.values
    
    # Write to TSV
    result_df.to_csv(output_path, sep='\t', index=False, float_format='%.6e')
    print(f"Merged TSV saved to: {output_path}")


def validate_pnnl_tsv(tsv_path: Path) -> bool:
    """
    Validate that a TSV file conforms to PNNL SIMS-PCA format.
    
    Checks:
    - First column is "Mass (u)"
    - Sample columns follow "{number}-{P|N}" pattern
    - Mass values are integers
    
    Args:
        tsv_path: Path to TSV file to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Check first column
        if df.columns[0] != 'Mass (u)':
            print(f"Error: First column should be 'Mass (u)', found '{df.columns[0]}'")
            return False
        
        # Check mass values are integers
        masses = df['Mass (u)']
        if not all(masses == masses.astype(int)):
            print("Error: Mass values should be integers")
            return False
        
        # Check sample column naming pattern
        import re
        sample_pattern = re.compile(r'^\d+-[PN]$')
        
        for col in df.columns[1:]:
            if not sample_pattern.match(col):
                print(f"Error: Column '{col}' doesn't match pattern 'number-P/N'")
                return False
        
        print(f"TSV file {tsv_path} is valid for PNNL SIMS-PCA")
        return True
        
    except Exception as e:
        print(f"Error validating TSV: {e}")
        return False