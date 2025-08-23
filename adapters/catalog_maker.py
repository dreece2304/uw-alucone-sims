"""
Catalog generation for PNNL SIMS-PCA analysis.

Creates catalog.csv files with required metadata for samples, ensuring
Sample # values match the numeric identifiers used in TSV column headers.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import re


def create_catalog_from_names(sample_mapping: Dict[int, str], 
                            operator: str = "Unknown",
                            testing_date: Optional[str] = None,
                            output_path: Union[str, Path] = "catalog.csv") -> None:
    """
    Create catalog.csv from a mapping of sample numbers to names.
    
    Creates the catalog format required by PNNL SIMS-PCA:
    - Sample #: Must match numeric part of TSV column headers (e.g., "1" for "1-P")
    - Sample Short Name: Descriptive name for the sample
    - Testing Date: Date of analysis
    - Operator: Person who performed the analysis
    
    Based on PNNL's pca_sims.py which uses re.split('-[PN]', label)[0] 
    to extract sample numbers from column headers.
    
    Args:
        sample_mapping: Dict mapping sample number -> sample name
        operator: Name of operator performing analysis
        testing_date: Date string (YYYY-MM-DD), defaults to today
        output_path: Path where catalog.csv will be written
    """
    if testing_date is None:
        testing_date = datetime.now().strftime("%Y-%m-%d")
    
    output_path = Path(output_path)
    
    # Required columns for PNNL SIMS-PCA
    fieldnames = ["Sample #", "Sample Short Name", "Testing Date", "Operator"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort by sample number for consistent ordering
        for sample_num in sorted(sample_mapping.keys()):
            sample_name = sample_mapping[sample_num]
            
            writer.writerow({
                "Sample #": sample_num,
                "Sample Short Name": sample_name,
                "Testing Date": testing_date,
                "Operator": operator
            })
    
    print(f"Catalog saved to: {output_path}")
    print(f"Created entries for {len(sample_mapping)} samples")


def create_catalog_from_file_list(file_paths: List[Union[str, Path]], 
                                 operator: str = "Unknown",
                                 testing_date: Optional[str] = None,
                                 sample_names: Optional[List[str]] = None,
                                 output_path: Union[str, Path] = "catalog.csv") -> None:
    """
    Create catalog.csv from a list of data files.
    
    Sample numbers are assigned sequentially (1, 2, 3, ...) in the order
    files are provided, matching the column numbering in the TSV output.
    
    Args:
        file_paths: List of data file paths
        operator: Name of operator performing analysis
        testing_date: Date string (YYYY-MM-DD), defaults to today
        sample_names: Optional custom names (default: use file stems)
        output_path: Path where catalog.csv will be written
    """
    if sample_names and len(sample_names) != len(file_paths):
        raise ValueError("Number of sample names must match number of files")
    
    # Create sample mapping
    sample_mapping = {}
    
    for i, file_path in enumerate(file_paths, 1):  # Start numbering from 1
        file_path = Path(file_path)
        
        if sample_names:
            sample_name = sample_names[i-1]  # sample_names is 0-indexed
        else:
            sample_name = file_path.stem
        
        sample_mapping[i] = sample_name
    
    create_catalog_from_names(sample_mapping, operator, testing_date, output_path)


def parse_sample_mapping(mapping_string: str) -> Dict[int, str]:
    """
    Parse sample mapping from string format.
    
    Accepts formats like:
    - "1:Sample A,2:Sample B,3:Sample C"
    - "1=Sample A;2=Sample B;3=Sample C" 
    
    Args:
        mapping_string: String with sample number to name mappings
        
    Returns:
        Dictionary mapping sample numbers to names
        
    Raises:
        ValueError: If mapping format is invalid
    """
    sample_mapping = {}
    
    # Try different delimiters
    if ',' in mapping_string:
        entries = mapping_string.split(',')
    elif ';' in mapping_string:
        entries = mapping_string.split(';')
    else:
        entries = [mapping_string]
    
    for entry in entries:
        entry = entry.strip()
        
        # Try different separators
        if ':' in entry:
            parts = entry.split(':', 1)
        elif '=' in entry:
            parts = entry.split('=', 1)
        else:
            raise ValueError(f"Invalid mapping format: {entry}. Expected 'number:name' or 'number=name'")
        
        if len(parts) != 2:
            raise ValueError(f"Invalid mapping format: {entry}")
        
        try:
            sample_num = int(parts[0].strip())
            sample_name = parts[1].strip()
            sample_mapping[sample_num] = sample_name
        except ValueError:
            raise ValueError(f"Invalid sample number in: {entry}")
    
    return sample_mapping


def validate_catalog(catalog_path: Union[str, Path], 
                    expected_sample_numbers: Optional[List[int]] = None) -> bool:
    """
    Validate that catalog.csv has required format and sample numbers.
    
    Checks:
    - Required columns present
    - Sample numbers are valid integers
    - If provided, sample numbers match expected list
    
    Args:
        catalog_path: Path to catalog.csv file
        expected_sample_numbers: Optional list of expected sample numbers
        
    Returns:
        True if valid, False otherwise
    """
    try:
        catalog_path = Path(catalog_path)
        
        if not catalog_path.exists():
            print(f"Error: Catalog file not found: {catalog_path}")
            return False
        
        required_columns = {"Sample #", "Sample Short Name", "Testing Date", "Operator"}
        
        with open(catalog_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check required columns
            if not required_columns.issubset(set(reader.fieldnames)):
                missing = required_columns - set(reader.fieldnames)
                print(f"Error: Missing required columns: {missing}")
                return False
            
            # Check sample numbers
            sample_numbers = []
            
            for row in reader:
                try:
                    sample_num = int(row["Sample #"])
                    sample_numbers.append(sample_num)
                except ValueError:
                    print(f"Error: Invalid sample number: {row['Sample #']}")
                    return False
                
                # Check for empty required fields
                for col in required_columns:
                    if not row[col].strip():
                        print(f"Error: Empty value in required column '{col}' for sample {row['Sample #']}")
                        return False
            
            # Check expected sample numbers if provided
            if expected_sample_numbers:
                if set(sample_numbers) != set(expected_sample_numbers):
                    print(f"Error: Sample numbers mismatch")
                    print(f"  Found: {sorted(sample_numbers)}")
                    print(f"  Expected: {sorted(expected_sample_numbers)}")
                    return False
        
        print(f"Catalog {catalog_path} is valid")
        return True
        
    except Exception as e:
        print(f"Error validating catalog: {e}")
        return False


def extract_sample_numbers_from_tsv(tsv_path: Union[str, Path]) -> List[int]:
    """
    Extract sample numbers from PNNL TSV column headers.
    
    Uses the same logic as PNNL's pca_sims.py: re.split('-[PN]', label)[0]
    
    Args:
        tsv_path: Path to TSV file
        
    Returns:
        List of sample numbers found in column headers
    """
    tsv_path = Path(tsv_path)
    
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    sample_numbers = []
    
    with open(tsv_path, 'r', encoding='utf-8') as tsvfile:
        # Read first line to get headers
        header_line = tsvfile.readline().strip()
        columns = header_line.split('\t')
        
        # Skip first column (Mass (u))
        for col in columns[1:]:
            # Use same logic as PNNL's pca_sims.py
            match = re.split('-[PN]', col)
            if match and match[0].isdigit():
                sample_numbers.append(int(match[0]))
    
    return sorted(list(set(sample_numbers)))  # Remove duplicates and sort