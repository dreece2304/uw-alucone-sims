#!/usr/bin/env python3
"""
Rename column headers from SQ codes to ebeam doses (microC/cm2).

Mapping:
- SQ1r → 500 μC/cm²
- SQ1  → 2000 μC/cm² 
- SQ2  → 5000 μC/cm²
- SQ3  → 10000 μC/cm²
- SQ4  → 15000 μC/cm²
"""

import pandas as pd
import re
from pathlib import Path

def rename_sq_to_ebeam_dose(input_file: str, output_file: str):
    """Rename column headers from SQ codes to ebeam doses"""
    
    # Mapping from SQ codes to ebeam doses
    sq_mapping = {
        'SQ1r': '500',
        'SQ1': '2000', 
        'SQ2': '5000',
        'SQ3': '10000',
        'SQ4': '15000'
    }
    
    # Read the TSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file, sep='\t')
    
    # Get current column names
    original_columns = df.columns.tolist()
    print(f"Original columns: {original_columns}")
    
    # Create new column names
    new_columns = []
    for col in original_columns:
        if col == 'Mass (u)':
            new_columns.append(col)  # Keep mass column unchanged
        else:
            # Extract pattern like "1-P", "2-P", etc and the SQ code
            # Column format should be like "1-P" where 1 corresponds to P1_SQ1r_01, etc
            match = re.match(r'(\d+)-([PN])', col)
            if match:
                sample_num = int(match.group(1))
                polarity = match.group(2)
                
                # Map sample numbers to pattern/SQ combinations
                # Based on the metadata file structure:
                # 1-P: P1_SQ1_01 → P1_2000
                # 2-P: P1_SQ1r_01 → P1_500  
                # 3-P: P1_SQ2_01 → P1_5000
                # etc.
                
                sample_mapping = {
                    1: ('P1', 'SQ1'),   # P1_SQ1_01
                    2: ('P1', 'SQ1r'),  # P1_SQ1r_01
                    3: ('P1', 'SQ2'),   # P1_SQ2_01
                    4: ('P1', 'SQ3'),   # P1_SQ3_01
                    5: ('P1', 'SQ4'),   # P1_SQ4_01
                    6: ('P2', 'SQ1'),   # P2_SQ1_01
                    7: ('P2', 'SQ1r'),  # P2_SQ1r_01
                    8: ('P2', 'SQ2'),   # P2_SQ2_01
                    9: ('P2', 'SQ3'),   # P2_SQ3_01
                    10: ('P2', 'SQ4'),  # P2_SQ4_01
                    11: ('P3', 'SQ1'),  # P3_SQ1_01
                    12: ('P3', 'SQ1r'), # P3_SQ1r_01
                    13: ('P3', 'SQ2'),  # P3_SQ2_01
                    14: ('P3', 'SQ3'),  # P3_SQ3_01
                    15: ('P3', 'SQ4'),  # P3_SQ4_01
                }
                
                if sample_num in sample_mapping:
                    pattern, sq_code = sample_mapping[sample_num]
                    dose = sq_mapping[sq_code]
                    new_col = f"{pattern}_{dose}μC-{polarity}"
                    new_columns.append(new_col)
                    print(f"  {col} → {new_col}")
                else:
                    new_columns.append(col)  # Keep unchanged if not found
            else:
                new_columns.append(col)  # Keep unchanged if pattern doesn't match
    
    # Rename columns
    df.columns = new_columns
    
    # Save renamed file
    print(f"Writing renamed file to {output_file}...")
    df.to_csv(output_file, sep='\t', index=False)
    print("Done!")

if __name__ == "__main__":
    # Rename positive ion data
    rename_sq_to_ebeam_dose("out/all_positive_data.tsv", "out/all_positive_data_renamed.tsv")
    
    # Rename negative ion data
    rename_sq_to_ebeam_dose("out/all_negative_data.tsv", "out/all_negative_data_renamed.tsv")