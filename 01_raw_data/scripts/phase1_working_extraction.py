#!/usr/bin/env python3
"""
Phase 1: Working Data Extraction from ITAX Files
===============================================

Extract data from ITAX files which are working correctly and organize into
standardized formats for downstream analysis.

ITAX files contain processed, calibrated ToF-SIMS spectra - ideal for MVA.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, 'pySPM_source')
sys.path.insert(0, '.')

import pySPM
from adapters.pyspm_fixes import open_iontof_with_fixes, validate_extracted_spectrum
from iontof_patcher import apply_pySPM_patches

class WorkingDataExtractor:
    """Extract data from working ITAX files and organize for analysis."""
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        
        # Data directories
        self.pos_dir = self.base_dir / '01_raw_data' / 'positive'
        self.neg_dir = self.base_dir / '01_raw_data' / 'negative'
        
        # Output directory
        self.output_dir = self.base_dir / '01_raw_data' / 'working_extraction'
        self.output_dir.mkdir(exist_ok=True)
        
        # Dose mapping
        self.dose_mapping = {
            'SQ1r': 500, 'SQ1': 2000, 'SQ2': 5000, 
            'SQ3': 10000, 'SQ4': 15000
        }
        
        # Apply patches
        apply_pySPM_patches()
        print("✅ pySPM patches applied")
    
    def extract_all_data(self):
        """Extract data from all ITAX files."""
        print("="*80)
        print("ToF-SIMS WORKING DATA EXTRACTION")
        print("="*80)
        print("Extracting from ITAX files (verified working)")
        print("Output: TSV matrices ready for multivariate analysis")
        print("="*80)
        
        # Process both polarities
        for polarity in ['positive', 'negative']:
            print(f"\n{'='*60}")
            print(f"PROCESSING {polarity.upper()} POLARITY")
            print(f"{'='*60}")
            
            data_dir = self.pos_dir if polarity == 'positive' else self.neg_dir
            self.process_polarity(polarity, data_dir)
        
        print(f"\n{'='*80}")
        print("✅ WORKING DATA EXTRACTION COMPLETE")
        print(f"✅ Data extracted to: {self.output_dir}")
        print("✅ Ready for multivariate analysis")
        print(f"{'='*80}")
    
    def process_polarity(self, polarity, data_dir):
        """Process all ITAX files for one polarity."""
        
        # Find ITAX files
        itax_files = list(data_dir.glob("*.itax"))
        print(f"Found {len(itax_files)} ITAX files")
        
        if len(itax_files) == 0:
            print(f"❌ No ITAX files found in {data_dir}")
            return
        
        # Extract data from each file
        all_samples = []
        all_spectra = []
        common_masses = None
        
        for file_path in tqdm(itax_files, desc=f"Extracting {polarity} ITAX"):
            sample_data = self.extract_sample_data(file_path, polarity)
            
            if sample_data and 'masses' in sample_data and 'intensities' in sample_data:
                all_samples.append(sample_data['metadata'])
                all_spectra.append(sample_data['intensities'])
                
                # Use first file's masses as reference
                if common_masses is None:
                    common_masses = sample_data['masses']
                    print(f"  Using {len(common_masses)} mass points from {file_path.name}")
        
        if len(all_samples) == 0:
            print(f"❌ No valid spectra extracted for {polarity}")
            return
        
        print(f"✅ Extracted {len(all_samples)} valid spectra")
        
        # Create data matrix
        self.save_polarity_data(polarity, all_samples, all_spectra, common_masses)
    
    def extract_sample_data(self, file_path, polarity):
        """Extract data from one ITAX file."""
        try:
            # Parse sample info from filename
            filename = file_path.stem  # P1_SQ1_01 or P1_SQ1r_01
            parts = filename.split('_')
            
            if len(parts) < 3:
                print(f"⚠ Cannot parse filename: {filename}")
                return None
            
            pattern = parts[0]  # P1
            square = parts[1]   # SQ1, SQ1r, etc.
            number = parts[2]   # 01, 06
            
            dose = self.dose_mapping.get(square, 0)
            
            # Extract spectrum using our working method
            masses, intensities = open_iontof_with_fixes(file_path)
            
            if not validate_extracted_spectrum(masses, intensities, file_path):
                return None
            
            # Create sample metadata
            metadata = {
                'sample_id': filename,
                'file_name': file_path.name,
                'pattern': pattern,
                'square': square,
                'number': number,
                'dose_uC_cm2': dose,
                'polarity': polarity,
                'n_points': len(masses),
                'mass_range_min': float(np.min(masses)),
                'mass_range_max': float(np.max(masses)),
                'intensity_sum': float(np.sum(intensities)),
                'intensity_max': float(np.max(intensities)),
            }
            
            return {
                'metadata': metadata,
                'masses': masses.astype(np.float32),
                'intensities': intensities.astype(np.float32)
            }
            
        except Exception as e:
            print(f"❌ Failed to extract {file_path.name}: {e}")
            return None
    
    def save_polarity_data(self, polarity, samples, spectra, masses):
        """Save extracted data in analysis-ready format."""
        print(f"\n--- Saving {polarity} data ---")
        
        # Create sample information DataFrame
        df_samples = pd.DataFrame(samples)
        df_samples = df_samples.sort_values(['pattern', 'square', 'number']).reset_index(drop=True)
        
        # Create intensity matrix DataFrame
        intensity_matrix = np.array(spectra)
        
        # Use sample_id as index and mass values as columns
        mass_columns = [f"m{mass:.4f}" for mass in masses]
        df_spectra = pd.DataFrame(
            intensity_matrix,
            index=df_samples['sample_id'],
            columns=mass_columns
        )
        
        # Add metadata columns to spectra DataFrame
        key_cols = ['pattern', 'square', 'dose_uC_cm2', 'polarity']
        for col in key_cols:
            df_spectra[col] = df_samples.set_index('sample_id')[col]
        
        # Save files
        samples_file = self.output_dir / f'{polarity}_samples.tsv'
        spectra_file = self.output_dir / f'{polarity}_spectra.tsv'  
        masses_file = self.output_dir / f'{polarity}_masses.tsv'
        
        df_samples.to_csv(samples_file, sep='\t', index=False)
        df_spectra.to_csv(spectra_file, sep='\t')
        
        # Save mass axis separately
        df_masses = pd.DataFrame({'mass_u': masses})
        df_masses.to_csv(masses_file, sep='\t', index=False)
        
        print(f"  ✅ Samples info: {samples_file}")
        print(f"  ✅ Spectra matrix: {spectra_file}")
        print(f"  ✅ Mass axis: {masses_file}")
        print(f"  📊 Shape: {len(df_samples)} samples × {len(masses)} masses")
        
        # Create summary
        summary = {
            'polarity': polarity,
            'n_samples': len(df_samples),
            'n_mass_points': len(masses),
            'mass_range': [float(np.min(masses)), float(np.max(masses))],
            'dose_levels': sorted(df_samples['dose_uC_cm2'].unique().tolist()),
            'patterns': sorted(df_samples['pattern'].unique().tolist()),
            'squares': sorted(df_samples['square'].unique().tolist()),
            'files_processed': df_samples['file_name'].tolist()
        }
        
        summary_file = self.output_dir / f'{polarity}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✅ Summary: {summary_file}")
        print(f"  📈 Dose levels: {summary['dose_levels']}")
        print(f"  🧪 Patterns: {summary['patterns']}")

if __name__ == "__main__":
    extractor = WorkingDataExtractor("/home/dreece23/sims-pca-ws")
    extractor.extract_all_data()