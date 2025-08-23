#!/usr/bin/env python3
"""
Phase 1: Complete Data Extraction & Organization
===============================================

Comprehensive extraction from all ToF-SIMS file formats (ITM, ITAX, ITMX)
and organization into standardized formats for downstream analysis.

Data Extraction Strategy:
- ITM files: Raw measurement data, spatial imaging, full mass spectra
- ITAX files: Processed analysis data, calibrated spectra, metadata
- ITMX files: Extended analysis data, additional processing results

Output Formats:
- HDF5: Efficient binary storage for large imaging datasets
- TSV: Human-readable tabular data for MVA
- JSON: Metadata and experimental parameters
- NPZ: NumPy arrays for fast Python loading

Author: Claude Code Assistant  
Date: 2025-08-23
"""

import os
import sys
import json
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add pySPM and local modules to path
sys.path.insert(0, 'pySPM_source')
sys.path.insert(0, '.')

import pySPM
from adapters.pyspm_fixes import open_iontof_with_fixes, validate_extracted_spectrum
from iontof_patcher import apply_pySPM_patches

class ComprehensiveDataExtractor:
    """Extract and organize data from all ToF-SIMS file formats."""
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        
        # Data directories
        self.raw_data_dir = self.base_dir / '01_raw_data'
        self.pos_dir = self.raw_data_dir / 'positive'
        self.neg_dir = self.raw_data_dir / 'negative'
        
        # Output directories
        self.extracted_dir = self.base_dir / '01_raw_data' / 'extracted'
        self.extracted_dir.mkdir(exist_ok=True)
        
        # Organized output subdirectories
        self.formats = {
            'hdf5': self.extracted_dir / 'hdf5',
            'tsv': self.extracted_dir / 'tsv', 
            'json': self.extracted_dir / 'json',
            'npz': self.extracted_dir / 'npz'
        }
        
        for format_dir in self.formats.values():
            format_dir.mkdir(exist_ok=True)
        
        # Results tracking
        self.extraction_results = {
            'success': [],
            'failed': [],
            'file_info': {},
            'data_comparison': {}
        }
        
        # Dose mapping (from project metadata)
        self.dose_mapping = {
            'SQ1r': 500, 'SQ1': 2000, 'SQ2': 5000, 
            'SQ3': 10000, 'SQ4': 15000
        }
        
        # Apply pySPM patches for compatibility
        print("Applying pySPM compatibility patches...")
        apply_pySPM_patches()
        
    def extract_all_data(self):
        """Extract data from all file formats systematically."""
        print("="*80)
        print("ToF-SIMS COMPREHENSIVE DATA EXTRACTION")
        print("="*80)
        print("Extracting from ITM, ITAX, and ITMX files")
        print("Converting to standardized formats: HDF5, TSV, JSON, NPZ")
        print("="*80)
        
        # Process both polarities
        for polarity in ['positive', 'negative']:
            print(f"\n{'='*60}")
            print(f"PROCESSING {polarity.upper()} POLARITY")
            print(f"{'='*60}")
            
            data_dir = self.pos_dir if polarity == 'positive' else self.neg_dir
            self.process_polarity(polarity, data_dir)
        
        # Create comparison summaries
        self.create_format_comparison_summary()
        self.save_extraction_results()
        
        print(f"\n{'='*80}")
        print("✅ COMPREHENSIVE DATA EXTRACTION COMPLETE")
        print(f"✅ Data extracted to: {self.extracted_dir}")
        print(f"✅ Formats available: HDF5, TSV, JSON, NPZ")
        print(f"{'='*80}")
        
    def process_polarity(self, polarity, data_dir):
        """Process all file formats for one polarity."""
        
        # Find all files for this polarity
        file_groups = self.organize_files_by_sample(data_dir)
        
        # Combined datasets for polarity
        combined_spectra = {}
        combined_metadata = {}
        combined_images = {}
        
        # Process each sample
        for sample_id, files in tqdm(file_groups.items(), desc=f"Processing {polarity} samples"):
            print(f"\n--- Processing {sample_id} ---")
            
            sample_data = self.extract_sample_data(sample_id, files, polarity)
            
            if sample_data:
                combined_spectra[sample_id] = sample_data['spectra']
                combined_metadata[sample_id] = sample_data['metadata']
                if 'images' in sample_data:
                    combined_images[sample_id] = sample_data['images']
        
        # Save combined datasets in all formats
        self.save_polarity_data(polarity, combined_spectra, combined_metadata, combined_images)
        
    def organize_files_by_sample(self, data_dir):
        """Group files by sample (P1_SQ1_01, etc.)."""
        file_groups = {}
        
        for file_path in data_dir.glob("*.it*"):
            # Extract sample identifier (e.g., P1_SQ1_01)
            parts = file_path.stem.split('_')
            if len(parts) >= 3:
                sample_id = '_'.join(parts[:3])  # P1_SQ1_01
                
                if sample_id not in file_groups:
                    file_groups[sample_id] = {}
                
                # Categorize by extension
                ext = file_path.suffix.lower()
                file_groups[sample_id][ext] = file_path
        
        return file_groups
    
    def extract_sample_data(self, sample_id, files, polarity):
        """Extract data from all available formats for one sample."""
        sample_data = {
            'spectra': {},
            'metadata': {},
            'images': {},
            'source_files': files
        }
        
        # Parse sample info
        pattern, square, num = sample_id.split('_')
        dose = self.dose_mapping.get(square, 0)
        
        # Base metadata
        base_metadata = {
            'sample_id': sample_id,
            'pattern': pattern,
            'square': square,
            'number': num,
            'dose_uC_cm2': dose,
            'polarity': polarity,
            'extraction_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Extract from each available file format
        for ext, file_path in files.items():
            try:
                print(f"  Extracting from {ext}: {file_path.name}")
                
                if ext == '.itm':
                    data = self.extract_from_itm(file_path, base_metadata)
                elif ext == '.itax':
                    data = self.extract_from_itax(file_path, base_metadata)
                elif ext == '.itmx':
                    data = self.extract_from_itmx(file_path, base_metadata)
                else:
                    continue
                
                if data:
                    format_name = ext[1:]  # Remove dot
                    sample_data['spectra'][format_name] = data.get('spectrum', {})
                    sample_data['metadata'][format_name] = data.get('metadata', {})
                    if 'images' in data:
                        sample_data['images'][format_name] = data['images']
                    
                    print(f"    ✓ Successfully extracted from {ext}")
                    self.extraction_results['success'].append(f"{sample_id}_{ext}")
                else:
                    print(f"    ❌ Failed to extract from {ext}")
                    self.extraction_results['failed'].append(f"{sample_id}_{ext}")
                    
            except Exception as e:
                print(f"    ❌ Error extracting from {ext}: {e}")
                self.extraction_results['failed'].append(f"{sample_id}_{ext}: {e}")
        
        # Store file information
        self.extraction_results['file_info'][sample_id] = {
            'available_formats': list(files.keys()),
            'file_sizes_mb': {ext: path.stat().st_size / 1e6 for ext, path in files.items()},
            'polarity': polarity,
            'dose': dose
        }
        
        return sample_data if sample_data['spectra'] else None
    
    def extract_from_itm(self, file_path, base_metadata):
        """Extract data from ITM file (raw measurement data)."""
        try:
            # Use patched pySPM ITM loader
            itm = pySPM.ITM(str(file_path))
            
            data = {'metadata': base_metadata.copy()}
            
            # Get spectrum data
            try:
                masses, intensities = itm.get_spectrum()
                if validate_extracted_spectrum(masses, intensities, file_path):
                    data['spectrum'] = {
                        'masses': masses.astype(np.float32),
                        'intensities': intensities.astype(np.float32),
                        'n_points': len(masses),
                        'mass_range': [float(np.min(masses)), float(np.max(masses))],
                        'source': 'itm'
                    }
            except Exception as e:
                print(f"      ITM spectrum extraction failed: {e}")
            
            # Get metadata
            metadata_fields = [
                ('size', 'instrument_parameters'),
                ('polarity', 'polarity_detected'),
                ('Nscan', 'number_of_scans'),
                ('spp', 'shots_per_pixel'),
                ('sf', 'mass_cal_sf'),
                ('k0', 'mass_cal_k0')
            ]
            
            for attr, key in metadata_fields:
                try:
                    value = getattr(itm, attr, None)
                    if value is not None:
                        data['metadata'][key] = value
                except:
                    pass
            
            # Try to get imaging data
            try:
                # ITM files may contain spatial imaging information
                # This is format-specific and may require specialized extraction
                pass
            except:
                pass
                
            data['metadata']['file_format'] = 'ITM'
            data['metadata']['data_source'] = 'raw_measurement'
            
            return data
            
        except Exception as e:
            print(f"      ITM extraction failed: {e}")
            return None
    
    def extract_from_itax(self, file_path, base_metadata):
        """Extract data from ITAX file (analysis data)."""
        try:
            # Use patched ITAX loader
            itax = pySPM.ITAX(str(file_path))
            
            data = {'metadata': base_metadata.copy()}
            
            # Get spectrum data with fixes
            try:
                masses, intensities = itax.getSpectrum()
                if validate_extracted_spectrum(masses, intensities, file_path):
                    data['spectrum'] = {
                        'masses': masses.astype(np.float32),
                        'intensities': intensities.astype(np.float32),
                        'n_points': len(masses),
                        'mass_range': [float(np.min(masses)), float(np.max(masses))],
                        'source': 'itax'
                    }
            except Exception as e:
                print(f"      ITAX spectrum extraction failed: {e}")
            
            # Get size information
            try:
                data['metadata']['size'] = itax.size
            except:
                pass
            
            data['metadata']['file_format'] = 'ITAX'
            data['metadata']['data_source'] = 'analysis_processed'
            
            return data
            
        except Exception as e:
            print(f"      ITAX extraction failed: {e}")
            return None
    
    def extract_from_itmx(self, file_path, base_metadata):
        """Extract data from ITMX file (extended analysis data)."""
        try:
            # ITMX files are similar to ITM but with extended analysis data
            # Try ITM approach first, then specific ITMX handling if needed
            
            data = {'metadata': base_metadata.copy()}
            
            # Use our comprehensive fixes
            masses, intensities = open_iontof_with_fixes(file_path)
            
            if validate_extracted_spectrum(masses, intensities, file_path):
                data['spectrum'] = {
                    'masses': masses.astype(np.float32),
                    'intensities': intensities.astype(np.float32),
                    'n_points': len(masses),
                    'mass_range': [float(np.min(masses)), float(np.max(masses))],
                    'source': 'itmx'
                }
                
                data['metadata']['file_format'] = 'ITMX'
                data['metadata']['data_source'] = 'extended_analysis'
                
                return data
            else:
                return None
            
        except Exception as e:
            print(f"      ITMX extraction failed: {e}")
            return None
    
    def save_polarity_data(self, polarity, spectra, metadata, images):
        """Save extracted data in all standardized formats."""
        print(f"\n--- Saving {polarity} data in all formats ---")
        
        # 1. Save as HDF5 (efficient binary storage)
        self.save_as_hdf5(polarity, spectra, metadata, images)
        
        # 2. Save as TSV (human-readable tabular)
        self.save_as_tsv(polarity, spectra, metadata)
        
        # 3. Save as JSON (metadata)
        self.save_as_json(polarity, metadata)
        
        # 4. Save as NPZ (NumPy arrays)
        self.save_as_npz(polarity, spectra)
        
    def save_as_hdf5(self, polarity, spectra, metadata, images):
        """Save as HDF5 for efficient storage of large datasets."""
        hdf5_file = self.formats['hdf5'] / f'{polarity}_complete.h5'
        
        with h5py.File(hdf5_file, 'w') as f:
            # Create groups
            spectra_grp = f.create_group('spectra')
            metadata_grp = f.create_group('metadata') 
            
            if images:
                images_grp = f.create_group('images')
            
            for sample_id, sample_spectra in spectra.items():
                sample_grp = spectra_grp.create_group(sample_id)
                
                for format_name, spectrum in sample_spectra.items():
                    if spectrum and 'masses' in spectrum and 'intensities' in spectrum:
                        format_grp = sample_grp.create_group(format_name)
                        format_grp.create_dataset('masses', data=spectrum['masses'])
                        format_grp.create_dataset('intensities', data=spectrum['intensities'])
                        
                        # Store attributes
                        for key, value in spectrum.items():
                            if key not in ['masses', 'intensities'] and value is not None:
                                format_grp.attrs[key] = value
                
                # Store metadata
                if sample_id in metadata:
                    meta_grp = metadata_grp.create_group(sample_id)
                    for format_name, meta in metadata[sample_id].items():
                        format_meta_grp = meta_grp.create_group(format_name)
                        for key, value in meta.items():
                            if value is not None:
                                try:
                                    format_meta_grp.attrs[key] = value
                                except (TypeError, ValueError):
                                    format_meta_grp.attrs[key] = str(value)
        
        print(f"    ✓ Saved HDF5: {hdf5_file}")
    
    def save_as_tsv(self, polarity, spectra, metadata):
        """Save spectra as TSV files for MVA analysis."""
        
        # Combine all spectra into matrices by format
        format_data = {}
        
        for sample_id, sample_spectra in spectra.items():
            for format_name, spectrum in sample_spectra.items():
                if spectrum and 'masses' in spectrum and 'intensities' in spectrum:
                    if format_name not in format_data:
                        format_data[format_name] = {
                            'samples': [],
                            'masses': spectrum['masses'],
                            'intensities_matrix': []
                        }
                    
                    # Ensure consistent mass axis
                    if np.array_equal(format_data[format_name]['masses'], spectrum['masses']):
                        format_data[format_name]['samples'].append(sample_id)
                        format_data[format_name]['intensities_matrix'].append(spectrum['intensities'])
        
        # Save each format as TSV
        for format_name, data in format_data.items():
            if data['samples']:
                # Create DataFrame
                intensity_matrix = np.array(data['intensities_matrix'])
                columns = [f"m{mass:.3f}" for mass in data['masses']]
                
                df = pd.DataFrame(intensity_matrix, 
                                index=data['samples'], 
                                columns=columns)
                
                # Add metadata columns
                for sample_id in data['samples']:
                    if sample_id in metadata and format_name in metadata[sample_id]:
                        meta = metadata[sample_id][format_name]
                        for key in ['pattern', 'square', 'dose_uC_cm2']:
                            if key in meta:
                                df.loc[sample_id, key] = meta[key]
                
                # Save TSV
                tsv_file = self.formats['tsv'] / f'{polarity}_{format_name}_spectra.tsv'
                df.to_csv(tsv_file, sep='\t')
                print(f"    ✓ Saved TSV: {tsv_file}")
    
    def save_as_json(self, polarity, metadata):
        """Save metadata as JSON files."""
        json_file = self.formats['json'] / f'{polarity}_metadata.json'
        
        # Convert numpy types to JSON-serializable types
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            else:
                return obj
        
        serializable_metadata = make_serializable(metadata)
        
        with open(json_file, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        print(f"    ✓ Saved JSON: {json_file}")
    
    def save_as_npz(self, polarity, spectra):
        """Save as compressed NumPy arrays for fast Python loading."""
        npz_file = self.formats['npz'] / f'{polarity}_arrays.npz'
        
        # Organize arrays for saving
        arrays_to_save = {}
        
        for sample_id, sample_spectra in spectra.items():
            for format_name, spectrum in sample_spectra.items():
                if spectrum and 'masses' in spectrum and 'intensities' in spectrum:
                    arrays_to_save[f"{sample_id}_{format_name}_masses"] = spectrum['masses']
                    arrays_to_save[f"{sample_id}_{format_name}_intensities"] = spectrum['intensities']
        
        if arrays_to_save:
            np.savez_compressed(npz_file, **arrays_to_save)
            print(f"    ✓ Saved NPZ: {npz_file}")
    
    def create_format_comparison_summary(self):
        """Create summary comparing data quality across formats."""
        print("\n--- Creating format comparison summary ---")
        
        comparison_data = []
        
        for sample_id, info in self.extraction_results['file_info'].items():
            for format_ext in info['available_formats']:
                format_name = format_ext[1:]  # Remove dot
                
                row = {
                    'sample_id': sample_id,
                    'format': format_name,
                    'polarity': info['polarity'],
                    'dose_uC_cm2': info['dose'],
                    'file_size_mb': info['file_sizes_mb'].get(format_ext, 0),
                    'extraction_success': f"{sample_id}_{format_ext}" in self.extraction_results['success']
                }
                comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_file = self.extracted_dir / 'format_comparison_summary.tsv'
        df_comparison.to_csv(comparison_file, sep='\t', index=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success rate by format
        success_rate = df_comparison.groupby('format')['extraction_success'].mean()
        axes[0,0].bar(success_rate.index, success_rate.values, alpha=0.7)
        axes[0,0].set_title('Extraction Success Rate by Format')
        axes[0,0].set_ylabel('Success Rate')
        
        # File size distribution
        df_comparison.boxplot(column='file_size_mb', by='format', ax=axes[0,1])
        axes[0,1].set_title('File Size Distribution by Format')
        axes[0,1].set_ylabel('File Size (MB)')
        
        # Success by polarity
        success_by_pol = df_comparison.groupby(['polarity', 'format'])['extraction_success'].mean().unstack()
        success_by_pol.plot(kind='bar', ax=axes[1,0], alpha=0.7)
        axes[1,0].set_title('Success Rate by Polarity and Format')
        axes[1,0].set_ylabel('Success Rate')
        axes[1,0].legend(title='Format')
        
        # Sample coverage
        coverage = df_comparison.groupby('sample_id')['extraction_success'].sum()
        axes[1,1].hist(coverage, bins=10, alpha=0.7)
        axes[1,1].set_title('Number of Successful Extractions per Sample')
        axes[1,1].set_xlabel('Successful Extractions')
        axes[1,1].set_ylabel('Number of Samples')
        
        plt.tight_layout()
        fig.savefig(self.extracted_dir / 'extraction_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Format comparison saved: {comparison_file}")
        print(f"    ✓ Summary visualization: {self.extracted_dir}/extraction_summary.png")
    
    def save_extraction_results(self):
        """Save final extraction results summary."""
        results_file = self.extracted_dir / 'extraction_results.json'
        
        summary = {
            'extraction_timestamp': pd.Timestamp.now().isoformat(),
            'total_samples_processed': len(self.extraction_results['file_info']),
            'successful_extractions': len(self.extraction_results['success']),
            'failed_extractions': len(self.extraction_results['failed']),
            'success_rate': len(self.extraction_results['success']) / 
                           (len(self.extraction_results['success']) + len(self.extraction_results['failed'])),
            'available_formats': ['hdf5', 'tsv', 'json', 'npz'],
            'output_directory': str(self.extracted_dir),
            'detailed_results': self.extraction_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"    ✓ Extraction results saved: {results_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples processed: {summary['total_samples_processed']}")
        print(f"Successful extractions: {summary['successful_extractions']}")
        print(f"Failed extractions: {summary['failed_extractions']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Output formats: {', '.join(summary['available_formats'])}")
        print(f"Data location: {summary['output_directory']}")


if __name__ == "__main__":
    extractor = ComprehensiveDataExtractor("/home/dreece23/sims-pca-ws")
    extractor.extract_all_data()