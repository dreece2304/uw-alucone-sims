#!/usr/bin/env python3
"""
Phase 2: ROI Definition and Overlays
- Define regions of interest for dose areas
- Generate ROI coordinate schema
- Create visualization overlays
- Save ROI definitions for extraction
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_config() -> Dict:
    """Load project configuration"""
    config_path = Path('roi/config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def define_dose_rois() -> Dict:
    """Define ROI coordinates for each dose area"""
    # Based on typical ToF-SIMS imaging layout for dose series
    # Assuming 256x256 pixel images with dose areas arranged in patterns
    
    # Define ROI coordinates (x, y, width, height) for each dose area
    # These coordinates represent typical spatial arrangement in dose series
    rois = {
        'SQ1r_500uC': {
            'coordinates': [40, 40, 40, 40],  # x, y, width, height
            'center': [60, 60],
            'dose': 500,
            'description': 'Low dose reference area'
        },
        'SQ1_2000uC': {
            'coordinates': [120, 40, 40, 40],
            'center': [140, 60], 
            'dose': 2000,
            'description': 'Low dose area'
        },
        'SQ2_5000uC': {
            'coordinates': [40, 120, 40, 40],
            'center': [60, 140],
            'dose': 5000,
            'description': 'Medium dose area'
        },
        'SQ3_10000uC': {
            'coordinates': [120, 120, 40, 40],
            'center': [140, 140],
            'dose': 10000,
            'description': 'High dose area'
        },
        'SQ4_15000uC': {
            'coordinates': [180, 80, 40, 40],
            'center': [200, 100],
            'dose': 15000,
            'description': 'Maximum dose area'
        }
    }
    
    return rois

def create_roi_schema(rois: Dict, config: Dict) -> Dict:
    """Create comprehensive ROI schema"""
    
    schema = {
        'project_info': {
            'name': config['project_name'],
            'date': config['analysis_date'],
            'image_dimensions': {
                'pixels_x': 256,
                'pixels_y': 256,
                'total_pixels': 65536
            }
        },
        'dose_mapping': config['dose_mapping'],
        'roi_definitions': rois,
        'extraction_parameters': {
            'unit_mass_binning': True,
            'mass_range': [0, 300],
            'normalization_method': 'roi_sum',
            'background_subtraction': False
        },
        'file_mapping': {
            'positive_patterns': ['P1', 'P2', 'P3'],
            'negative_patterns': ['P1', 'P2', 'P3'],
            'file_suffix_positive': '_01.itm',
            'file_suffix_negative': '_06.itm'
        }
    }
    
    return schema

def create_roi_overlay(rois: Dict, output_path: Path, title: str = "ROI Layout") -> None:
    """Create ROI visualization overlay"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Create base image (simulated)
    base_image = np.random.rand(256, 256) * 0.3 + 0.2  # Simulated background
    ax.imshow(base_image, cmap='gray', alpha=0.7, extent=[0, 256, 256, 0])
    
    # Color scheme for different doses
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Draw ROI rectangles
    for i, (roi_name, roi_data) in enumerate(rois.items()):
        x, y, w, h = roi_data['coordinates']
        dose = roi_data['dose']
        
        # Create rectangle
        rect = patches.Rectangle((x, y), w, h, 
                               linewidth=2, 
                               edgecolor=colors[i % len(colors)], 
                               facecolor=colors[i % len(colors)],
                               alpha=0.3)
        ax.add_patch(rect)
        
        # Add dose label
        center_x, center_y = roi_data['center']
        ax.text(center_x, center_y, f'{dose}\nµC/cm²', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add ROI name
        ax.text(x, y-8, roi_name.replace('_', ' '), 
               ha='left', va='bottom', fontsize=10, fontweight='bold',
               color=colors[i % len(colors)])
    
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)  # Flip y-axis for image coordinates
    ax.set_xlabel('X Pixels', fontsize=12)
    ax.set_ylabel('Y Pixels', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add colorbar legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, (roi_name, roi_data) in enumerate(rois.items()):
        dose = roi_data['dose']
        legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                    markerfacecolor=colors[i % len(colors)], 
                                    markersize=10, alpha=0.7,
                                    label=f'{dose} µC/cm²'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ROI overlay saved: {output_path}")

def create_pattern_overlays(rois: Dict, config: Dict) -> None:
    """Create ROI overlays for each pattern"""
    
    overlays_dir = Path('roi/overlays')
    overlays_dir.mkdir(exist_ok=True)
    
    patterns = ['P1', 'P2', 'P3']
    polarities = ['positive', 'negative']
    
    for pattern in patterns:
        for polarity in polarities:
            output_file = overlays_dir / f'{pattern}_{polarity}_roi_overlay.png'
            title = f'ROI Layout - Pattern {pattern} ({polarity.capitalize()} Ions)'
            create_roi_overlay(rois, output_file, title)

def validate_rois(rois: Dict) -> bool:
    """Validate ROI definitions"""
    
    print("Validating ROI definitions...")
    
    valid = True
    
    for roi_name, roi_data in rois.items():
        x, y, w, h = roi_data['coordinates']
        
        # Check bounds
        if x < 0 or y < 0 or (x + w) > 256 or (y + h) > 256:
            print(f"  ❌ {roi_name}: ROI extends beyond image boundaries")
            valid = False
            continue
            
        # Check minimum size
        if w < 10 or h < 10:
            print(f"  ⚠️  {roi_name}: ROI size very small ({w}x{h} pixels)")
            
        # Check area
        area = w * h
        if area < 100:
            print(f"  ⚠️  {roi_name}: Small ROI area ({area} pixels)")
        
        print(f"  ✓ {roi_name}: Valid ({w}x{h} pixels, {area} area)")
    
    # Check for overlaps
    print("\nChecking for ROI overlaps...")
    roi_names = list(rois.keys())
    for i, roi1_name in enumerate(roi_names):
        for roi2_name in roi_names[i+1:]:
            roi1 = rois[roi1_name]
            roi2 = rois[roi2_name]
            
            x1, y1, w1, h1 = roi1['coordinates']
            x2, y2, w2, h2 = roi2['coordinates']
            
            # Check overlap
            if not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1):
                print(f"  ⚠️  Overlap detected: {roi1_name} and {roi2_name}")
            else:
                print(f"  ✓ No overlap: {roi1_name} and {roi2_name}")
    
    return valid

def main():
    """Execute Phase 2 analysis"""
    print("=== Phase 2: ROI Definition and Overlays ===")
    
    # Load configuration
    config = load_config()
    print("Configuration loaded")
    
    # Define ROIs
    print("\nDefining ROI coordinates...")
    rois = define_dose_rois()
    print(f"Defined {len(rois)} ROI areas:")
    for roi_name, roi_data in rois.items():
        dose = roi_data['dose']
        x, y, w, h = roi_data['coordinates']
        area = w * h
        print(f"  {roi_name}: {dose} µC/cm², {w}×{h} pixels ({area} area)")
    
    # Validate ROIs
    print()
    if not validate_rois(rois):
        print("❌ ROI validation failed - please review coordinates")
        return
    
    print("✓ ROI validation passed")
    
    # Create ROI schema
    print("\nCreating ROI schema...")
    schema = create_roi_schema(rois, config)
    
    # Save schema
    schema_path = Path('roi/schema.json')
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"ROI schema saved: {schema_path}")
    
    # Create overlays
    print("\nGenerating ROI overlays...")
    create_pattern_overlays(rois, config)
    
    # Create summary overlay
    summary_overlay = Path('roi/overlays/roi_summary.png')
    create_roi_overlay(rois, summary_overlay, "ROI Layout Summary")
    
    print(f"\nGenerated overlay files:")
    overlays_dir = Path('roi/overlays')
    if overlays_dir.exists():
        overlay_files = list(overlays_dir.glob('*.png'))
        for file in sorted(overlay_files):
            print(f"  {file}")
    
    print("\n=== Phase 2 Complete ===")
    print("Generated files:")
    print("  - roi/schema.json")
    print(f"  - {len(list(overlays_dir.glob('*.png')))} overlay images")
    
    return schema

if __name__ == "__main__":
    schema = main()