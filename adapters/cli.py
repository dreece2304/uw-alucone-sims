"""
Command line interface for adapters package.

Provides Typer-based CLI commands for converting IONTOF and imzML data
to PNNL SIMS-PCA format and generating catalog files.
"""

from pathlib import Path
from typing import List, Optional
import sys

try:
    import typer
except ImportError:
    print("ERROR: typer is required for CLI functionality.")
    print("Install with: pip install typer")
    sys.exit(1)

from .catalog_maker import (
    create_catalog_from_file_list, 
    create_catalog_from_names,
    parse_sample_mapping,
    validate_catalog,
    extract_sample_numbers_from_tsv
)
from .binning import validate_pnnl_tsv

app = typer.Typer(help="Convert ToF-SIMS data to PNNL SIMS-PCA format", add_completion=False)

@app.command("iontof")
def cmd_iontof(
    files: List[Path] = typer.Argument(..., help="IONTOF files (.ita/.itm/.itax)"),
    pol: str = typer.Option("P", "--pol", help="Polarity label for column names (P or N)"),
    out_tsv: Path = typer.Option(..., "--out-tsv", help="Output TSV path"),
    names: Optional[str] = typer.Option(None, "--names", help="Comma-separated sample labels"),
    prefer: str = typer.Option("auto", "--prefer", help="Preferred spectrum source: auto|ita|itax|itm"),
):
    try:
        from .iontof_to_pnnl import iontof_to_pnnl_tsv
    except ImportError:
        typer.echo("Error: iontof converter requires pySPM. Install with: pip install pySPM", err=True)
        raise typer.Exit(1)
    lbls = [s.strip() for s in names.split(",")] if names else None
    try:
        out = iontof_to_pnnl_tsv(files, out_tsv, polarity=pol, names=lbls, prefer=prefer)
    except Exception as e:
        typer.echo(f"IONTOF conversion failed: {e}", err=True)
        raise typer.Exit(2)
    typer.echo(f"Wrote TSV: {out}")

@app.command("debug-iontof")
def cmd_debug_iontof(
    file: Path = typer.Argument(..., help="Path to .ita/.itm/.itax"),
    prefer: str = typer.Option("auto", "--prefer"),
):
    try:
        from .iontof_to_pnnl import _try_paths
    except ImportError:
        typer.echo("pySPM not installed: pip install pySPM", err=True)
        raise typer.Exit(1)
    try:
        m, I, used = _try_paths(file, prefer=prefer)
        typer.echo(f"OK {used}: points={m.size}, m/z=[{float(m.min()):.2f}, {float(m.max()):.2f}]")
    except Exception as e:
        typer.echo(f"Debug error: {e}", err=True)
        raise typer.Exit(2)

@app.command("legacy-iontof")
def convert_iontof(
    files: List[Path] = typer.Argument(..., help="IONTOF data files (.ita, .itm, .itax)"),
    pol: str = typer.Option("P", "--pol", help="Polarity: P (positive) or N (negative)"),
    out_tsv: Path = typer.Option(..., "--out-tsv", help="Output TSV file path"),
    names: Optional[str] = typer.Option(None, "--names", help="Comma-separated custom sample names"),
    validate_output: bool = typer.Option(True, "--validate/--no-validate", help="Validate output TSV"),
    create_catalog: bool = typer.Option(False, "--catalog", help="Also create catalog.csv"),
    operator: str = typer.Option("Unknown", "--operator", help="Operator name for catalog"),
    date: Optional[str] = typer.Option(None, "--date", help="Testing date (YYYY-MM-DD) for catalog"),
) -> None:
    """
    Legacy IONTOF converter with full validation and catalog options.
    """
    # Validate polarity
    if pol not in ['P', 'N']:
        typer.echo("Error: Polarity must be 'P' (positive) or 'N' (negative)", err=True)
        raise typer.Exit(1)
    
    # Validate input files
    valid_files = []
    for file_path in files:
        if not file_path.exists():
            typer.echo(f"Warning: File not found, skipping: {file_path}", err=True)
            continue
        
        if file_path.suffix.lower() not in ['.ita', '.itm', '.itax']:
            typer.echo(f"Warning: Unsupported file type, skipping: {file_path}", err=True)
            continue
        
        valid_files.append(file_path)
    
    if not valid_files:
        typer.echo("Error: No valid IONTOF files found", err=True)
        raise typer.Exit(1)
    
    # Parse sample names
    sample_names = None
    if names:
        sample_names = [name.strip() for name in names.split(',')]
        if len(sample_names) != len(valid_files):
            typer.echo("Error: Number of sample names must match number of files", err=True)
            raise typer.Exit(1)
    
    # Create output directory if needed
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Lazy import IONTOF backend
        try:
            from .iontof_to_pnnl import iontof_to_pnnl_tsv
        except ImportError as e:
            typer.echo(f"Error: IONTOF backend not available: {e}", err=True)
            typer.echo("Install pySPM to enable IONTOF support: pip install pySPM", err=True)
            raise typer.Exit(1)
        
        # Convert files
        typer.echo(f"Converting {len(valid_files)} IONTOF files...")
        iontof_to_pnnl_tsv(valid_files, out_tsv, pol, sample_names)
        
        # Validate output if requested
        if validate_output:
            typer.echo("Validating output TSV...")
            if not validate_pnnl_tsv(out_tsv):
                typer.echo("Warning: Output TSV validation failed", err=True)
        
        # Create catalog if requested
        if create_catalog:
            catalog_path = out_tsv.parent / "catalog.csv"
            typer.echo(f"Creating catalog: {catalog_path}")
            create_catalog_from_file_list(
                valid_files, operator=operator, testing_date=date,
                sample_names=sample_names, output_path=catalog_path
            )
        
        typer.echo(f"✓ Successfully created: {out_tsv}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("imzml")
def convert_imzml(
    files: List[Path] = typer.Argument(..., help="imzML data files"),
    pol: str = typer.Option("P", "--pol", help="Polarity: P (positive) or N (negative)"),
    out_tsv: Path = typer.Option(..., "--out-tsv", help="Output TSV file path"),
    names: Optional[str] = typer.Option(None, "--names", help="Comma-separated custom sample names"),
    subset: Optional[int] = typer.Option(None, "--subset", help="Randomly sample N pixels per file for speed"),
    validate_output: bool = typer.Option(True, "--validate/--no-validate", help="Validate output TSV"),
    create_catalog: bool = typer.Option(False, "--catalog", help="Also create catalog.csv"),
    operator: str = typer.Option("Unknown", "--operator", help="Operator name for catalog"),
    date: Optional[str] = typer.Option(None, "--date", help="Testing date (YYYY-MM-DD) for catalog"),
) -> None:
    """
    Convert imzML data files to PNNL SIMS-PCA TSV format.
    
    Each imzML file represents one sample. All pixels in the file are summed
    to create a representative spectrum for that sample.
    
    Example:
        adapters imzml --pol P --out-tsv data.tsv sample1.imzML sample2.imzML
        adapters imzml --pol P --out-tsv data.tsv --subset 1000 large_file.imzML
    """
    # Validate polarity
    if pol not in ['P', 'N']:
        typer.echo("Error: Polarity must be 'P' (positive) or 'N' (negative)", err=True)
        raise typer.Exit(1)
    
    # Validate input files
    valid_files = []
    for file_path in files:
        if not file_path.exists():
            typer.echo(f"Warning: File not found, skipping: {file_path}", err=True)
            continue
        
        if file_path.suffix.lower() not in ['.imzml']:
            typer.echo(f"Warning: Expected .imzML file, skipping: {file_path}", err=True)
            continue
        
        valid_files.append(file_path)
    
    if not valid_files:
        typer.echo("Error: No valid imzML files found", err=True)
        raise typer.Exit(1)
    
    # Parse sample names
    sample_names = None
    if names:
        sample_names = [name.strip() for name in names.split(',')]
        if len(sample_names) != len(valid_files):
            typer.echo("Error: Number of sample names must match number of files", err=True)
            raise typer.Exit(1)
    
    # Create output directory if needed
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Lazy import imzML backend
        try:
            from .imzml_to_pnnl import imzml_to_pnnl_tsv
        except ImportError as e:
            typer.echo(f"Error: imzML backend not available: {e}", err=True)
            typer.echo("Install pyimzML to enable imzML support: pip install pyimzML", err=True)
            raise typer.Exit(1)
        
        # Convert files
        typer.echo(f"Converting {len(valid_files)} imzML files...")
        if subset:
            typer.echo(f"Using random subset of {subset} pixels per file")
        
        imzml_to_pnnl_tsv(valid_files, out_tsv, pol, sample_names, subset)
        
        # Validate output if requested
        if validate_output:
            typer.echo("Validating output TSV...")
            if not validate_pnnl_tsv(out_tsv):
                typer.echo("Warning: Output TSV validation failed", err=True)
        
        # Create catalog if requested
        if create_catalog:
            catalog_path = out_tsv.parent / "catalog.csv"
            typer.echo(f"Creating catalog: {catalog_path}")
            create_catalog_from_file_list(
                valid_files, operator=operator, testing_date=date,
                sample_names=sample_names, output_path=catalog_path
            )
        
        typer.echo(f"✓ Successfully created: {out_tsv}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("catalog")
def create_catalog(
    out: Path = typer.Option("catalog.csv", "--out", help="Output catalog.csv file path"),
    names: Optional[str] = typer.Option(None, "--names", help="Sample mapping: '1:Sample A,2:Sample B'"),
    tsv_file: Optional[Path] = typer.Option(None, "--from-tsv", help="Extract sample numbers from existing TSV"),
    operator: str = typer.Option("Unknown", "--operator", help="Operator name"),
    date: Optional[str] = typer.Option(None, "--date", help="Testing date (YYYY-MM-DD), defaults to today"),
    validate_output: bool = typer.Option(True, "--validate/--no-validate", help="Validate output catalog"),
) -> None:
    """
    Create a catalog.csv file for PNNL SIMS-PCA analysis.
    
    Examples:
        adapters catalog --names "1:Sample A,2:Sample B" --operator "John Doe"
        adapters catalog --from-tsv data.tsv --names "1:Control,2:Treatment" 
    """
    # Create output directory if needed  
    out.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if tsv_file:
            # Extract sample numbers from existing TSV
            if not tsv_file.exists():
                typer.echo(f"Error: TSV file not found: {tsv_file}", err=True)
                raise typer.Exit(1)
            
            sample_numbers = extract_sample_numbers_from_tsv(tsv_file)
            typer.echo(f"Found sample numbers in TSV: {sample_numbers}")
            
            if names:
                # Use provided names with TSV sample numbers
                sample_mapping = parse_sample_mapping(names)
                
                # Check that provided numbers match TSV
                if set(sample_mapping.keys()) != set(sample_numbers):
                    typer.echo("Warning: Sample numbers in --names don't match TSV file", err=True)
                    typer.echo(f"  TSV has: {sorted(sample_numbers)}")
                    typer.echo(f"  Names provided for: {sorted(sample_mapping.keys())}")
            else:
                # Generate default names
                sample_mapping = {num: f"Sample {num}" for num in sample_numbers}
        
        elif names:
            # Parse sample mapping from string
            sample_mapping = parse_sample_mapping(names)
        
        else:
            typer.echo("Error: Must provide either --names or --from-tsv", err=True)
            raise typer.Exit(1)
        
        # Create catalog
        create_catalog_from_names(sample_mapping, operator, date, out)
        
        # Validate output if requested
        if validate_output:
            typer.echo("Validating catalog...")
            if not validate_catalog(out):
                typer.echo("Warning: Catalog validation failed", err=True)
        
        typer.echo(f"✓ Successfully created: {out}")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("validate")
def validate_files(
    tsv: Optional[Path] = typer.Option(None, "--tsv", help="Validate TSV file"),
    catalog: Optional[Path] = typer.Option(None, "--catalog", help="Validate catalog file"),
    check_consistency: bool = typer.Option(False, "--check-consistency", help="Check TSV/catalog consistency"),
) -> None:
    """
    Validate PNNL SIMS-PCA format files.
    
    Examples:
        adapters validate --tsv data.tsv --catalog catalog.csv --check-consistency
        adapters validate --tsv data.tsv
    """
    if not tsv and not catalog:
        typer.echo("Error: Specify at least one file to validate with --tsv or --catalog", err=True)
        raise typer.Exit(1)
    
    all_valid = True
    
    # Validate TSV
    if tsv:
        typer.echo(f"Validating TSV: {tsv}")
        if not validate_pnnl_tsv(tsv):
            all_valid = False
    
    # Validate catalog
    if catalog:
        typer.echo(f"Validating catalog: {catalog}")
        if not validate_catalog(catalog):
            all_valid = False
    
    # Check consistency between files
    if check_consistency and tsv and catalog:
        typer.echo("Checking TSV/catalog consistency...")
        
        try:
            tsv_sample_numbers = extract_sample_numbers_from_tsv(tsv)
            
            # Read catalog sample numbers
            import csv
            catalog_sample_numbers = []
            with open(catalog, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    catalog_sample_numbers.append(int(row["Sample #"]))
            
            if set(tsv_sample_numbers) == set(catalog_sample_numbers):
                typer.echo("✓ Sample numbers consistent between TSV and catalog")
            else:
                typer.echo("✗ Sample number mismatch between TSV and catalog", err=True)
                typer.echo(f"  TSV: {sorted(tsv_sample_numbers)}")
                typer.echo(f"  Catalog: {sorted(catalog_sample_numbers)}")
                all_valid = False
        
        except Exception as e:
            typer.echo(f"Error checking consistency: {e}", err=True)
            all_valid = False
    
    if all_valid:
        typer.echo("✓ All validations passed")
    else:
        typer.echo("✗ Some validations failed", err=True)
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()