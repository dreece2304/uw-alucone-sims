"""
Adapters package for converting IONTOF and imzML data to PNNL SIMS-PCA format.

This package provides tools to convert various ToF-SIMS data formats into the 
tab-delimited format expected by PNNL's ATOFSIMSCLASS PCA analysis tool.
"""

__version__ = "0.1.0"

from . import binning
from . import iontof_to_pnnl
from . import imzml_to_pnnl
from . import catalog_maker

__all__ = ["binning", "iontof_to_pnnl", "imzml_to_pnnl", "catalog_maker"]