"""
Principal Component Analysis utilities for ToF-SIMS data.

This module provides PCA-specific functions for data preprocessing, component analysis,
statistical testing, and visualization support for dose-response studies.

Version: 0.1.0
"""

def mean_center_rows(data, axis=0):
    """
    Mean-center data matrix along specified axis.
    
    Removes mean values to prepare data for PCA analysis while preserving
    variance structure needed for dose-response interpretation.
    
    Parameters
    ----------
    data : array_like, shape (n_samples, n_features)
        Input data matrix to be mean-centered
    axis : int, default=0
        Axis along which to compute means (0=center features, 1=center samples)
        
    Returns
    -------
    centered_data : ndarray
        Mean-centered data matrix
    means : ndarray
        Mean values that were subtracted
    centering_info : dict
        Metadata about centering operation (axis, n_zeros, mean_of_means)
        
    Notes
    -----
    Preserves original data variance structure essential for dose-response PCA.
    """
    pass


def align_pc_signs_to_dose(scores, loadings, dose_levels, pc_indices=None):
    """
    Align principal component signs for consistent dose interpretation.
    
    Adjusts PC signs so that increasing dose corresponds to positive score direction,
    ensuring consistent interpretation across different PCA runs and datasets.
    
    Parameters
    ----------
    scores : array_like, shape (n_samples, n_components)
        PCA score matrix
    loadings : array_like, shape (n_features, n_components)  
        PCA loading matrix
    dose_levels : array_like
        Dose values corresponding to each sample
    pc_indices : list of int, optional
        Which PCs to align (default: first 3 components)
        
    Returns
    -------
    aligned_scores : ndarray
        Sign-corrected score matrix
    aligned_loadings : ndarray
        Sign-corrected loading matrix  
    sign_flips : ndarray of bool
        Boolean array indicating which PCs were sign-flipped
    dose_correlations : ndarray
        Spearman correlations between PC scores and dose levels
        
    Notes
    -----
    Uses Spearman correlation to determine optimal sign for dose monotonicity.
    """
    pass


def compute_hotelling_t2_ellipse(scores, confidence_level=0.95, pc_x=0, pc_y=1):
    """
    Compute Hotelling's T² confidence ellipse for PCA scores.
    
    Calculates multivariate confidence ellipse for visualizing sample clustering
    and identifying outliers in PC space.
    
    Parameters
    ----------
    scores : array_like, shape (n_samples, n_components)
        PCA score matrix
    confidence_level : float, default=0.95
        Confidence level for ellipse (0.90, 0.95, or 0.99)
    pc_x : int, default=0
        PC index for x-axis
    pc_y : int, default=1  
        PC index for y-axis
        
    Returns
    -------
    ellipse_points : ndarray, shape (n_points, 2)
        X,Y coordinates of ellipse boundary
    center : tuple
        Ellipse center coordinates (mean_x, mean_y)
    axes_lengths : tuple
        Semi-major and semi-minor axis lengths
    rotation_angle : float
        Ellipse rotation angle in radians
        
    Notes
    -----
    Uses F-distribution critical values for multivariate confidence regions.
    """
    pass


def create_loadings_table(loadings, feature_names, pc_indices=None, top_n=20):
    """
    Create formatted loading tables for PC interpretation.
    
    Generates tables of highest-magnitude loadings for each principal component,
    facilitating chemical interpretation of dose-response patterns.
    
    Parameters
    ----------
    loadings : array_like, shape (n_features, n_components)
        PCA loading matrix
    feature_names : array_like
        Feature names (e.g., m/z values)
    pc_indices : list of int, optional
        Which PCs to tabulate (default: first 5 components)  
    top_n : int, default=20
        Number of top loadings per component
        
    Returns
    -------
    loading_tables : dict
        Dictionary with PC names as keys, DataFrames as values
    summary_stats : dict
        Loading distribution statistics for each PC
    chemical_families : dict
        Grouped loadings by m/z ranges for chemical interpretation
        
    Notes
    -----
    Tables include loading values, rankings, and cumulative contribution percentages.
    """
    pass


def overlay_raw_spectra_on_loadings(loadings, raw_spectra, mz_values, pc_index=0, 
                                   overlay_alpha=0.3):
    """
    Overlay raw mass spectra on PC loading plots for interpretation.
    
    Combines PCA loadings with representative raw spectra to aid in
    chemical identification of dose-responsive peaks.
    
    Parameters
    ----------
    loadings : array_like, shape (n_features, n_components)
        PCA loading matrix
    raw_spectra : array_like, shape (n_samples, n_features)
        Raw or normalized mass spectra
    mz_values : array_like
        m/z values corresponding to features
    pc_index : int, default=0
        Which principal component to plot
    overlay_alpha : float, default=0.3
        Transparency for spectrum overlay
        
    Returns
    -------
    plot_data : dict
        Formatted data for loading/spectrum overlay plot
    peak_annotations : list
        Suggested peak annotations based on loading magnitudes
    spectrum_metadata : dict
        Information about overlaid spectra (mean, dose levels, etc.)
        
    Notes
    -----
    Automatically selects representative spectra based on PC score extremes.
    """
    pass


def compute_explained_variance_ci(explained_variance, n_samples, confidence_level=0.95):
    """
    Compute confidence intervals for explained variance ratios.
    
    Estimates uncertainty in PC explained variance using bootstrap or
    analytical methods for robust component selection.
    
    Parameters
    ----------
    explained_variance : array_like
        Explained variance ratios for each PC
    n_samples : int
        Number of samples used in PCA
    confidence_level : float, default=0.95
        Confidence level for intervals
        
    Returns
    -------
    ci_lower : ndarray
        Lower confidence bounds
    ci_upper : ndarray  
        Upper confidence bounds
    standard_errors : ndarray
        Standard errors for explained variance estimates
    
    Notes
    -----
    Uses asymptotic theory for large samples, bootstrap for small samples.
    """
    pass