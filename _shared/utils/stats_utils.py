"""
Statistical analysis utilities for ToF-SIMS data analysis.

This module provides statistical functions for dose-response analysis, significance testing,
and effect size calculations specific to the alucone dose-response study.

Version: 0.1.0
"""

def exact_spearman_correlation(x, y, n_max=9):
    """
    Calculate exact Spearman correlation coefficient for small samples.
    
    For sample sizes n ≤ 9, computes exact p-values using the complete distribution
    of Spearman's rho rather than asymptotic approximations.
    
    Parameters
    ----------
    x : array_like
        First variable values (e.g., dose levels)
    y : array_like  
        Second variable values (e.g., chemical index values)
    n_max : int, default=9
        Maximum sample size for exact computation
        
    Returns
    -------
    rho : float
        Spearman correlation coefficient
    p_value : float
        Exact two-tailed p-value
    confidence_interval : tuple
        95% confidence interval for rho
        
    Raises
    ------
    ValueError
        If n > n_max or if x and y have different lengths
    """
    pass


def benjamini_hochberg_fdr(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg False Discovery Rate correction.
    
    Adjusts p-values for multiple testing using the BH-FDR procedure,
    controlling expected proportion of false discoveries among rejected hypotheses.
    
    Parameters
    ----------
    p_values : array_like
        Raw p-values from multiple hypothesis tests
    alpha : float, default=0.05
        Desired FDR level
        
    Returns
    -------
    adjusted_p : ndarray
        FDR-adjusted p-values
    rejected : ndarray of bool
        Boolean array indicating which hypotheses are rejected
    critical_values : ndarray
        BH critical values for each test
        
    Notes
    -----
    Implements the step-up procedure of Benjamini & Hochberg (1995).
    """
    pass


def paired_volcano_analysis(data_low, data_high, pattern_ids, effect_size_threshold=0.5):
    """
    Perform paired volcano plot analysis across spatial patterns.
    
    Computes paired t-tests between dose conditions using P1-P3 pattern replicates,
    with effect size filtering to identify practically significant changes.
    
    Parameters
    ----------
    data_low : array_like, shape (n_patterns, n_peaks)
        Intensity data for lower dose condition
    data_high : array_like, shape (n_patterns, n_peaks)  
        Intensity data for higher dose condition
    pattern_ids : array_like
        Pattern identifiers (P1, P2, P3) for pairing
    effect_size_threshold : float, default=0.5
        Minimum Cohen's d for practical significance
        
    Returns
    -------
    fold_changes : ndarray
        Log2 fold changes for each peak
    p_values : ndarray
        P-values from paired t-tests
    effect_sizes : ndarray
        Cohen's d effect sizes
    significant_peaks : ndarray of bool
        Boolean mask for peaks passing both p-value and effect size thresholds
        
    Notes
    -----
    Uses paired t-test to account for pattern-to-pattern variability.
    """
    pass


def calculate_rsd_percent(data, axis=0):
    """
    Calculate relative standard deviation as percentage.
    
    Computes RSD% = (standard deviation / mean) × 100 for quality control
    and technical replicate assessment.
    
    Parameters
    ----------
    data : array_like
        Input data array
    axis : int, default=0
        Axis along which to calculate RSD% (0=across samples, 1=across peaks)
        
    Returns
    -------
    rsd_percent : ndarray
        Relative standard deviation as percentage
    mean_values : ndarray
        Mean values used in RSD calculation
    cv_categories : ndarray of str
        Qualitative CV categories: 'excellent' (<5%), 'good' (5-15%), 
        'acceptable' (15-30%), 'poor' (>30%)
        
    Notes
    -----
    Handles zero means by returning NaN for corresponding RSD values.
    """
    pass


def dose_response_monotonicity(doses, responses, method='spearman'):
    """
    Test for monotonic dose-response relationships.
    
    Evaluates whether chemical indices show monotonic trends with increasing dose,
    using either Spearman correlation or Mann-Kendall trend test.
    
    Parameters
    ----------
    doses : array_like
        Dose levels (e.g., 500, 2000, 5000, 10000, 15000)
    responses : array_like, shape (n_samples, n_indices)
        Chemical index values for each sample
    method : str, default='spearman'
        Statistical method: 'spearman' or 'mann_kendall'
        
    Returns
    -------
    trend_statistics : ndarray
        Test statistics for monotonicity
    p_values : ndarray
        P-values for trend tests
    trend_directions : ndarray of str
        Trend direction: 'increasing', 'decreasing', or 'no_trend'
    slope_estimates : ndarray
        Slope estimates for trend magnitude
        
    Notes
    -----
    Accounts for tied dose levels and provides robust trend detection.
    """
    pass