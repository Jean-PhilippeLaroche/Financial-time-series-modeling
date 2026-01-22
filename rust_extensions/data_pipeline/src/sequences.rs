use pyo3::prelude::*;
use numpy::{PyArray3, PyArray1, PyReadonlyArray2, PyReadonlyArray1};
use ndarray::{Array1, Array3, s};

/// Create sequences with forward returns for time series prediction
/// 
/// Args:
///     df_scaled: 2D array of scaled features (rows=timesteps, cols=features)
///     feature_names: List of feature column names (to identify 'close')
///     raw_close: 1D array of unscaled close prices
///     window_size: Lookback period
///     forward_bars: Prediction horizon
/// 
/// Returns:
///     Tuple of (X, y) where:
///         X: 3D array (samples, window_size, features)
///         y: 1D array of forward returns
#[pyfunction]
pub fn create_sequences_with_forward_returns<'py>(
    py: Python<'py>,
    df_scaled: PyReadonlyArray2<f64>,
    feature_names: Vec<String>,
    raw_close: PyReadonlyArray1<f64>,
    window_size: usize,
    forward_bars: usize,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray1<f64>>)> {
    
    let df_array = df_scaled.as_array();
    let close_array = raw_close.as_array();
    
    let n_rows = df_array.nrows();
    let n_features = df_array.ncols();
    
    // Find the index of 'close' column if it exists
    let close_idx = feature_names.iter().position(|name| name == "close");
    
    // Calculate forward returns on raw close prices
    let mut forward_returns = Array1::<f64>::zeros(n_rows);
    for i in 0..(n_rows - forward_bars) {
        let current_price = close_array[i];
        let future_price = close_array[i + forward_bars];
        forward_returns[i] = (future_price / current_price) - 1.0;
    }
    
    // Create a modified feature array
    // If 'close' exists, replace it with 'close_return'
    let mut features_array = df_array.to_owned();
    
    if let Some(idx) = close_idx {
        // Calculate close returns (pct_change) on raw close prices
        let mut close_returns = Array1::<f64>::zeros(n_rows);
        
        // First value is NaN conceptually, set to 0.0
        close_returns[0] = 0.0;
        
        for i in 1..n_rows {
            let prev_price = close_array[i - 1];
            let curr_price = close_array[i];
            if prev_price != 0.0 {
                close_returns[i] = (curr_price / prev_price) - 1.0;
            } else {
                close_returns[i] = 0.0;
            }
        }
        
        // Replace the 'close' column with close_returns
        // Note: close_returns are already percentage changes, no scaling needed
        for i in 0..n_rows {
            features_array[[i, idx]] = close_returns[i];
        }
    }
    
    // Calculate max valid index
    // Need enough data for the window + forward prediction
    let max_idx = n_rows.saturating_sub(window_size).saturating_sub(forward_bars - 1);
    
    if max_idx == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Not enough data: need at least {} rows, got {}", 
                    window_size + forward_bars, n_rows)
        ));
    }
    
    // Pre-allocate output arrays
    let mut x_data = Array3::<f64>::zeros((max_idx, window_size, n_features));
    let mut y_data = Array1::<f64>::zeros(max_idx);
    
    // Create sequences
    for i in 0..max_idx {
        // Extract window of features
        let window = features_array.slice(s![i..i + window_size, ..]);
        x_data.slice_mut(s![i, .., ..]).assign(&window);
        
        // Extract target (forward return at end of window)
        y_data[i] = forward_returns[i + window_size - 1];
    }
    
    // Convert to numpy arrays
    let x_py = PyArray3::from_owned_array_bound(py, x_data);
    let y_py = PyArray1::from_owned_array_bound(py, y_data);
    
    Ok((x_py, y_py))
}