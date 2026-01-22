use pyo3::prelude::*;

mod sequences;

use sequences::create_sequences_with_forward_returns;

#[pymodule]
fn data_pipeline(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_sequences_with_forward_returns, m)?)?;
    Ok(())
}