// This code is part of Qiskit.
//
// (C) Copyright IBM 2024.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;

pub mod heavyhex;
mod graph;
mod mock;


#[pymodule]
fn gem_core(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<heavyhex::PyHeavyHexLattice>()?;
    m.add_class::<heavyhex::PyQubit>()?;
    m.add_class::<heavyhex::PyPlaquette>()?;
    m.add_class::<heavyhex::PyScheduledGate>()?;
    Ok(())
}
