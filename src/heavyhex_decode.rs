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


use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableUnGraph;
use hashbrown::HashMap;

use pyo3::prelude::*;

use crate::heavyhex::PyHeavyHexLattice;
use crate::graph::*;


#[pyclass]
pub struct PyAnalysis {
    pub num_bonds: usize,
    pub num_sites: usize,
    pub num_plaquettes: usize,
    pub bond_neighbors: (Vec<usize>, Vec<usize>),
    pub check_matrix: Vec<Vec<bool>>,
    pub decoding_bonds: Vec<bool>,
    pub gauge_map: Vec<(usize, usize, usize)>,
}

// #[pymethods]
// impl PyAnalysis {
//     #[new]
//     pub fn new(lattice: PyHeavyHexLattice) -> Self {

//     }
// }
