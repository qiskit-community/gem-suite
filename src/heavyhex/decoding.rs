// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

use bitvec::prelude::*;
use hashbrown::HashMap;
use fusion_blossom::mwpm_solver::{PrimalDualSolver, SolverSerial};
use fusion_blossom::util::{SolverInitializer, SyndromePattern};
use itertools::Itertools;

use pyo3::prelude::*;

use crate::graph::*;
use crate::utils::decode_magnetization;
use super::graph_builder::traverse_snake;
use super::PyHeavyHexLattice;


/// Build minimum weight perfect matching solver from the decode graph.
/// Variables are bond qubits with having is_decode_variable flag set.
/// Syndrome should have the same size with the plaquettes.
pub(super) fn build_mwpm_solver(
    lattice: &PyHeavyHexLattice,
) -> SolverSerial {
    // We have one boundary (virtual) node, where unpaired syndromes are paired with.
    // We don't need multiple virtual nodes because boundary condition is uniform 
    // and coupling graph (plaquette lattice) is a single SCC.
    let vertex_num = lattice.plaquette_qubits_map.len() + 1;
    let mut reverse_pq_map = HashMap::<QubitIndex, Vec<usize>>::new();
    lattice.plaquette_graph
        .node_weights()
        .for_each(|pw| {
            for qi in &lattice.plaquette_qubits_map[&pw.index] {
                if let Some(qis) = reverse_pq_map.get_mut(qi) {
                    qis.push(pw.syndrome_index);
                } else {
                    reverse_pq_map.insert(*qi, vec![pw.syndrome_index]);
                }
            }
        });
    let decoding_edges = lattice.decode_graph
        .edge_weights()
        .filter_map(|ew| {
            if let Some(index) = ew.variable_index {
                let syndromes = &reverse_pq_map[&ew.index];
                let edge = match syndromes.len() {
                    // Boundary bond. Connect with the virtual vertex.
                    1 => (syndromes[0], vertex_num - 1, 2),
                    // Make real edge.
                    2 => (syndromes[0], syndromes[1], 2),
                    _ => panic!("Bond {} belongs to more than two or zero syndromes.", ew.index),
                };
                Some((index, edge))
            } else {
                None
            }
        })
        // Sort by bond (bit) index
        .collect::<std::collections::BTreeMap<usize, (usize, usize, isize)>>()
        .into_values()
        .collect_vec();
    let ini = SolverInitializer::new(vertex_num, decoding_edges, vec![vertex_num - 1]);
    SolverSerial::new(&ini)
}


/// Build syndrome pattern to feed MWPM solver.
pub(super) fn build_syndrome_pattern(
    syndrome: &BitVec
) -> SyndromePattern {
    let defect_vertices = syndrome
        .iter()
        .enumerate()
        .filter_map(|(i, b)| if b == true { Some(i) } else { None })
        .collect();
    SyndromePattern::new(defect_vertices, vec![])
}


/// Decode raw circuit outcome with plaquette lattice information
/// to compute quantities associated with prepared state magnetization 
/// and other set of quantities associated with device error.
/// This function is implemented upon the fusion-blossom decoder.
/// 
/// # Arguments
/// * `lattice`: Plaquette lattice to provide lattice topology.
/// * `counts`: Count dictionary keyed on measured bitstring in little endian format.
/// 
/// # Returns
/// A tuple of decoded count dictionary, plaquette and ZXZ bond observables,
/// and f and g values associated with decoded magnetization.
pub(super) fn decode_outcomes_fb(
    lattice: &PyHeavyHexLattice,
    counts: &HashMap<String, usize>,
) -> (HashMap<String, usize>, Vec<f64>, Vec<f64>, (f64, f64), (f64, f64)) {
    let mut solver = build_mwpm_solver(&lattice);
    let decoding_bits = lattice.decode_graph
        .edge_weights()
        .filter_map(|ew| {
            if let Some(index) = ew.variable_index {
                Some((index, ew.bit_index.unwrap()))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();
    let snake_line = traverse_snake(&lattice.decode_graph);
    let n_bonds = lattice.bit_specifier.n_bonds;
    let n_syndrome = lattice.plaquette_qubits_map.len();
    let mut decoded_counts = HashMap::<String, usize>::with_capacity(counts.len());
    let mut syndrome_sum = vec![0_usize; n_syndrome];
    let mut bond_sum = vec![0_isize; n_bonds];
    let mut total_shots = 0_usize;
    for (meas_string, count_num) in counts.iter() {
        solver.clear();
        let (site_bits, bond_bits, syndromes) = decode_preprocess(
            lattice, 
            meas_string, 
            count_num, 
            &mut bond_sum, 
            &mut syndrome_sum,
        );
        // Decode syndrome string (errors) with minimum weight perfect matching.
        // The subgraph bond index is the index of decoding bonds, 
        // i.e. index of fusion-blossom decoding graph edge.
        // If return is [0, 2] and the decoding bond indices are [2, 4, 6, 7, 8],
        // the actual bond index to flip is [2, 6].
        solver.solve(&build_syndrome_pattern(&syndromes));
        let mut match_string = bitvec![0; n_bonds];
        for ei in solver.subgraph().iter() {
            match_string.set(decoding_bits[ei], true);
        }
        let site_key = decode_postprocess(
            match_string, 
            bond_bits, 
            site_bits, 
            &snake_line
        );
        *decoded_counts.entry_ref(&site_key).or_insert(0) += count_num;
        total_shots += count_num;
    }
    let w_ops = syndrome_sum
        .into_iter()
        .map(|v| 1.0 - 2.0 * v as f64 / total_shots as f64)
        .collect_vec();
    let zxz_ops = bond_sum
        .into_iter()
        .map(|v| v as f64 / total_shots as f64)
        .collect_vec();
    let (f, g) = decode_magnetization(&decoded_counts);

    (decoded_counts, w_ops, zxz_ops, f, g)
}


/// Generate check matrix of this plaquette lattice.
pub(crate) fn check_matrix_csc(
    lattice: &PyHeavyHexLattice,
) -> ((Vec<usize>, Vec<usize>), (usize, usize)) {
    let num_syndrome = lattice.plaquette_graph.node_count();
    let mut num_variables = 0_usize;
    let row_col: (Vec<usize>, Vec<usize>) = lattice.decode_graph
        .edge_weights()
        .filter_map(|ew| {
            if let Some(col_index) = ew.variable_index {
                num_variables += 1;
                let col = lattice.plaquette_graph
                    .node_weights()
                    .filter_map(|pw| {
                        if lattice.plaquette_qubits_map[&pw.index].contains(&ew.index) {
                            Some((pw.syndrome_index, col_index))
                        } else {
                            None
                        }
                    })
                    .collect_vec();
                Some(col)
            } else {
                None
            }
        })
        .flatten()
        .unzip();

    (row_col, (num_syndrome, num_variables))
}


/// Decode raw circuit outcome with plaquette lattice information
/// to compute quantities associated with prepared state magnetization 
/// and other set of quantities associated with device error.
/// This function is implemented upon the pymatching decoder in the batch mode.
/// 
/// # Arguments
/// * `solver`: Call to pymatching decoder as a Python function.
/// * `lattice`: Plaquette lattice to provide lattice topology.
/// * `counts`: Count dictionary keyed on measured bitstring in little endian format.
/// 
/// # Returns
/// A tuple of decoded count dictionary, plaquette and ZXZ bond observables,
/// and f and g values associated with decoded magnetization.
pub(super) fn decode_outcomes_pm(
    py: Python,
    solver: PyObject,
    lattice: &PyHeavyHexLattice,
    counts: &HashMap<String, usize>,
) -> (HashMap<String, usize>, Vec<f64>, Vec<f64>, (f64, f64), (f64, f64)) {
    let decoding_bits = lattice.decode_graph
        .edge_weights()
        .filter_map(|ew| {
            if let Some(index) = ew.variable_index {
                Some((index, ew.bit_index.unwrap()))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();
    let snake_line = traverse_snake(&lattice.decode_graph);
    let n_bonds = lattice.bit_specifier.n_bonds;
    let n_syndrome = lattice.plaquette_qubits_map.len();
    let n_variable = decoding_bits.len();
    let mut decoded_counts = HashMap::<String, usize>::with_capacity(counts.len());
    let mut syndrome_sum = vec![0_usize; n_syndrome];
    let mut bond_sum = vec![0_isize; n_bonds];
    let mut total_shots = 0_usize;
    let mut stack = Vec::<((BitVec, BitVec, BitVec), usize)>::with_capacity(counts.len());
    for (meas_string, count_num) in counts.iter() {
        let data = decode_preprocess(
            lattice, 
            meas_string, 
            count_num, 
            &mut bond_sum, 
            &mut syndrome_sum,
        );
        stack.push((data, *count_num));
        total_shots += count_num;
    }
    // TODO better error handling
    let shots = stack
        .iter()
        .flat_map(|s| s.0.2.iter().by_vals().collect::<Vec<bool>>())
        .collect_vec();
    let decoded_shots = solver
        .call1(py, (shots, (stack.len(), n_syndrome)))
        .unwrap()
        .extract::<Vec<bool>>(py)
        .unwrap();
    for (i, data) in stack.into_iter().enumerate() {
        let mut match_string = bitvec![0; n_bonds];
        for (vi, bi) in decoding_bits.iter() {
            let idx = i * n_variable + vi;
            match_string.set(*bi, decoded_shots[idx]);
        }
        let site_key = decode_postprocess(
            match_string, 
            data.0.1, 
            data.0.0, 
            &snake_line
        );
        *decoded_counts.entry_ref(&site_key).or_insert(0) += data.1;
    }
    let w_ops = syndrome_sum
        .into_iter()
        .map(|v| 1.0 - 2.0 * v as f64 / total_shots as f64)
        .collect_vec();
    let zxz_ops = bond_sum
        .into_iter()
        .map(|v| v as f64 / total_shots as f64)
        .collect_vec();
    let (f, g) = decode_magnetization(&decoded_counts);

    (decoded_counts, w_ops, zxz_ops, f, g)    
}


fn decode_preprocess(
    lattice: &PyHeavyHexLattice,
    meas_string: &String,
    count_num: &usize,
    bond_sum: &mut Vec<isize>,
    syndrome_sum: &mut Vec<usize>,
) -> (BitVec, BitVec, BitVec) {
    let meas_string = meas_string.chars().collect_vec();
    let site_bits = lattice.bit_specifier.to_site_string(&meas_string);
    let bond_bits = lattice.bit_specifier.to_bond_string(&meas_string);
    // Add up frustrated syndromes
    let syndrome = lattice.bit_specifier.calculate_syndrome(&bond_bits);
    syndrome
        .iter()
        .enumerate()
        .for_each(|(i, s)| if s == true { syndrome_sum[i] += *count_num });
    // Add up correlated bits
    lattice
        .bit_specifier
        .correlated_bits
        .iter()
        .for_each(|idxs| {
            if bond_bits[idxs.0] ^ site_bits[idxs.1] ^ site_bits[idxs.2] {
                bond_sum[idxs.0] += *count_num as isize;
            } else {
                bond_sum[idxs.0] -= *count_num as isize;
            }
        });
    (site_bits, bond_bits, syndrome)
}


fn decode_postprocess(
    match_string: BitVec,
    bond_bits: BitVec,
    site_bits: BitVec,
    snake_line: &Vec<(usize, usize, usize)>,
) -> String {
    // Combination of bond outcomes + decoder output.
    // Value of 0 (1) indicates AFM (FM) bond.
    let loop_string = !(bond_bits ^ match_string);
    // Compute gauge string and decode site string as new dict key.
    let n_sites = site_bits.len();
    let mut gauge = bitvec![0; n_sites];
    for gsb in snake_line.iter() {
        let gauge_bit = gauge[gsb.1] ^ loop_string[gsb.2];
        gauge.set(gsb.0, gauge_bit);
    }
    // Decode and convert into little endian.
    (site_bits ^ gauge)
        .iter()
        .rev()
        .map(|bit| if *bit { '1' } else { '0' })
        .collect::<String>()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{FALCON_CMAP, EAGLE_CMAP};
    use approx::assert_relative_eq;

    #[test]
    fn test_check_matrix() {
        let coupling_map = FALCON_CMAP.lock().unwrap().to_owned();
        let lattice = PyHeavyHexLattice::new(coupling_map);
        let tmp = check_matrix_csc(&lattice);
        let rows = tmp.0.0;
        let cols = tmp.0.1;
        let shape = tmp.1;

        assert_eq!(rows, vec![0, 0, 1, 1]);
        assert_eq!(cols, vec![0, 1, 1, 2]);
        assert_eq!(shape, (2, 3));
    }

    #[test]
    fn test_mwpm_18_syndromes() {
        // Test against pymatching decoder initialized by a check matrix H.
        let coupling_map = EAGLE_CMAP.lock().unwrap().to_owned();
        let lattice = PyHeavyHexLattice::new(coupling_map);
        let mut solver = build_mwpm_solver(&lattice);
        // # test 1
        let syndrome = build_syndrome_pattern(&bitvec![0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0]);
        solver.clear();
        solver.solve(&syndrome);
        assert_eq!(solver.subgraph(), vec![10, 11, 28, 35, 40]);
        // # test 2
        let syndrome = build_syndrome_pattern(&bitvec![1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]);
        solver.clear();
        solver.solve(&syndrome);
        assert_eq!(solver.subgraph(), vec![1, 5, 24, 29, 36, 43]);
        // # test 3
        let syndrome = build_syndrome_pattern(&bitvec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
        solver.clear();
        solver.solve(&syndrome);
        assert_eq!(solver.subgraph(), vec![24, 31]);
    }


    #[test]
    fn test_decode_outcome() {
        // See tests.utils.process_counts_debug script for reference answer.
        let coupling_map = FALCON_CMAP.lock().unwrap().to_owned();
        let lattice = PyHeavyHexLattice::new(coupling_map);
        let outcomes = HashMap::<String, usize>::from([
            (format!("111111100001100101001"), 100),
            (format!("000101001111111110100"), 100),
            (format!("101100000110111101000"), 100),
            (format!("010001110100011101011"), 100),
            (format!("011111010001111110110"), 100),
        ]);
        let tmp = decode_outcomes_fb(&lattice, &outcomes);
        let count = tmp.0;
        let w_ops = tmp.1;
        let zxz_ops = tmp.2;
        let f = tmp.3;
        let g = tmp.4;
        assert_eq!(
            count,
            HashMap::<String, usize>::from([
                (format!("1111111000"), 100),
                (format!("1001111111"), 100),
                (format!("0111101110"), 100),
                (format!("0000001110"), 100),
                (format!("0110001010"), 100),
            ])
        );
        let ref_w_ops = vec![0.6, 0.2];
        for (v, ref_v) in w_ops.iter().zip(ref_w_ops.iter()){
            assert_relative_eq!(v, ref_v);
        }
        let ref_zxz_ops = vec![-0.2, 0.6, 0.6, -0.2, 0.2, 0.6, -0.2, 0.2, 0.2, -0.2, -0.2];
        for (v, ref_v) in zxz_ops.iter().zip(ref_zxz_ops.iter()){
            assert_relative_eq!(v, ref_v);
        }
        // Std part uses higher order moment.
        // Computation of power with larger exponent might cause non-negiligible numerical error.
        // This test has higher tolerance to accept the numerical error.
        assert_relative_eq!(f.0, 1.5040000000000004, max_relative=0.001);
        assert_relative_eq!(f.1, 0.04361780095970634, max_relative=0.3);
        assert_relative_eq!(g.0, 0.1062399999999999, max_relative=0.001);
        assert_relative_eq!(g.1, 0.019372556978285625, max_relative=0.3);
    }
}
