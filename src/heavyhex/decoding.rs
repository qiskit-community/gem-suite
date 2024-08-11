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

use bitvec::prelude::*;
use hashbrown::HashMap;
use fusion_blossom::mwpm_solver::{PrimalDualSolver, SolverSerial};
use fusion_blossom::util::{SolverInitializer, SyndromePattern};
use itertools::Itertools;

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
    counts: HashMap<String, usize>,
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
    let bond_qubits = lattice.decode_graph
        .edge_weights()
        .map(|ew| (ew.bit_index.unwrap(), ew.index))
        .collect::<std::collections::BTreeMap<_, _>>()
        .into_values()
        .collect_vec();
    let site_qubits = lattice.decode_graph
        .node_weights()
        .map(|nw| (nw.bit_index.unwrap(), nw.index))
        .collect::<std::collections::BTreeMap<_, _>>()
        .into_values()
        .collect_vec();
    let qubit_clbit_map = lattice.qubits()
        .into_iter()
        .enumerate()
        .map(|(ci, qi)| (qi.index, ci))
        .collect::<HashMap<_, _>>();
    let syndrome_qubits = lattice.plaquette_qubits_map
        .iter()
        .collect::<std::collections::BTreeMap<_, _>>()
        .into_values()
        .collect_vec();
    let correlation_indices = lattice.decode_graph
        .edge_weights()
        .map(|ew| {
            let s0 = site_qubits.iter().position(|qi| *qi == ew.neighbor0);
            let s1 = site_qubits.iter().position(|qi| *qi == ew.neighbor1);
            (ew.bit_index.unwrap(), s0.unwrap(), s1.unwrap())
        })
        .collect_vec();
    let snake_line = traverse_snake(&lattice.decode_graph);
    let n_bonds = bond_qubits.len();
    let n_sites = site_qubits.len();
    let n_syndrome = lattice.plaquette_qubits_map.len();
    let mut decoded_counts = HashMap::<String, usize>::with_capacity(counts.len());
    let mut syndrome_sum = vec![0_usize; n_syndrome];
    let mut bond_correlation_sum = vec![0_isize; n_bonds];
    let mut total_shots = 0_usize;
    for (key, count_num) in counts.iter() {
        solver.clear();
        let bitstring = key.chars().collect_vec();
        let get_bit = |qi: &QubitIndex| -> bool {
            let ci = if let Some(index) = qubit_clbit_map.get(qi) {
                index
            } else {
                panic!("Qubit {} doesn't exist in the measured bitstring.", qi);
            };
            match bitstring.get(bitstring.len() - 1 - *ci) {
                Some('0') => false,
                Some('1') => true,
                _ => panic!("Measured outcome of qubit {} is not a bit.", qi),
            }
        };
        let site_bits = site_qubits
            .iter()
            .map(|qi| get_bit(qi))
            .collect::<BitVec>();
        let bond_bits = bond_qubits
            .iter()
            .map(|qi| get_bit(qi))
            .collect::<BitVec>();
        // Compute bond correlation
        for idxs in correlation_indices.iter() {
            if bond_bits[idxs.0] ^ site_bits[idxs.1] ^ site_bits[idxs.2] {
                bond_correlation_sum[idxs.0] += *count_num as isize;
            } else {
                bond_correlation_sum[idxs.0] -= *count_num as isize;
            }
        }
        // Compute frustration of plaquette. i.e. syndromes.
        let syndrome = syndrome_qubits
            .iter()
            .map(|sub_qubits| {
                // Compute total number of '1' in the plaquette bonds.
                let sum = bond_qubits
                    .iter()
                    .zip(bond_bits.iter())
                    .fold(0_usize, |sum, (qi, bit)| {
                        if sub_qubits.contains(qi) & (bit == true) {
                            sum + 1
                        } else {
                            sum
                        }
                    });
                // Compute parity; even or odd. 
                // Odd parity must be an error and syndrome is set.
                sum % 2 == 1
            })
            .collect::<BitVec>();
        syndrome
            .iter()
            .enumerate()
            .for_each(|(i, s)| if s == true { syndrome_sum[i] += count_num });
        // Decode syndrome string (errors) with minimum weight perfect matching.
        // The subgraph bond index is the index of decoding bonds, 
        // i.e. index of fusion-blossom decoding graph edge.
        // If return is [0, 2] and the decoding bond indices are [2, 4, 6, 7, 8],
        // the actual bond index to flip is [2, 6].
        solver.solve(&build_syndrome_pattern(&syndrome));
        let mut bond_flips = bitvec![0; n_bonds];
        for ei in solver.subgraph().iter() {
            let bond_index = decoding_bits[ei];
            *bond_flips.get_mut(bond_index).unwrap() = true;
        }
        // Combination of bond outcomes + decoder output.
        // Value of 0 (1) indicates AFM (FM) bond.
        let loop_string = !(bond_bits ^ bond_flips);
        // Compute gauge string and decode site string as new dict key.
        let mut gauge = bitvec![0; n_sites];
        for gsb in snake_line.iter() {
            let gauge_bit = gauge[gsb.1] ^ loop_string[gsb.2];
            *gauge.get_mut(gsb.0).unwrap() = gauge_bit;
        }
        // Decode and convert into little endian.
        let site_key = (site_bits ^ gauge)
            .iter()
            .rev()
            .map(|bit| if *bit { '1' } else { '0' })
            .collect::<String>();
        *decoded_counts.entry_ref(&site_key).or_insert(0) += count_num;
        total_shots += count_num;
    }
    let w_ops = syndrome_sum
        .into_iter()
        .map(|v| 1.0 - 2.0 * v as f64 / total_shots as f64)
        .collect_vec();
    let zxz_ops = bond_correlation_sum
        .into_iter()
        .map(|v| v as f64 / total_shots as f64)
        .collect_vec();
    let (f, g) = decode_magnetization(&decoded_counts);

    (decoded_counts, w_ops, zxz_ops, f, g)
}


/// Generate check matrix H of this plaquette lattice.
/// Matrix is flattened and returned with dimension,
/// where the first dimension is the size of parity check (plaquettes)
/// and the second dimension is the size of variable (bond qubits).
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



#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{FALCON_CMAP, EAGLE_CMAP};

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
        // Validation
        //
        // Eagle coupling map
        // [q01]--<q04>--[q07]--<q10>--[q12]--<q15>--[q18]--<q21>--[q23]
        //   |                           |                           |
        // <q02>                       <q13>                       <q24>
        //   |                           |                           |
        // [q03]--<q05>--[q08]--<q11>--[q14]--<q16>--[q19]--<q22>--[q25]
        //
        // test string
        // 100111000111010001010
        //
        // mapping
        //       0      1      2      3      4      5      6      7      8      9      10
        // site: 0[q01] 0[q03] 0[q07] 0[q08] 1[q12] 1[q14] 0[q18] 1[q19] 0[q23] 1[q25]
        // bond: 1<q02> 1<q04> 0<q05> 1<q10> 0<q11> 1<q13> 0<q15> 0<q16> 1<q21> 1<q22> 0<q24> 
        //
        // bond string
        // 11010100110
        //
        // site string
        // 0000110101
        //
        // syndrome 0 (plaquette 0)
        // 110101 (0: even parity)
        // 
        // syndrome 1 (plaquette 1)
        // 100110 (1: odd parity)
        //
        // syndrome 
        // 01
        //
        // mwpm solution for decoding <q04> <q13> <q15> (select bond that flips only syndrome 1)
        // q15
        // 
        // loop (bond with q15 flipped, and invert bits)
        // 00101001001
        //
        // compute gauge
        // snake line
        //          : gauge[ 8] = 0
        // (6, 8, 8): gauge[ 6] = gauge[ 8] ^ loop[ 8] = 0 ^ 0 = 0
        // (4, 6, 6): gauge[ 4] = gauge[ 6] ^ loop[ 6] = 0 ^ 0 = 0
        // (2, 4, 3): gauge[ 2] = gauge[ 4] ^ loop[ 3] = 0 ^ 0 = 0
        // (0, 2, 1): gauge[ 0] = gauge[ 2] ^ loop[ 1] = 0 ^ 0 = 0
        // (1, 0, 0): gauge[ 1] = gauge[ 0] ^ loop[ 0] = 0 ^ 0 = 0
        // (3, 1, 2): gauge[ 3] = gauge[ 1] ^ loop[ 2] = 0 ^ 1 = 1
        // (5, 3, 4): gauge[ 5] = gauge[ 3] ^ loop[ 4] = 1 ^ 1 = 0
        // (7, 5, 7): gauge[ 7] = gauge[ 5] ^ loop[ 7] = 0 ^ 1 = 1
        // (9, 7, 9): gauge[ 9] = gauge[ 7] ^ loop[ 9] = 1 ^ 0 = 1
        // -------------------------------------------------------
        // gauge
        // 0001000101
        //
        // site ^ gauge
        // 0001110000
        //
        // to little endian
        // 0000111000
        //
        // bond correlation
        // bond[ 0] ^ site[ 0] ^ site[ 1] = 1 ^ 0 ^ 0 = 1
        // bond[ 1] ^ site[ 0] ^ site[ 2] = 1 ^ 0 ^ 0 = 1
        // bond[ 2] ^ site[ 1] ^ site[ 3] = 0 ^ 0 ^ 0 = 0
        // bond[ 3] ^ site[ 2] ^ site[ 4] = 1 ^ 0 ^ 1 = 0
        // bond[ 4] ^ site[ 3] ^ site[ 5] = 0 ^ 0 ^ 1 = 1
        // bond[ 5] ^ site[ 4] ^ site[ 5] = 1 ^ 1 ^ 1 = 1
        // bond[ 6] ^ site[ 4] ^ site[ 6] = 0 ^ 1 ^ 0 = 1
        // bond[ 7] ^ site[ 5] ^ site[ 7] = 0 ^ 1 ^ 1 = 0
        // bond[ 8] ^ site[ 6] ^ site[ 8] = 1 ^ 0 ^ 0 = 1
        // bond[ 9] ^ site[ 7] ^ site[ 9] = 1 ^ 1 ^ 1 = 1
        // bond[10] ^ site[ 8] ^ site[ 9] = 0 ^ 0 ^ 1 = 1
        //
        let coupling_map = FALCON_CMAP.lock().unwrap().to_owned();
        let lattice = PyHeavyHexLattice::new(coupling_map);
        let outcomes = HashMap::<String, usize>::from([(format!("100111000111010001010"), 123)]);
        let tmp = decode_outcomes_fb(&lattice, outcomes);
        let count = tmp.0;
        let w_ops = tmp.1;
        let zxz_ops = tmp.2;
        let f = tmp.3.0;
        let g = tmp.4.0;
        assert_eq!(
            count,
            HashMap::<String, usize>::from([(format!("0000111000"), 123_usize)])
        );
        assert_eq!(
            w_ops,
            vec![1.0, -1.0]
        );
        assert_eq!(
            zxz_ops,
            vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(f, 0.0);
        assert_eq!(g, 0.0);
    }
}
