# This code is part of Qiskit.
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Decoder debugger with hard-coded Falcom coupling map."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def process_counts_debug(
    counts: dict[str, int]
) -> tuple[
    dict[str, int], list[float], list[float], tuple[float, float], tuple[float, float]
]:
    """Generate reference decoded outcomes.

    Cheat sheet:

    # Coupling map

      [q01]--<q04>--[q07]--<q10>--[q12]--<q15>--[q18]--<q21>--[q23]
        |                           |                           |
      <q02>                       <q13>                       <q24>
        |                           |                           |
      [q03]--<q05>--[q08]--<q11>--[q14]--<q16>--[q19]--<q22>--[q25]

    # Bit mapping

            0     1     2     3     4     5     6     7     8     9     10
      site: [q01] [q03] [q07] [q08] [q12] [q14] [q18] [q19] [q23] [q25]
      bond: <q02> <q04> <q05> <q10> <q11> <q13> <q15> <q16> <q21> <q22> <q24>

    # Register mapping (little endian)

      [q01]: 0
      <q02>: 1
      [q03]: 2
      <q04>: 3
      <q05>: 4
      [q07]: 5
      [q08]: 6
      <q10>: 7
      <q11>: 8
      [q12]: 9
      <q13>: 10
      [q14]: 11
      <q15>: 12
      <q16>: 13
      [q18]: 14
      [q19]: 15
      <q21>: 16
      <q22>: 17
      [q23]: 18
      <q24>: 19
      [q25]: 20

    # Gauge calculation (snake line)

                : gauge[ 8] = 0
      (6, 8, 8): gauge[ 6] = gauge[ 8] ^ loop[ 8]
      (4, 6, 6): gauge[ 4] = gauge[ 6] ^ loop[ 6]
      (2, 4, 3): gauge[ 2] = gauge[ 4] ^ loop[ 3]
      (0, 2, 1): gauge[ 0] = gauge[ 2] ^ loop[ 1]
      (1, 0, 0): gauge[ 1] = gauge[ 0] ^ loop[ 0]
      (3, 1, 2): gauge[ 3] = gauge[ 1] ^ loop[ 2]
      (5, 3, 4): gauge[ 5] = gauge[ 3] ^ loop[ 4]
      (7, 5, 7): gauge[ 7] = gauge[ 5] ^ loop[ 7]
      (9, 7, 9): gauge[ 9] = gauge[ 7] ^ loop[ 9]
    """
    decoded_counts = defaultdict(int)
    syndrome_sum = [0, 0]
    bond_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    num_shots = sum(counts.values())
    for bits, count_num in counts.items():
        be_bits = [b == "1" for b in bits[::-1]]
        site_bits = np.array(
            [
                be_bits[0],
                be_bits[2],
                be_bits[5],
                be_bits[6],
                be_bits[9],
                be_bits[11],
                be_bits[14],
                be_bits[15],
                be_bits[18],
                be_bits[20],
            ],
            dtype=bool,
        )
        bond_bits = np.array(
            [
                be_bits[1],
                be_bits[3],
                be_bits[4],
                be_bits[7],
                be_bits[8],
                be_bits[10],
                be_bits[12],
                be_bits[13],
                be_bits[16],
                be_bits[17],
                be_bits[19],
            ],
            dtype=bool,
        )
        # List bonds inside each plaquette
        syndrome0 = bond_bits[:6]
        syndrome1 = bond_bits[5:]
        # Check parity of each plaquette
        syndromes = (sum(syndrome0) % 2, sum(syndrome1) % 2)
        # Add up frustration
        if syndromes[0] == 1:
            syndrome_sum[0] += count_num
        if syndromes[1] == 1:
            syndrome_sum[1] += count_num
        # Add up bond correlation
        bond_sum[0] += (
            count_num if bond_bits[0] ^ site_bits[0] ^ site_bits[1] else -count_num
        )
        bond_sum[1] += (
            count_num if bond_bits[1] ^ site_bits[0] ^ site_bits[2] else -count_num
        )
        bond_sum[2] += (
            count_num if bond_bits[2] ^ site_bits[1] ^ site_bits[3] else -count_num
        )
        bond_sum[3] += (
            count_num if bond_bits[3] ^ site_bits[2] ^ site_bits[4] else -count_num
        )
        bond_sum[4] += (
            count_num if bond_bits[4] ^ site_bits[3] ^ site_bits[5] else -count_num
        )
        bond_sum[5] += (
            count_num if bond_bits[5] ^ site_bits[4] ^ site_bits[5] else -count_num
        )
        bond_sum[6] += (
            count_num if bond_bits[6] ^ site_bits[4] ^ site_bits[6] else -count_num
        )
        bond_sum[7] += (
            count_num if bond_bits[7] ^ site_bits[5] ^ site_bits[7] else -count_num
        )
        bond_sum[8] += (
            count_num if bond_bits[8] ^ site_bits[6] ^ site_bits[8] else -count_num
        )
        bond_sum[9] += (
            count_num if bond_bits[9] ^ site_bits[7] ^ site_bits[9] else -count_num
        )
        bond_sum[10] += (
            count_num if bond_bits[10] ^ site_bits[8] ^ site_bits[9] else -count_num
        )
        # Hardcoded decoder
        # There are three check variables (<q04>, <q13>, <q15>) to solve.
        decode_in_out = {
            (0, 0): [False, False, False],
            (0, 1): [False, False, True],
            (1, 0): [True, False, False],
            (1, 1): [False, True, False],
        }
        out = decode_in_out[syndromes]
        # Loop := flipped bond bits ^ match string and invert
        loop_string = bond_bits.copy()
        for flag, bit in zip(out, [1, 5, 6]):
            # Flip when bond is highlighted
            loop_string[bit] ^= flag
        loop_string = np.invert(loop_string)
        # Compute gauge
        gauge = np.array([False] * site_bits.size, dtype=bool)
        gauge[6] = gauge[8] ^ loop_string[8]
        gauge[4] = gauge[6] ^ loop_string[6]
        gauge[2] = gauge[4] ^ loop_string[3]
        gauge[0] = gauge[2] ^ loop_string[1]
        gauge[1] = gauge[0] ^ loop_string[0]
        gauge[3] = gauge[1] ^ loop_string[2]
        gauge[5] = gauge[3] ^ loop_string[4]
        gauge[7] = gauge[5] ^ loop_string[7]
        gauge[9] = gauge[7] ^ loop_string[9]
        # Decoded site
        decoded_bits = site_bits ^ gauge
        decoded_bitstring = ["1" if b else "0" for b in decoded_bits[::-1]]
        decoded_counts["".join(decoded_bitstring)] += count_num
    w_ops = [1 - 2 * s / num_shots for s in syndrome_sum]
    zxz_ops = [s / num_shots for s in bond_sum]
    f, g = calculate_f_and_g(decoded_counts)
    return dict(decoded_counts), w_ops, zxz_ops, f, g


def calculate_f_and_g(dist):
    """Compute f and g values from count dictionary."""
    m1 = 0
    m2 = 0
    m4 = 0
    m8 = 0

    num_data = 10
    num_shots = sum(dist.values())
    dist = {key: val / num_shots for key, val in dist.items()}

    ### calculate moments
    for creg_data, count in dist.items():
        creg_data_array = np.array([int(val) for val in creg_data])
        mj_array = (-1) ** creg_data_array
        m1 += count * sum(mj_array)
        m2 += count * sum(mj_array) ** 2
        m4 += count * sum(mj_array) ** 4
        m8 += count * sum(mj_array) ** 8

    out_f = 1 / num_data * (m2 - m1**2)
    out_g = 1 / num_data**3 * (m4 - m2**2)

    std_f = (
        m4 / num_shots
        - (m2 - m1**2) ** 2 * (num_shots - 3) / num_shots / (num_shots - 1)
    ) ** 0.5
    std_g = (
        m8 / num_shots - std_f**4 * (num_shots - 3) / num_shots / (num_shots - 1)
    ) ** 0.5

    std_f /= num_data
    std_g /= num_data**3

    return (out_f, std_f), (out_g, std_g)
