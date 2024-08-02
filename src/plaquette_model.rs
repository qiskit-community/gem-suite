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


#[derive(Debug, PartialEq, Clone, Copy)]
pub enum QubitRole {
    Site,
    Bond,
}


#[derive(Debug, PartialEq, Clone, Copy)]
pub enum OpGroup {
    A,
    B,
}


#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SchedulingGroup {
    E1,
    E2,
    E3,
    E4,
    E5,
    E6,
}


#[derive(Debug, PartialEq, Clone, Copy)]
pub struct QubitNode {
    pub index: usize,
    pub role: Option<QubitRole>,
    pub group: Option<OpGroup>,
    pub coordinate: Option<(usize, usize)>,
}

impl QubitNode {
    pub fn to_dot(&self) -> String {
        let mut options = Vec::<String>::new();
        match self.role {
            Some(QubitRole::Site) => {
                options.push(format!("fillcolor=darkgrey"));
                options.push(format!("style=filled"));
            },
            _ => {
                options.push(format!("fillcolor=lightgrey"));
                options.push(format!("style=solid"));
            },
        }
        match self.group {
            Some(OpGroup::A) => {
                options.push(format!("label=\"Q{} (A)\"", self.index));
            },
            Some(OpGroup::B) => {
                options.push(format!("label=\"Q{} (B)\"", self.index));
            },
            None => {
                options.push(format!("label=Q{}", self.index));
            },
        }
        match self.coordinate {
            Some(xy) => {
                options.push(format!("pos=\"{},-{}\"", xy.0, xy.1));
                options.push(format!("pin=True"))
            },
            None => {},
        }
        format!("{} [{}, shape=circle];", self.index, options.join(", "))
    }
}


#[derive(Debug, PartialEq, Clone, Copy)]
pub struct CouplingEdge {
    pub q0: usize,
    pub q1: usize,
    pub group: Option<SchedulingGroup>,
}

impl CouplingEdge {
    pub fn to_dot(&self) -> String {
        let mut options = Vec::<String>::new();
        match self.group {
            Some(SchedulingGroup::E1) => {
                options.push(format!("color=mediumseagreen"));
            },
            Some(SchedulingGroup::E2) => {
                options.push(format!("color=thistle"));
            },
            Some(SchedulingGroup::E3) => {
                options.push(format!("color=lightsalmon"));
            },
            Some(SchedulingGroup::E4) => {
                options.push(format!("color=khaki"));
            },
            Some(SchedulingGroup::E5) => {
                options.push(format!("color=dodgerblue"));
            },
            Some(SchedulingGroup::E6) => {
                options.push(format!("color=mediumvioletred"));
            },
            None => {
                options.push(format!("color=black"));                
            },
        }
        format!("{} -- {} [{}];", self.q0, self.q1, options.join(", "))
    }
}
