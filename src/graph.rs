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

pub type PlaquetteIndex = usize;
pub type QubitIndex = usize;
pub type BitIndex = usize;
pub type SyndromeIndex = usize;

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

pub trait WriteDot {
    fn to_dot(&self) -> String;
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct QubitNode {
    pub index: QubitIndex,
    pub role: Option<QubitRole>,
    pub group: Option<OpGroup>,
    pub coordinate: Option<(isize, isize)>,
}

impl WriteDot for QubitNode {
    fn to_dot(&self) -> String {
        let mut options = Vec::<String>::new();
        match self.role {
            Some(QubitRole::Site) => {
                options.push("fillcolor=darkgrey".to_string());
                options.push("style=filled".to_string());
            }
            _ => {
                options.push("fillcolor=lightgrey".to_string());
                options.push("style=solid".to_string());
            }
        }
        match self.group {
            Some(OpGroup::A) => {
                options.push(format!("label=\"Q{} (A)\"", self.index));
            }
            Some(OpGroup::B) => {
                options.push(format!("label=\"Q{} (B)\"", self.index));
            }
            None => {
                options.push(format!("label=Q{}", self.index));
            }
        }
        if let Some(xy) = self.coordinate {
            options.push(format!("pos=\"{},-{}\"", xy.0, xy.1));
            options.push("pin=True".to_string());
        }
        format!("{} [{}, shape=\"circle\"];", self.index, options.join(", "))
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct QubitEdge {
    pub neighbor0: QubitIndex,
    pub neighbor1: QubitIndex,
    pub group: Option<SchedulingGroup>,
}

impl WriteDot for QubitEdge {
    fn to_dot(&self) -> String {
        let mut options = Vec::<String>::new();
        match self.group {
            Some(SchedulingGroup::E1) => {
                options.push("color=mediumseagreen".to_string());
            }
            Some(SchedulingGroup::E2) => {
                options.push("color=thistle".to_string());
            }
            Some(SchedulingGroup::E3) => {
                options.push("color=lightsalmon".to_string());
            }
            Some(SchedulingGroup::E4) => {
                options.push("color=khaki".to_string());
            }
            Some(SchedulingGroup::E5) => {
                options.push("color=dodgerblue".to_string());
            }
            Some(SchedulingGroup::E6) => {
                options.push("color=mediumvioletred".to_string());
            }
            None => {
                options.push("color=black".to_string());
            }
        }
        format!(
            "{} -- {} [{}];",
            self.neighbor0,
            self.neighbor1,
            options.join(", ")
        )
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct PlaquetteNode {
    pub index: PlaquetteIndex,
    pub syndrome_index: SyndromeIndex,
}

impl WriteDot for PlaquetteNode {
    fn to_dot(&self) -> String {
        format!(
            "{} [label=P{}, shape=hexagon, width=0.81];",
            self.index, self.index
        )
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct PlaquetteEdge {
    pub neighbor0: PlaquetteIndex,
    pub neighbor1: PlaquetteIndex,
}

impl WriteDot for PlaquetteEdge {
    fn to_dot(&self) -> String {
        format!("{} -- {};", self.neighbor0, self.neighbor1)
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct DecodeNode {
    pub index: QubitIndex,
    pub bit_index: Option<BitIndex>,
    pub coordinate: (isize, isize),
}

impl WriteDot for DecodeNode {
    fn to_dot(&self) -> String {
        let label = if let Some(bi) = self.bit_index {
            format!("Q{}:S[{}]", self.index, bi)
        } else {
            format!("Q{}", self.index)
        };
        format!(
            "{} [pos=\"{},-{}\", pin=True, fillcolor=darkgrey, style=filled, shape=circle, label=\"{}\"];", 
            self.index,
            self.coordinate.0,
            self.coordinate.1,
            label,
        )
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct DecodeEdge {
    pub index: QubitIndex,
    pub neighbor0: QubitIndex,
    pub neighbor1: QubitIndex,
    pub bit_index: Option<BitIndex>,
    pub variable_index: Option<BitIndex>,
    pub keep_in_snake: bool,
}

impl WriteDot for DecodeEdge {
    fn to_dot(&self) -> String {
        let mut options = Vec::<String>::new();
        if self.variable_index.is_some() {
            options.push("color=blue".to_string());
            options.push("penwidth=2.5".to_string());
        } else {
            options.push("penwidth=1.0".to_string());
        }
        if !self.keep_in_snake {
            options.push("style=dashed".to_string());
        }
        let label = if let Some(bi) = self.bit_index {
            format!("Q{}:B[{}]", self.index, bi)
        } else {
            format!("Q{}", self.index)
        };
        format!(
            "{} -- {} [{}, label=\"{}\"];",
            self.neighbor0,
            self.neighbor1,
            options.join(", "),
            label,
        )
    }
}
