# GEM experiment suite

[![License](https://img.shields.io/github/license/Qiskit/qiskit-experiments.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)

The GEM (Generation of Entanglement by Measurement) protocol prepares a quantum state across many qubits using an efficient, measurement-based protocol. 
Generally, it is an extraordinary technical challenge to achieve this on current generation quantum devices because most preparation protocols are vulnerable to imperfections in gates and to noise. The protocol that realizes the Nishimori transition, also understood as a fault-tolerant threshold, is a constant depth method that demonstrates how measurement-based state preparation might be used on large-scale quantum devices exceeding a few hundred qubits.
Through injecting coherent errors, the GEM experiment provides an estimate for how far below the error correction threshold a device is operating.

All technical details are available on arXiv: https://arxiv.org/abs/2309.02863

Because of the computationally expensive protocol of the measurement outcome decoding, this library is mostly implemented in Rust and interfaced with [Qiskit Experiments](https://github.com/Qiskit-Extensions/qiskit-experiments) via [PyO3](https://docs.rs/pyo3/latest/pyo3/index.html). We support two minimum weight perfect matching decoders; [pymatching](https://github.com/oscarhiggott/PyMatching) and [fusion-blossom](https://github.com/yuewuo/fusion-blossom).

This library is designed to work with, not limited to, IBM Quantum processors.

## Installation

This package can be installed from the downloaded repository using pip as

```bash
cd nishimori-cats
pip install .
```

## License

[Apache License 2.0](LICENSE.txt)
