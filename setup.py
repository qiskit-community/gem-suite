# (C) Copyright IBM 2024
#
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

"GEM Suite setup file."

from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension(
            "gem_suite.gem_core",
            binding=Binding.PyO3,
        )
    ],
    # Build a single wheel against the CPython limited API (abi3) that works on
    # every supported Python version. This floor must be kept in sync with the
    # pyo3 "abi3-py*" feature in Cargo.toml and requires-python in pyproject.toml.
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
    long_description_content_type="text/markdown",
)
