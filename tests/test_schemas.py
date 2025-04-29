#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os

import pytest
from nomad.client import normalize_all, parse

from nomad_auto_xrd.schema import AutoXRDModel

data_dir = os.path.abspath(os.path.join('tests', 'data', 'schemas'))
log_levels = ['error', 'critical']


@pytest.mark.parametrize(
    'caplog',
    [log_levels],
    indirect=True,
)
def test_auto_xrd_model(caplog):
    """
    Test the AutoXRDModel schema by parsing the schema file.
    """
    entry_archive = parse(os.path.join(data_dir, 'AutoXRDModel.archive.yaml'))[0]
    normalize_all(entry_archive)
    assert entry_archive.data is not None
    assert isinstance(entry_archive.data, AutoXRDModel)
    assert entry_archive.results.material.topology[0].label == 'Cu6P2S8-31'
    assert entry_archive.data.reference_structures[0].system.label == 'Cu6P2S8-31'


@pytest.mark.parametrize(
    'caplog',
    [log_levels],
    indirect=True,
)
def test_auto_xrd_analysis(caplog, clean_up):
    """
    Test the AutoXRDAnalysis schema by parsing the schema file.
    """
    entry_archive = parse(os.path.join(data_dir, 'AutoXRDAnalysis.archive.yaml'))[0]
    normalize_all(entry_archive)

    assert entry_archive.data.method == 'Auto XRD Analysis'

    clean_up.track(os.path.join(data_dir, entry_archive.data.notebook))


@pytest.mark.parametrize(
    'caplog',
    [log_levels],
    indirect=True,
)
def test_auto_xrd_training(caplog, clean_up):
    """
    Test the AutoXRDAnalysis schema by parsing the schema file.
    """
    entry_archive = parse(os.path.join(data_dir, 'AutoXRDTraining.archive.yaml'))[0]
    normalize_all(entry_archive)

    assert entry_archive.data.method == 'Auto XRD Model Training'

    clean_up.track(os.path.join(data_dir, entry_archive.data.notebook))
