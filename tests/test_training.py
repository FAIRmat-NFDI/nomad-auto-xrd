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
import tempfile

import pytest
from nomad.client import normalize_all, parse

from nomad_auto_xrd.training import train

# Check environment variable
run_pipeline_tests = os.environ.get('RUN_PIPELINE_TESTS', 'false').lower() == 'true'

data_dir = os.path.abspath(os.path.join('tests', 'data', 'training'))
log_levels = ['error', 'critical']


@pytest.mark.skipif(
    not run_pipeline_tests,
    reason='Skipping training test. Set environment variable RUN_PIPELINE_TESTS=true '
    'to run.',
)
@pytest.mark.parametrize(
    'caplog',
    [log_levels],
    indirect=True,
)
def test_train(caplog):
    """
    Test the `train` function of the AutoXRD package. Set ups the NOMAD entries for the
    Auto XRD model, passes it to the training module and checks if the training was
    successful.
    """
    # Setup the model entry
    model = parse(os.path.join(data_dir, 'AutoXRDModel.archive.yaml'))[0]
    structure_files = os.listdir(os.path.join(data_dir, 'structure_files'))
    structure_files = [
        os.path.join(data_dir, 'structure_files', path) for path in structure_files
    ]
    model.m_setdefault('data/simulation_settings')
    model.data.simulation_settings.structure_files = structure_files
    normalize_all(model)

    # Create a temporary directory for the training
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Set the training directory
        model.data.working_directory = tmpdirname
        train(model.data)
        normalize_all(model)

        assert os.path.exists(os.path.join(tmpdirname, model.data.xrd_model))
        assert os.path.exists(os.path.join(tmpdirname, model.data.pdf_model))
        assert os.path.exists(os.path.join(tmpdirname, model.data.reference_files[0]))
