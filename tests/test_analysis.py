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

from nomad_auto_xrd.analysis import analyse

data_dir = os.path.abspath(os.path.join('tests', 'data', 'analysis'))

xrd_files = [
    os.path.join(data_dir, 'XRD-918-16_10.xrdml'),
    os.path.join(data_dir, 'TwoTheta_scan_powder.rasx'),
]
log_levels = ['error', 'critical']


# TODO separate into analysis entry setup fixture and analysis functionality
@pytest.mark.parametrize(
    'parsed_measurement_archives, caplog',
    [(xrd_files, log_levels)],
    indirect=True,
)
def test_analysis(parsed_measurement_archives, caplog):
    """
    Test the `analyse` functions of the AutoXRD package. Set ups the NOMAD entries that
    are needed for the analysis and then runs the analysis.
        - The XRD entries are created from the raw files. Handled by the
          `parsed_measurement_archives` fixture.
        - The model entry mimics a pre-trained model.
        - The analysis entry stores the settings for the analysis including references
          to the XRD entry and the model entry.
    """
    # prepare the pre-trained model entry
    reference_files = [
        os.path.join(data_dir, 'References', path)
        for path in os.listdir(os.path.join(data_dir, 'References'))
        if path.endswith('.cif')
    ]
    model = parse(os.path.join(data_dir, 'AutoXRDModel.archive.yaml'))[0]
    model.data.reference_files = reference_files
    model.data.xrd_model = os.path.join(data_dir, 'Models', 'XRD_Model.h5')
    model.data.pdf_model = os.path.join(data_dir, 'Models', 'PDF_Model.h5')
    normalize_all(model)

    # prepare the analysis entry
    analysis = parse(os.path.join(data_dir, 'AutoXRDAnalysis.archive.yaml'))[0]
    analysis.data.analysis_settings.auto_xrd_model = model.data
    analysis.m_setdefault('data/inputs/0')
    analysis.data.inputs[0].reference = parsed_measurement_archives[0].data
    analysis.m_setdefault('data/inputs/1')
    analysis.data.inputs[1].reference = parsed_measurement_archives[1].data
    normalize_all(analysis)

    # run the analysis
    analyse(analysis.data)

    assert analysis.data.results[0].identified_phases[0].name == 'CuPS3_136'
    assert analysis.data.results[1].identified_phases[0].name == 'Cu3P_165'
