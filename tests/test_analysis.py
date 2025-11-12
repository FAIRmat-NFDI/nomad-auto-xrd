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

from nomad_auto_xrd.common.analysis import analyze
from nomad_auto_xrd.schema_packages.schema import ReferenceStructure

# Check environment variable
run_pipeline_tests = os.environ.get('RUN_PIPELINE_TESTS', 'false').lower() == 'true'

data_dir = os.path.abspath(os.path.join('tests', 'data', 'analysis'))

xrd_files = [
    os.path.join(data_dir, 'XRD-918-16_10.xrdml'),
    os.path.join(data_dir, 'TwoTheta_scan_powder.rasx'),
]
log_levels = ['error', 'critical']


@pytest.mark.skipif(
    not run_pipeline_tests,
    reason='Skipping analysis test. Set environment variable RUN_PIPELINE_TESTS=true '
    'to run.',
)
@pytest.mark.parametrize(
    'parsed_measurement_archives, caplog',
    [(xrd_files, log_levels)],
    indirect=True,
)
def test_analysis(parsed_measurement_archives, caplog, clean_up):
    """
    Test the `analyze` functions of the AutoXRD package. Set ups the NOMAD entries that
    are needed for the analysis and then runs the analysis.
        - The XRD entries are created from the raw files. Handled by the
          `parsed_measurement_archives` fixture.
        - The model entry mimics a pre-trained model.
        - The analysis entry stores the settings for the analysis including references
          to the XRD entry and the model entry.
    """
    expected_num_results = 4  # 2 from single patterns + 2 from multi-pattern

    # prepare the pre-trained model entry
    reference_files = [
        os.path.join(data_dir, 'References', path)
        for path in os.listdir(os.path.join(data_dir, 'References'))
        if path.endswith('.cif')
    ]
    model = parse(os.path.join(data_dir, 'AutoXRDModel.archive.yaml'))[0]
    for reference_file in reference_files:
        reference_structure = ReferenceStructure(
            name=os.path.basename(reference_file).split('.cif')[0],
            cif_file=reference_file,
        )
        model.data.reference_structures.append(reference_structure)
    model.data.reference_files = reference_files
    model.data.xrd_model = os.path.join(data_dir, 'Models', 'XRD_Model.h5')
    model.data.pdf_model = os.path.join(data_dir, 'Models', 'PDF_Model.h5')
    normalize_all(model)

    # prepare the multi-pattern XRD entry
    multi_pattern_xrd = parsed_measurement_archives[0]
    multi_pattern_xrd.data.results[1] = multi_pattern_xrd.data.results[0]
    normalize_all(multi_pattern_xrd)

    # prepare the analysis entry
    analysis = parse(os.path.join(data_dir, 'AutoXRDAnalysis.archive.yaml'))[0]
    analysis.m_setdefault('data/analysis_settings')
    analysis.data.analysis_settings.auto_xrd_model = model.data
    analysis.m_setdefault('data/inputs/0')
    analysis.data.inputs[0].reference = parsed_measurement_archives[0].data
    analysis.m_setdefault('data/inputs/1')
    analysis.data.inputs[1].reference = parsed_measurement_archives[1].data
    analysis.m_setdefault('data/inputs/2')
    analysis.data.inputs[2].reference = multi_pattern_xrd.data
    normalize_all(analysis)

    # run the analysis
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        try:
            analyze(analysis.data)
        finally:
            os.chdir(original_dir)

    # check that results are created
    assert len(analysis.data.results) == expected_num_results
    for result in analysis.data.results:
        assert result is not None

    # clean up the created files
    clean_up.track(os.path.join(data_dir, analysis.data.notebook))
