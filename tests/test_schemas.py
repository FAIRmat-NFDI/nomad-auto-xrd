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
import pytest
from nomad.client import normalize_all, parse

from nomad_auto_xrd.schema import AutoXRDAnalysis, AutoXRDModel

log_levels = ['error', 'critical']
schema_files = [
    'tests/data/schemas/AutoXRDModel.archive.yaml',
    'tests/data/schemas/AutoXRDAnalysis.archive.yaml',
]


@pytest.mark.parametrize(
    'caplog',
    [log_levels],
    indirect=True,
)
def test_schemas(caplog):
    """
    Test the schema of the AutoXRD package. This test checks if the schema of the
    AutoXRD package is valid and if the analysis function works as expected.
    """
    for schema_file in schema_files:
        entry_archive = parse(schema_file)[0]
        normalize_all(entry_archive)
        assert entry_archive.data is not None
        assert isinstance(entry_archive.data, (AutoXRDModel, AutoXRDAnalysis))
