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
