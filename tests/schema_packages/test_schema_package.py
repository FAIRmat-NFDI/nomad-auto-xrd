import os.path

from nomad.client import normalize_all, parse

from nomad_auto_xrd.schema_packages import (
    analysis_schema,
    model_schema,
    training_schema,
)


def test_schema_package():
    test_file = os.path.join('tests', 'data', 'test.archive.yaml')
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    assert entry_archive.data is not None


def test_loading_schema_packages():
    model_schema.load()
    training_schema.load()
    analysis_schema.load()
