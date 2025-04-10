import os

import pytest
from nomad.client import normalize_all
from nomad.datamodel import EntryArchive

from nomad_auto_xrd.analysis import analyse
from nomad_auto_xrd.schema import (
    AutoXRDAnalysis,
    AutoXRDModel,
)

data_dir = os.path.join('tests', 'data', 'analysis')

xrd_files = [
    os.path.join(data_dir, 'XRD-918-16_10.xrdml'),
    os.path.join(data_dir, 'TwoTheta_scan_powder.rasx'),
]
log_levels = ['error', 'critical']


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
    # prepare the pre-trained model
    reference_files = [
        os.path.join(data_dir, 'References', path)
        for path in os.listdir(os.path.join(data_dir, 'References'))
        if path.endswith('.cif')
    ]
    model = AutoXRDModel(includes_pdf=True)
    model.reference_files = reference_files
    model.xrd_model = os.path.join(data_dir, 'Models', 'XRD_Model.h5')
    model.pdf_model = os.path.join(data_dir, 'Models', 'PDF_Model.h5')

    normalize_all(EntryArchive(data=model))

    # prepare the analysis
    analysis = AutoXRDAnalysis()
    analysis.m_setdefault('analysis_settings')
    analysis.analysis_settings.auto_xrd_model = model
    analysis.analysis_settings.min_angle = 10
    analysis.analysis_settings.max_angle = 60
    analysis.m_setdefault('inputs/0')
    analysis.inputs[0].reference = parsed_measurement_archives[0].data
    analysis.m_setdefault('inputs/1')
    analysis.inputs[1].reference = parsed_measurement_archives[1].data
    normalize_all(EntryArchive(data=analysis))

    results = analyse(analysis)
    print(results)
