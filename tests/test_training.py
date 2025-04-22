import os
import tempfile

import pytest
from nomad.client import normalize_all, parse

from nomad_auto_xrd.training import train

data_dir = os.path.abspath(os.path.join('tests', 'data', 'training'))
log_levels = ['error', 'critical']


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
