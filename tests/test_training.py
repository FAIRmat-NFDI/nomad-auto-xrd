import os

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

    os.chdir('tests')
    train(model.data)
    # Check if the model was trained successfully
