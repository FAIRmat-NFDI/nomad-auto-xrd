from nomad_analysis.auto_xrd.schema import (
    AutoXRDModel,
    SimulationSettings,
    TrainingSettings,
)

from nomad_auto_xrd.training import train

training_settings = TrainingSettings(
    num_epochs=2,
    batch_size=32,
    learning_rate=0.001,
    seed=43,
)
simulation_settings = SimulationSettings()
model = AutoXRDModel(
    working_directory='tests/data',
    training_settings=training_settings,
    simulation_settings=simulation_settings,
    includes_pdf=True,
)


def test_placeholder():
    import os

    structure_files = os.listdir('tests/data/All_CIFs')
    structure_files = [
        os.path.join('tests', 'data', 'All_CIFs', path) for path in structure_files
    ]
    print(structure_files)
    model.simulation_settings.structure_files = structure_files

    train(model)
