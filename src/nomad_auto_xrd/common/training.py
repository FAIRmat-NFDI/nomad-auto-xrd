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
import shutil
import tempfile
from random import shuffle

import numpy as np
import tensorflow as tf
import wandb
from autoXRD import solid_solns, spectrum_generation, tabulate_cifs
from wandb.integration.keras import WandbMetricsLogger

from nomad_auto_xrd.common.ml_models import build_model
from nomad_auto_xrd.common.models import (
    SimulationSettingsInput,
    TrainingSettingsInput,
    TrainModelOutput,
)
from nomad_auto_xrd.common.utils import get_total_memory_mb, timestamped_print
from nomad_auto_xrd.schema_packages.schema import AutoXRDModel, ReferenceStructure


class DataSetUp:
    """
    Class used to prepare data for training a convolutional neural network
    on a given set of X-ray diffraction spectra to perform phase identification.
    """

    def __init__(self, xrd, test_fraction=0):
        """
        Args:
            xrd: a numpy array containing xrd spectra categorized by
                their associated reference phase.
                The shape of the array should be NxMx4501x1 where:
                N = the number of reference phases,
                M = the number of augmented spectra per reference phase,
                4501 = intensities as a function of 2-theta
                (spanning from 10 to 80 degrees by default)
            test_fraction: fraction of data (xrd patterns) to reserve for testing.
                By default, all spectra will be used for training.
        """
        self.xrd = xrd
        self.test_fraction = test_fraction
        self.num_phases = len(xrd)

    @property
    def phase_indices(self):
        """List of indices to keep track of xrd spectra such that each index is
        associated with a reference phase."""
        return list(range(self.num_phases))

    @property
    def x(self):
        """Feature matrix (array of intensities used for training)."""
        intensities = []
        for augmented_spectra in self.xrd:
            intensities.extend(augmented_spectra)
        return np.array(intensities)

    @property
    def y(self):
        """Target property to predict (one-hot encoded vectors
        associated with the reference phases)."""
        one_hot_vectors = []
        num_phases = self.num_phases
        for index, augmented_spectra in enumerate(self.xrd):
            one_hot = [0] * num_phases
            one_hot[index] = 1.0
            one_hot_vectors.extend([one_hot] * len(augmented_spectra))
        return np.array(one_hot_vectors)

    def split_training_testing(self):
        """
        Splits data into training and testing sets according to self.test_fraction.

        Returns:
            train_x, train_y, test_x, test_y: numpy arrays for training and testing.
        """
        x = self.x
        y = self.y
        combined_xy = list(zip(x, y))
        shuffle(combined_xy)

        total_samples = len(combined_xy)
        n_testing = int(self.test_fraction * total_samples)

        test_xy = combined_xy[:n_testing]
        train_xy = combined_xy[n_testing:]

        train_x, train_y = zip(*train_xy) if train_xy else ([], [])
        test_x, test_y = zip(*test_xy) if test_xy else ([], [])

        return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


def fit_model(
    train_x,
    train_y,
    model,
    settings: TrainingSettingsInput,
    callbacks: list[tf.keras.callbacks.Callback] = None,
) -> str | None:
    """
    Fits the model with the given training data and labels, with optional W&B logging.

    Returns:
        wandb_run_url: The W&B run URL if logging is enabled, else None.
    """
    if not callbacks:
        callbacks = []

    if not settings.enable_wandb:
        # Train the model without W&B logging
        model.fit(
            train_x,
            train_y,
            batch_size=settings.batch_size,
            epochs=settings.num_epochs,
            validation_split=0.2,
            shuffle=True,
            callbacks=callbacks,
        )
        return None

    with wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        reinit=True,  # Ensure a new run is started
    ) as wandb_run:
        # Train the model with W&B logging
        model.fit(
            train_x,
            train_y,
            batch_size=settings.batch_size,
            epochs=settings.num_epochs,
            validation_split=0.2,
            shuffle=True,
            callbacks=callbacks + [WandbMetricsLogger()],
        )
        wandb_run_url = wandb_run.url

    return wandb_run_url


def test_model(model, test_x, test_y):
    """
    Evaluates the model on the test set.
    """
    if test_x.size > 0 and test_y.size > 0:
        loss, acc = model.evaluate(test_x, test_y)
        timestamped_print(f'Test Accuracy: {acc * 100:.2f}%, Loss: {loss:.4f}')
    else:
        timestamped_print('No test data available for evaluation.')


def generate_reference_structures(skip_filter: bool, include_elems: bool) -> str:
    """
    Generates hypothetical solid solution structure files from CIF files available in
    the working directory and saves them in the working directory under 'References/'
    path. Also filters the CIF files based on the specified criteria and saves the
    filtered structures in the 'Filtered_CIFs/' directory.

    Args:
        structure_files (list): List of CIF files to be processed.
        skip_filter (bool): If True, skips the filtering step.
        include_elems (bool): If True, include structures with only one element in their
          composition.

    Returns:
        str: Path to the directory containing the generated reference structures.
    """
    # Reset the directory for reference structures
    reference_structures_dir = os.path.join('References')
    if os.path.exists(reference_structures_dir):
        shutil.rmtree(reference_structures_dir)
    os.makedirs(reference_structures_dir, exist_ok=True)

    # Remove the contents of the Filtered_CIFs folder if it exists
    filter_cifs_dir = os.path.join('Filtered_CIFs')
    if os.path.exists(filter_cifs_dir):
        shutil.rmtree(filter_cifs_dir)

    # Check if the input structure files exist
    if not any([cif for cif in os.listdir() if cif.endswith('.cif')]):
        raise ValueError('Either no CIF files are provided.')

    if skip_filter:
        # Copy the input files to the reference directory without filtering
        for path in os.listdir():
            if not path.endswith('.cif'):
                continue
            file = os.path.basename(path)
            reference_path = os.path.join(reference_structures_dir, file)
            if os.path.isfile(path):
                shutil.copy(path, reference_path)
    else:
        # Run the filtering process: adds the filtered structures to the
        # reference_structures_dir
        with tempfile.TemporaryDirectory() as tmp_input_dir:
            for path in os.listdir():
                if not path.endswith('.cif'):
                    continue
                if os.path.isfile(path):
                    shutil.copy(path, tmp_input_dir)
            tabulate_cifs.main(
                tmp_input_dir,
                reference_structures_dir,
                include_elems,
            )

    # Generate hypothetical solid solutions
    solid_solns.main(reference_structures_dir)

    return reference_structures_dir


def get_cif_files_from_folder(folder_name):
    """Returns a list of CIF files with their full paths in the specified folder."""
    cif_files_names = []
    for file in os.listdir(folder_name):
        if file.endswith('.cif'):
            full_path = os.path.join(folder_name, file)
            cif_files_names.append(full_path)
    return cif_files_names


def setup_data_and_model(
    reference_structures_dir: str,
    simulation_settings: SimulationSettingsInput,
    test_fraction: float,
    is_pdf: bool = False,
):
    """
    Generates simulated spectra (XRD or PDF) for the reference structures.
    """
    spectra_generator = spectrum_generation.SpectraGenerator(
        reference_dir=reference_structures_dir,
        num_spectra=simulation_settings.num_patterns,
        max_texture=simulation_settings.max_texture,
        min_domain_size=simulation_settings.min_domain_size,
        max_domain_size=simulation_settings.max_domain_size,
        max_strain=simulation_settings.max_strain,
        max_shift=simulation_settings.max_shift,
        impur_amt=simulation_settings.impur_amt,
        min_angle=simulation_settings.min_angle,
        max_angle=simulation_settings.max_angle,
        separate=simulation_settings.separate,
        is_pdf=is_pdf,
    )
    spectra = spectra_generator.augmented_spectra
    dataset = DataSetUp(spectra, test_fraction=test_fraction)
    train_x, train_y, test_x, test_y = dataset.split_training_testing()
    model = build_model(train_x.shape[1:], dataset.num_phases, is_pdf=is_pdf)

    return train_x, train_y, test_x, test_y, model


def train(
    working_directory,
    simulation_settings: SimulationSettingsInput,
    training_settings: TrainingSettingsInput,
    includes_pdf: bool = True,
    callbacks: list[tf.keras.callbacks.Callback] = None,
):
    """
    Main function to run the XRD model pipeline: generate reference structures,
    simulate XRD patterns, setup data for training, initialize the model, and train it.
    Runs the autoXRD training pipeline in the specified working directory. Saves the
    path of the reference structures and the trained model along with the W&B run URL
    in the `model_entry`.
    """
    output = TrainModelOutput()
    original_dir = os.getcwd()
    rel_working_dir = (
        working_directory if working_directory else os.path.join('auto_xrd_training')
    )
    working_dir = os.path.join(original_dir, rel_working_dir)
    os.makedirs(working_dir, exist_ok=True)

    try:
        # move the cifs into the working directory
        for cif_file in simulation_settings.structure_files:
            shutil.copy(cif_file, working_dir)
        os.chdir(working_dir)

        # Generate and save the reference structures
        reference_structures_dir = generate_reference_structures(
            simulation_settings.skip_filter,
            simulation_settings.include_elems,
        )
        output.reference_structure_paths = []
        for reference_cif_file in get_cif_files_from_folder(
            os.path.join(working_dir, reference_structures_dir)
        ):
            output.reference_structure_paths.append(reference_cif_file)

        # Create datasets and build models
        xrd_train_x, xrd_train_y, xrd_test_x, xrd_test_y, xrd_model = (
            setup_data_and_model(
                reference_structures_dir=reference_structures_dir,
                simulation_settings=simulation_settings,
                test_fraction=training_settings.test_fraction,
            )
        )
        if includes_pdf:
            pdf_train_x, pdf_train_y, pdf_test_x, pdf_test_y, pdf_model = (
                setup_data_and_model(
                    reference_structures_dir=reference_structures_dir,
                    simulation_settings=simulation_settings,
                    test_fraction=training_settings.test_fraction,
                    is_pdf=True,
                )
            )

        timestamped_print(
            'Dataset and model setup complete. Current Memory Usage: '
            f'{get_total_memory_mb():.1f} MB'
        )

        # Clean up the Models directory
        models_dir = os.path.join(working_dir, 'Models')
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)
        os.makedirs(models_dir, exist_ok=True)

        # Train the models
        timestamped_print('Starting XRD model training...')
        output.wandb_run_url_xrd = fit_model(
            xrd_train_x,
            xrd_train_y,
            xrd_model,
            training_settings,
            callbacks,
        )
        xrd_model_path = os.path.join(models_dir, 'XRD_Model.h5')
        xrd_model.save(xrd_model_path, include_optimizer=False)
        output.xrd_model_path = xrd_model_path
        timestamped_print(
            f'XRD model training complete. Current Memory Usage: '
            f'{get_total_memory_mb():.1f} MB'
        )
        test_model(xrd_model, xrd_test_x, xrd_test_y)

        if includes_pdf:
            timestamped_print('Starting PDF model training...')
            output.wandb_run_url_pdf = fit_model(
                pdf_train_x,
                pdf_train_y,
                pdf_model,
                training_settings,
                callbacks,
            )
            pdf_model_path = os.path.join(models_dir, 'PDF_Model.h5')
            pdf_model.save(pdf_model_path, include_optimizer=False)
            output.pdf_model_path = pdf_model_path
            timestamped_print(
                'PDF model training complete. Current Memory Usage: '
                f'{get_total_memory_mb():.1f} MB'
            )
            test_model(pdf_model, pdf_test_x, pdf_test_y)

        return output

    except Exception as e:
        timestamped_print(f'Error during training: {e}')
    finally:
        # Restore the original working directory
        os.chdir(original_dir)


def train_nomad_model(model: AutoXRDModel):
    """
    Trains an autoXRD model using the NOMAD schema instance `AutoXRDModel`. This wrapper
    ensures compatibility between the NOMAD schema and the autoXRD training pipeline.
    The function updates the `model` instance with trained model paths and reference
    structure paths.

    Args:
        model (AutoXRDModel): An instance of the AutoXRDModel schema containing all
            the necessary settings for training. The model will be updated with the
            trained model.
    """
    if not isinstance(model, AutoXRDModel):
        raise TypeError('`model` must be an instance of AutoXRDModel')

    simulation_settings = SimulationSettingsInput(
        structure_files=model.simulation_settings.structure_files,
        max_texture=float(model.simulation_settings.max_texture),
        min_domain_size=float(model.simulation_settings.min_domain_size.magnitude),
        max_domain_size=float(model.simulation_settings.max_domain_size.magnitude),
        max_strain=float(model.simulation_settings.max_strain),
        num_patterns=int(model.simulation_settings.num_patterns),
        min_angle=float(model.simulation_settings.min_angle.magnitude),
        max_angle=float(model.simulation_settings.max_angle.magnitude),
        max_shift=float(model.simulation_settings.max_shift.magnitude),
        separate=model.simulation_settings.separate,
        impur_amt=float(model.simulation_settings.impur_amt),
        skip_filter=model.simulation_settings.skip_filter,
        include_elems=model.simulation_settings.include_elems,
    )
    training_settings = TrainingSettingsInput(
        num_epochs=int(model.training_settings.num_epochs),
        batch_size=int(model.training_settings.batch_size),
        learning_rate=float(model.training_settings.learning_rate),
        seed=int(model.training_settings.seed),
        test_fraction=float(model.training_settings.test_fraction),
    )

    # Train the model
    output = train(
        model.working_directory,
        simulation_settings,
        training_settings,
        includes_pdf=model.includes_pdf,
    )

    # Update the model with the training results
    model.xrd_model = output.xrd_model_path
    model.pdf_model = output.pdf_model_path
    model.wandb_run_url_xrd = output.wandb_run_url_xrd
    model.wandb_run_url_pdf = output.wandb_run_url_pdf
    reference_structures = []
    for cif_path in output.reference_structure_paths:
        reference_structure = ReferenceStructure(
            name=os.path.basename(cif_path).split('.cif')[0],
            cif_file=cif_path,
        )
        reference_structures.append(reference_structure)
    model.reference_structures = reference_structures
