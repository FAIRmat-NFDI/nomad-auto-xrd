import logging
import os
import shutil
import warnings
from dataclasses import dataclass
from random import shuffle

import numpy as np
import tensorflow as tf
from autoXRD import solid_solns, spectrum_generation, tabulate_cifs  # noqa: E402
from nomad.datamodel import EntryArchive
from tensorflow.keras.callbacks import Callback  # type: ignore

# Import necessary modules from autoXRD
from nomad_auto_xrd.schema import (
    AutoXRDModel,
    SimulationSettings,
    TrainingSettings,
)

# Suppress specific warnings
warnings.filterwarnings('ignore')
try:
    import wandb
except ImportError:
    wandb = None  # Wandb is optional

# Optional NOMAD imports

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for running the XRD model."""

    # Paths and directories
    references_dir: str = 'References'
    all_cifs_dir: str = 'All_CIFs'
    models_dir: str = 'Models'
    xrd_output_file: str = 'XRD.npy'
    pdf_output_file: str = 'PDF.npy'

    # Spectra generation parameters
    max_texture: float = 0.5
    min_domain_size: float = 5.0
    max_domain_size: float = 30.0
    max_strain: float = 0.03
    num_spectra: int = 50
    min_angle: float = 20.0
    max_angle: float = 80.00
    max_shift: float = 0.1
    separate: bool = True
    impur_amt: float = 0.0
    skip_filter: bool = False
    include_elems: bool = True
    inc_pdf: bool = False
    save_pdf: bool = False

    # Training parameters
    num_epochs: int = 50
    test_fraction: float = 0.2

    # Wandb configuration
    enable_wandb: bool = False
    wandb_project: str = 'xrd_model'
    wandb_entity: str = 'your_entity_name'

    # NOMAD configuration
    save_nomad_metadata: bool = True


# Custom Dropout layer used in the model
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config

    # Always apply dropout
    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)


class DataSetUp(object):  # noqa: UP004
    """
    Class used to prepare data for training a convolutional neural network
    on a given set of X-ray diffraction spectra to perform phase identification.
    """

    def __init__(self, xrd, testing_fraction=0):
        """
        Args:
            xrd: a numpy array containing xrd spectra categorized by
                their associated reference phase.
                The shape of the array should be NxMx4501x1 where:
                N = the number of reference phases,
                M = the number of augmented spectra per reference phase,
                4501 = intensities as a function of 2-theta
                (spanning from 10 to 80 degrees by default)
            testing_fraction: fraction of data (xrd patterns) to reserve for testing.
                By default, all spectra will be used for training.
        """
        self.xrd = xrd
        self.testing_fraction = testing_fraction
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
        Splits data into training and testing sets according to self.testing_fraction.

        Returns:
            train_x, train_y, test_x, test_y: numpy arrays for training and testing.
        """
        x = self.x
        y = self.y
        combined_xy = list(zip(x, y))
        shuffle(combined_xy)

        total_samples = len(combined_xy)
        n_testing = int(self.testing_fraction * total_samples)

        test_xy = combined_xy[:n_testing]
        train_xy = combined_xy[n_testing:]

        train_x, train_y = zip(*train_xy) if train_xy else ([], [])
        test_x, test_y = zip(*test_xy) if test_xy else ([], [])

        return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


def build_model(input_shape, n_phases, is_pdf, n_dense=[3100, 1200], dropout_rate=0.7):
    """
    Builds the CNN model based on whether it is for PDF or XRD analysis.
    """
    layers = []
    if is_pdf:
        # Architecture for PDF analysis
        layers.extend(
            [
                tf.keras.layers.Conv1D(
                    64, 60, activation='relu', padding='same', input_shape=input_shape
                ),
                tf.keras.layers.MaxPooling1D(3, strides=2, padding='same'),
                tf.keras.layers.MaxPooling1D(3, strides=2, padding='same'),
                tf.keras.layers.MaxPooling1D(2, strides=2, padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
            ]
        )
    else:
        # Architecture for XRD analysis
        layers.extend(
            [
                tf.keras.layers.Conv1D(
                    64, 35, activation='relu', padding='same', input_shape=input_shape
                ),
                tf.keras.layers.MaxPooling1D(3, strides=2, padding='same'),
                tf.keras.layers.Conv1D(64, 30, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling1D(3, strides=2, padding='same'),
                tf.keras.layers.Conv1D(64, 25, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling1D(2, strides=2, padding='same'),
                tf.keras.layers.Conv1D(64, 20, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
                tf.keras.layers.Conv1D(64, 15, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
                tf.keras.layers.Conv1D(64, 10, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling1D(1, strides=2, padding='same'),
            ]
        )

    # Common layers
    layers.extend(
        [
            tf.keras.layers.Flatten(),
            CustomDropout(dropout_rate),
            tf.keras.layers.Dense(n_dense[0], activation='relu'),
            tf.keras.layers.BatchNormalization(),
            CustomDropout(dropout_rate),
            tf.keras.layers.Dense(n_dense[1], activation='relu'),
            tf.keras.layers.BatchNormalization(),
            CustomDropout(dropout_rate),
            tf.keras.layers.Dense(n_phases, activation='softmax'),
        ]
    )

    model = tf.keras.Sequential(layers)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'],
    )
    return model


def train_model(train_x, train_y, model, settings: TrainingSettings):
    """
    Trains the model with optional W&B logging.

    Returns:
        wandb_run_url: The wandb run URL if logging is enabled, else None.
    """
    wandb_run_url = None  # Initialize to None
    # Prepare callbacks
    callbacks = []
    if settings.enable_wandb and wandb:
        run = wandb.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            reinit=True,  # Ensure a new run is started
        )
        wandb.config.update(vars(settings))  # TODO check what this does
        callbacks.append(WandbMetricsLogger())
    else:
        run = None  # Wandb is not enabled

    # Train the model
    model.fit(
        train_x,
        train_y,
        batch_size=32,
        epochs=settings.num_epochs,
        validation_split=0.2,
        shuffle=True,
        callbacks=callbacks,
    )

    # Get the run URL if wandb is enabled
    if settings.enable_wandb and wandb and run:
        wandb_run_url = run.url
        # Finish W&B run
        wandb.finish()
        return wandb_run_url

    return None


def test_model(model, test_x, test_y):
    """
    Evaluates the model on the test set.
    """
    if test_x.size > 0 and test_y.size > 0:
        loss, acc = model.evaluate(test_x, test_y)
        print(f'Test Accuracy: {acc * 100:.2f}%')
    else:
        print('No test data available for evaluation.')


# Custom W&B callback
class WandbMetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if wandb and logs:
            wandb.log(
                {
                    'epoch': epoch,
                    'training_loss': logs.get('loss'),
                    'training_accuracy': logs.get('categorical_accuracy'),
                    'validation_loss': logs.get('val_loss'),
                    'validation_accuracy': logs.get('val_categorical_accuracy'),
                }
            )


def generate_reference_structures(working_directory: str, settings: SimulationSettings):
    """
    Generates hypothetical solid solution structure files from provided CIF files and
    saves them to the working directory under '<working_directory>/References/'.

    Args:
        working_directory (str): Path to the working directory where generated data will
            be saved.
        settings (SimulationSettings): Settings for generating hypothetical solid
            solutions.
    """
    unprocessed_structures_path = os.path.join(
        working_directory, 'unprocessed_structure_files'
    )
    reference_structures_path = os.path.join(working_directory, 'References')
    os.makedirs(reference_structures_path, exist_ok=True)

    # make dirs and copy CIFs to the working directory
    os.makedirs(working_directory, exist_ok=True)
    os.makedirs(unprocessed_structures_path, exist_ok=True)
    for file in settings.structure_files:
        try:
            shutil.copy(file, unprocessed_structures_path)
        except shutil.SameFileError:
            continue

    # Filter CIFs
    if settings.skip_filter and not os.path.exists(reference_structures_path):
        raise FileNotFoundError(
            f'"skip_filter" is True, but "{reference_structures_path}" directory was '
            'not found.'
        )
    elif not settings.skip_filter:
        if not unprocessed_structures_path:
            raise ValueError(
                'No structure files were provided. Please provide a list of CIF files.'
            )
        if os.listdir(reference_structures_path):
            raise FileExistsError(
                f'"{reference_structures_path}" already contains structure files. '
                'Please clear the directory or set "skip_filter=True".'
            )
        tabulate_cifs.main(
            unprocessed_structures_path,
            reference_structures_path,
            settings.include_elems,
        )

    # Generate hypothetical solid solutions
    solid_solns.main(reference_structures_path)

    return reference_structures_path


def train(model_entry: AutoXRDModel):
    """
    Main function to run the XRD model pipeline: generate reference structures,
    simulate XRD patterns, setup data for training, initialize the model, and train it.

    Args:
        model (AutoXRDModel): The model object containing all the necessary settings.
            The object will also be updated with the trained model and W&B run URLs.
    """

    # Prepare data
    reference_structures_path = generate_reference_structures(
        model_entry.working_directory, model_entry.simulation_settings
    )
    xrd_obj = spectrum_generation.SpectraGenerator(
        reference_dir=reference_structures_path,
        num_spectra=model_entry.simulation_settings.num_patterns,
        max_texture=model_entry.simulation_settings.max_texture,
        min_domain_size=model_entry.simulation_settings.min_domain_size.magnitude,
        max_domain_size=model_entry.simulation_settings.max_domain_size.magnitude,
        max_strain=model_entry.simulation_settings.max_strain,
        max_shift=model_entry.simulation_settings.max_shift.magnitude,
        impur_amt=model_entry.simulation_settings.impur_amt,
        min_angle=model_entry.simulation_settings.min_angle.magnitude,
        max_angle=model_entry.simulation_settings.max_angle.magnitude,
        separate=model_entry.simulation_settings.separate,
        is_pdf=False,
    )
    xrd_spectras = xrd_obj.augmented_spectra
    dataset = DataSetUp(
        xrd_spectras, testing_fraction=model_entry.training_settings.test_fraction
    )
    num_phases = dataset.num_phases
    train_x, train_y, test_x, test_y = dataset.split_training_testing()

    # Clean up the Models directory
    models_dir = os.path.join(model_entry.working_directory, 'Models')
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    model_entry.wandb_run_urls = []
    model_entry.models = []

    # Build the model
    model = build_model(train_x.shape[1:], num_phases, is_pdf=False)

    # Train the model and get the wandb run URL
    wandb_run_url_xrd = train_model(
        train_x, train_y, model, model_entry.training_settings
    )
    model_entry.wandb_run_urls.append(wandb_run_url_xrd)

    # Save the trained model
    xrd_model_path = os.path.join(models_dir, 'XRD_Model.h5')
    model.save(xrd_model_path, include_optimizer=False)
    model_entry.models.append(xrd_model_path)

    # Test the model
    test_model(model, test_x, test_y)

    if not model_entry.includes_pdf:
        return

    # If `model_config.includes_pdf` is True, train another model on PDFs
    pdf_spectras = spectrum_generation.SpectraGenerator(
        reference_dir=reference_structures_path,
        num_spectra=model_entry.simulation_settings.num_patterns,
        max_texture=model_entry.simulation_settings.max_texture,
        min_domain_size=model_entry.simulation_settings.min_domain_size.magnitude,
        max_domain_size=model_entry.simulation_settings.max_domain_size.magnitude,
        max_strain=model_entry.simulation_settings.max_strain,
        max_shift=model_entry.simulation_settings.max_shift.magnitude,
        impur_amt=model_entry.simulation_settings.impur_amt,
        min_angle=model_entry.simulation_settings.min_angle.magnitude,
        max_angle=model_entry.simulation_settings.max_angle.magnitude,
        separate=model_entry.simulation_settings.separate,
        is_pdf=True,
    ).augmented_spectra
    dataset_pdf = DataSetUp(
        pdf_spectras, testing_fraction=model_entry.training_settings.test_fraction
    )
    num_phases_pdf = dataset_pdf.num_phases
    train_x_pdf, train_y_pdf, test_x_pdf, test_y_pdf = (
        dataset_pdf.split_training_testing()
    )

    # Build the PDF model
    model_pdf = build_model(train_x_pdf.shape[1:], num_phases_pdf, is_pdf=True)

    # Train the PDF model and get the wandb run URL
    wandb_run_url_pdf = train_model(
        train_x_pdf, train_y_pdf, model_pdf, model_entry.training_settings
    )
    model_entry.wandb_run_urls.append(wandb_run_url_pdf)

    # Save the PDF model
    pdf_model_path = os.path.join(models_dir, 'PDF_Model.h5')
    model_entry.models.append(pdf_model_path)
    model_pdf.save(pdf_model_path, include_optimizer=False)

    # Test the PDF model
    test_model(model_pdf, test_x_pdf, test_y_pdf)


def get_cif_files_from_folder(folder_name):
    """Returns a list of CIF files with their full paths in the specified folder."""
    cif_files_names = []
    for file in os.listdir(folder_name):
        if file.endswith('.cif'):
            full_path = os.path.join(folder_name, file)
            cif_files_names.append(full_path)
    return cif_files_names


def save_model_metadata(
    config: ModelConfig,
    wandb_run_url_xrd: str = None,
    wandb_run_url_pdf: str = None,
):
    """Creates a NOMAD archive to store model metadata."""
    cif_files = get_cif_files_from_folder(config.references_dir)

    # Corrected parameters
    model_params = {
        'cif_files': cif_files,
        'xrd_model_file': os.path.join(config.models_dir, 'XRD_Model.h5')
        if os.path.exists(os.path.join(config.models_dir, 'XRD_Model.h5'))
        else None,
        'pdf_model_file': os.path.join(config.models_dir, 'PDF_Model.h5')
        if os.path.exists(os.path.join(config.models_dir, 'PDF_Model.h5'))
        else None,
        'wandb_run_url_xrd': wandb_run_url_xrd,
        'wandb_run_url_pdf': wandb_run_url_pdf,
        'max_texture': config.max_texture,
        'min_domain_size': config.min_domain_size,
        'max_domain_size': config.max_domain_size,
        'max_strain': config.max_strain,
        'num_patterns': config.num_spectra,  # Corrected key
        'min_angle': config.min_angle,
        'max_angle': config.max_angle,
        'max_shift': config.max_shift,
        'separate': config.separate,
        'impur_amt': config.impur_amt,
        'skip_filter': config.skip_filter,
        'num_epochs': config.num_epochs,
        'test_fraction': config.test_fraction,
    }

    # Remove keys with None values
    model_params = {k: v for k, v in model_params.items() if v is not None}

    archive = EntryArchive(data=AutoXRDModel(**model_params))
    output_file = 'model_metadata.archive.json'
    with open(output_file, 'w') as f:
        f.write(archive.m_to_json(indent=4))
    print(f'Model metadata saved to {output_file}')
