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


def generate_reference_structures(
    structure_files,
    skip_filter,
    include_elems,
):
    """
    Generates hypothetical solid solution structure files from provided CIF files and
    saves them in the working directory under 'References/' path.

    Args:
        structure_files (list): List of CIF files to be processed.
        skip_filter (bool): If True, skips the filtering step.
        include_elems (bool): If True, include structures with only one element in their
          composition.

    Returns:
        str: Path to the directory containing the generated reference structures.
    """
    # Reset the directories for input and reference structures
    input_structures_dir = os.path.join('Input_Structures')
    if os.path.exists(input_structures_dir):
        shutil.rmtree(input_structures_dir)
    os.makedirs(input_structures_dir, exist_ok=True)
    reference_structures_dir = os.path.join('References')
    if os.path.exists(reference_structures_dir):
        shutil.rmtree(reference_structures_dir)
    os.makedirs(reference_structures_dir, exist_ok=True)

    # Remove the contents of the Filtered_CIFs folder if it exists
    filter_cifs_dir = os.path.join('Filtered_CIFs')
    if os.path.exists(filter_cifs_dir):
        shutil.rmtree(filter_cifs_dir)

    # Create symbolic links for the input CIF files
    for file in structure_files:
        if not os.path.exists(file):
            print(f'File does not exist: {file}')
            continue
        input_path = os.path.join(input_structures_dir, os.path.basename(file))
        if os.path.islink(input_path):
            print(f'Symlink already exists: {input_path}')
            continue
        os.symlink(file, input_path)
    if not os.listdir(input_structures_dir):
        raise ValueError(
            'No CIF files found at the paths provided in the `structure_files`.'
        )

    if skip_filter:
        # Create symlinks to the inputs with running filtering
        for file in os.listdir(input_structures_dir):
            input_path = os.path.join(input_structures_dir, file)
            reference_path = os.path.join(reference_structures_dir, file)
            if os.path.islink(input_path):
                target_path = os.readlink(input_path)
                os.symlink(target_path, reference_path)
    else:
        # Run the filtering process: adds the filtered structures to the
        # reference_structures_path directory
        tabulate_cifs.main(
            input_structures_dir,
            reference_structures_dir,
            include_elems,
        )

    # Generate hypothetical solid solutions
    solid_solns.main(reference_structures_dir)

    return reference_structures_dir


def train(model_entry: AutoXRDModel):
    """
    Main function to run the XRD model pipeline: generate reference structures,
    simulate XRD patterns, setup data for training, initialize the model, and train it.
    Runs the autoXRD training pipeline in the specified working directory. Saves the
    path of the reference structures and the trained model along with the W&B run URL
    in the `model_entry`.

    Args:
        model_entry (AutoXRDModel): The model object containing all the necessary
            settings. The object will also be updated with the trained model and W&B
            run URLs.
    """
    original_dir = os.getcwd()
    working_dir = (
        model_entry.working_directory
        if model_entry.working_directory
        else os.path.join('auto_xrd_training')
    )
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)
    reference_structures_dir = generate_reference_structures(
        model_entry.simulation_settings.structure_files,
        model_entry.simulation_settings.skip_filter,
        model_entry.simulation_settings.include_elems,
    )
    # Generate XRD patterns
    xrd_obj = spectrum_generation.SpectraGenerator(
        reference_dir=reference_structures_dir,
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
    models_dir = os.path.join(working_dir, 'Models')
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
    model_entry.wandb_run_url_xrd = wandb_run_url_xrd

    # Save the trained model
    xrd_model_path = os.path.join(models_dir, 'XRD_Model.h5')
    model.save(xrd_model_path, include_optimizer=False)
    model_entry.xrd_model = xrd_model_path

    # Test the model
    test_model(model, test_x, test_y)

    if not model_entry.includes_pdf:
        return

    # If `model_config.includes_pdf` is True, train another model on PDFs
    pdf_spectras = spectrum_generation.SpectraGenerator(
        reference_dir=reference_structures_dir,
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
    model_entry.wandb_run_url_pdf = wandb_run_url_pdf

    # Save the PDF model
    pdf_model_path = os.path.join(models_dir, 'PDF_Model.h5')
    model_pdf.save(pdf_model_path, include_optimizer=False)
    model_entry.pdf_model = pdf_model_path

    # Test the PDF model
    test_model(model_pdf, test_x_pdf, test_y_pdf)

    # Restore the original working directory and saves the reference structures
    os.chdir(original_dir)
    model_entry.reference_files = get_cif_files_from_folder(
        os.path.join(working_dir, reference_structures_dir)
    )


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
