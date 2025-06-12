#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
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
from typing import (
    TYPE_CHECKING,
)

import nbformat
import numpy as np
from ase.io import read
from matid import SymmetryAnalyzer
from nomad.datamodel import ArchiveSection
from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.annotations import (
    BrowserAnnotation,
    ELNAnnotation,
    ELNComponentEnum,
    Filter,
    SectionProperties,
)
from nomad.datamodel.metainfo.basesections import Entity, SectionReference
from nomad.datamodel.results import Material, SymmetryNew, System
from nomad.metainfo import (
    MProxy,
    Quantity,
    SchemaPackage,
    Section,
    SubSection,
)
from nomad.normalizing.common import nomad_atoms_from_ase_atoms
from nomad.normalizing.topology import add_system, add_system_info
from nomad_analysis.jupyter.schema import JupyterAnalysis
from nomad_measurements.xrd.schema import XRayDiffraction

if TYPE_CHECKING:
    from structlog.stdlib import (
        BoundLogger,
    )

m_package = SchemaPackage()


class Model(Entity, Schema):
    """
    Schema for a generic machine learning model. If the saved model file is from PyTorch
    or TensorFlow, the model can be loaded using the `load_model` method. These model
    file also normalized to populate the current model section.
    """


class SimulationSettings(ArchiveSection):
    """
    A schema for the settings for simulating XRD patterns.
    """

    structure_files = Quantity(
        type=str,
        shape=['*'],
        description='Path to structure file (CIF) containing crystal structure.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.FileEditQuantity,
        ),
        a_browser=BrowserAnnotation(adaptor='RawFileAdaptor'),
    )
    max_texture = Quantity(
        type=np.float64,
        description='Maximum texture value for the simualtions.',
        default=0.5,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    min_domain_size = Quantity(
        type=np.float64,
        description='Minimum domain size.',
        unit='nm',
        default=5.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    max_domain_size = Quantity(
        type=np.float64,
        description='Maximum domain size.',
        unit='nm',
        default=30.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    max_strain = Quantity(
        type=np.float64,
        description='Maximum strain value.',
        default=0.03,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    num_patterns = Quantity(
        type=int,
        description='Number of XRD patterns simulated per phase.',
        default=50,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    min_angle = Quantity(
        type=np.float64,
        description='Minimum angle value.',
        unit='deg',
        default=10.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    max_angle = Quantity(
        type=np.float64,
        description='Maximum angle value.',
        unit='deg',
        default=80.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    max_shift = Quantity(
        type=np.float64,
        description='Maximum shift value.',
        unit='deg',
        default=0.1,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    separate = Quantity(
        type=bool,
        description='Separate flag.',
        default=True,
        a_eln=ELNAnnotation(
            component='BoolEditQuantity',
        ),
    )
    impur_amt = Quantity(
        type=np.float64,
        description='Impurity amount.',
        default=0.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    skip_filter = Quantity(
        type=bool,
        description='Skip filter flag.',
        default=False,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    include_elems = Quantity(
        type=bool,
        description='Include elements flag.',
        default=True,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )


class TrainingSettings(ArchiveSection):
    """
    A schema for the settings for training the model.
    """

    num_epochs = Quantity(
        type=int,
        description='Number of training epochs.',
        default=50,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    batch_size = Quantity(
        type=int,
        description='Batch size for training.',
        default=32,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    learning_rate = Quantity(
        type=np.float64,
        description='Learning rate for training.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    seed = Quantity(
        type=int,
        description='Seed for random number generator.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    test_fraction = Quantity(
        type=np.float64,
        description='Fraction of data used for testing.',
        default=0.2,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    enable_wandb = Quantity(
        type=bool,
        description='Flag to enable "Weights and Biases" logging.',
        default=False,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    wandb_project = Quantity(
        type=str,
        description='"Weights and Biases" project name.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    wandb_entity = Quantity(
        type=str,
        description='"Weights and Biases" entity name.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )


class ReferenceStructure(ArchiveSection):
    """
    A schema for the reference structures.
    """

    name = Quantity(
        type=str,
        description="""
        A label for the reference structure that is also generated from model
        inference.
        """,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    cif_file = Quantity(
        type=str,
        description='Path to the CIF file of the reference structure.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.FileEditQuantity,
        ),
        a_browser=BrowserAnnotation(adaptor='RawFileAdaptor'),
    )
    system = Quantity(
        type=System,
        description='`System` section generated based on the CIF file of the phase.',
    )


class AutoXRDModel(Entity, Schema):
    """
    Section for describing an auto XRD model.
    """

    m_def = Section(
        description="""
        Based on the structure files (CIF files) added, XRD patterns are simulated
        for different phase compositions and structures. The simulated XRD patterns are
        then used to train a machine learning model to predict the phase composition
        and structure from the XRD data.""",
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'working_directory',
                    'xrd_model',
                    'wandb_run_url_xrd',
                    'includes_pdf',
                    'pdf_model',
                    'wandb_run_url_pdf',
                    'simulation_settings',
                    'training_settings',
                    'reference_structures',
                ],
            ),
        ),
    )
    working_directory = Quantity(
        type=str,
        description='Path to the directory containing the simulated data and trained '
        'models.',
        default='auto_xrd_training',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.StringEditQuantity,
        ),
    )
    xrd_model = Quantity(
        type=str,
        description='Path to the trained XRD model file.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.FileEditQuantity,
        ),
        a_browser=BrowserAnnotation(adaptor='RawFileAdaptor'),
    )
    pdf_model = Quantity(
        type=str,
        description='Path to the trained XRD model file.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.FileEditQuantity,
        ),
        a_browser=BrowserAnnotation(adaptor='RawFileAdaptor'),
    )
    wandb_run_url_xrd = Quantity(
        type=str,
        description='URL to the "Weights and Biases" run for training XRD model.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.URLEditQuantity),
    )
    wandb_run_url_pdf = Quantity(
        type=str,
        description='URL to the "Weights and Biases" run for training PDF model.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.URLEditQuantity),
    )
    includes_pdf = Quantity(
        type=bool,
        description='Flag to indicate if an additional model was trained using the '
        'virtual pairwise distribution functions or PDFs computed through a Fourier '
        'transform of the simulated XRD patterns.',
        default=True,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
            default=True,
        ),
    )
    simulation_settings = SubSection(
        section_def=SimulationSettings,
        description='Settings for simulating XRD patterns.',
    )
    training_settings = SubSection(
        section_def=TrainingSettings,
        description='Settings for training the model.',
    )
    reference_structures = SubSection(
        section_def=ReferenceStructure,
        repeats=True,
    )

    def normalize(self, archive: 'ArchiveSection', logger: 'BoundLogger'):
        super().normalize(archive, logger)
        if self.reference_structures:
            # Read the reference CIF files and convert them into ase atoms
            ase_atoms_list = []
            for reference_structure in self.reference_structures:
                cif_file = reference_structure.cif_file
                if not cif_file or not cif_file.endswith('.cif'):
                    logger.warn(
                        f'Cannot parse structure file: {cif_file}. '
                        'Should be a "*.cif" file.'
                    )
                    continue
                with archive.m_context.raw_file(cif_file) as file:
                    try:
                        ase_atoms_list.append(read(file.name))
                    except RuntimeError:
                        logger.warn(f'Cannot parse cif file: {cif_file}.')

            # Let's save the composition and structure into archive.results.material
            if not archive.results.material:
                archive.results.material = Material()

            # populate elemets from a set aof all the elemsts in ase_atoms
            elements = set()
            for ase_atoms in ase_atoms_list:
                elements.update(ase_atoms.get_chemical_symbols())
            archive.results.material.elements = list(elements)

            # Create a System: this is a NOMAD specific data structure for
            # storing structural and chemical information that is suitable for both
            # experiments and simulations.
            topology = {}
            labels = []
            for ase_atoms in ase_atoms_list:
                symmetry = SymmetryNew()
                symmetry_analyzer = SymmetryAnalyzer(ase_atoms, symmetry_tol=1)
                symmetry.bravais_lattice = symmetry_analyzer.get_bravais_lattice()
                symmetry.space_group_number = symmetry_analyzer.get_space_group_number()
                symmetry.space_group_symbol = (
                    symmetry_analyzer.get_space_group_international_short()
                )
                symmetry.crystal_system = symmetry_analyzer.get_crystal_system()
                symmetry.point_group = symmetry_analyzer.get_point_group()
                label = (
                    f'{ase_atoms.get_chemical_formula()}-{symmetry.space_group_number}'
                )
                labels.append(label)
                system = System(
                    atoms=nomad_atoms_from_ase_atoms(ase_atoms),
                    label=label,
                    description='Reference structure used to train the auto-XRD model.',
                    structural_type='bulk',
                    dimensionality='3D',
                    symmetry=symmetry,
                )
                add_system_info(system, topology)
                add_system(system, topology)

            archive.results.material.topology = list(topology.values())
            topology_m_proxies = dict()
            for i, system in enumerate(archive.results.material.topology):
                topology_m_proxies[system.label] = f'#/results/material/topology/{i}'

            # connect `data.reference_structures[i].system` and
            # `results.material.topology[j]` using the label
            for i, label in enumerate(labels):
                self.reference_structures[i].system = topology_m_proxies[label]


class AutoXRDModelReference(SectionReference):
    """
    A reference to an `AutoXRDModel` entry.
    """

    reference = Quantity(
        type=AutoXRDModel,
        description='A reference to an `AutoXRDModel` entry.',
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )

    def normalize(self, archive: 'ArchiveSection', logger: 'BoundLogger'):
        super().normalize(archive, logger)
        if self.reference and self.reference.name:
            self.name = self.reference.name


class AutoXRDMeasurementReference(SectionReference):
    """
    A reference to an `XRayDiffraction` entry.
    """

    reference = Quantity(
        type=XRayDiffraction,
        description='A reference to an `Measurement` entry.',
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )

    def normalize(self, archive: 'ArchiveSection', logger: 'BoundLogger'):
        super().normalize(archive, logger)
        if self.reference and self.reference.name:
            self.name = self.reference.name


class AnalysisSettings(ArchiveSection):
    """
    A schema for the settings for running the analysis.
    """

    auto_xrd_model = Quantity(
        type=AutoXRDModel,
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
        description='Reference to an `AutoXRDModel` entry.',
    )
    max_phases = Quantity(
        type=int,
        description='Maximum number of phases to identify.',
        default=5,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    cutoff_intensity = Quantity(
        type=float,
        description='Cutoff intensity for the XRD pattern.',
        default=0.05,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    min_confidence = Quantity(
        type=float,
        description='Minimum confidence for phase identification.',
        default=10.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    unknown_threshold = Quantity(
        type=float,
        description='Threshold for unknown phase identification.',
        default=0.2,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    show_reduced = Quantity(
        type=bool,
        description='Flag to show reduced patterns.',
        default=False,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    include_pdf = Quantity(
        type=bool,
        description='Flag to include PDFs in the analysis.',
        default=False,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    parallel = Quantity(
        type=bool,
        description='Flag to run the analysis in parallel.',
        default=False,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    raw = Quantity(
        type=bool,
        description='Flag to show raw data.',
        default=False,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    show_individual = Quantity(
        type=bool,
        description='Flag to shows individual prediction results: XRD and PDF.',
        default=False,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    wavelength = Quantity(
        type=float,
        unit='m',
        description='Wavelength of the X-ray tube source used for the measurement.',
        default=1.540598e-10,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    min_angle = Quantity(
        type=float,
        unit='deg',
        description="""
        Minimum 2-theta angle to be assumed for the analysis. Even if the measured
        spectra has a lower angle, the analysis will be performed on the spectra
        starting from this angle.
        """,
        default=10.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    max_angle = Quantity(
        type=float,
        unit='deg',
        description="""
        Maximum 2-theta angle to be assumed for the analysis. Even if the measured
        spectra has a higher angle, the analysis will be performed on the spectra up to
        this angle.
        """,
        default=80.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )


class IdentifiedPhase(ArchiveSection):
    """
    Section for the identified phase.
    """

    name = Quantity(
        type=str,
        description='The name of the identified phase in the XRD data.',
    )
    reference_structure = Quantity(
        type=ReferenceStructure,
        description='The reference structure of the identified phase in the training '
        'data.',
    )
    confidence = Quantity(
        type=float,
        description='The confidence that the phase is present, ranging from 0 to 100.',
    )


class AutoXRDAnalysisResult(ArchiveSection):
    """
    Section for the results of the auto XRD analysis.
    """

    name = Quantity(
        type=str,
        description='The name of the analysis result.',
    )
    identified_phases_plot = Quantity(
        type=str,
        description='Path to the plot showing the identified phases.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.FileEditQuantity,
        ),
        a_browser=BrowserAnnotation(adaptor='RawFileAdaptor'),
    )
    xrd_measurement = SubSection(
        section_def=SectionReference,
        description='The XRD pattern used for analysis.',
    )
    identified_phases = SubSection(
        section_def=IdentifiedPhase,
        repeats=True,
        description='The identified phases in the XRD data.',
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        if self.xrd_measurement and self.xrd_measurement.name:
            self.name = self.xrd_measurement.name


class AutoXRDTraining(JupyterAnalysis):
    """
    Schema for training an auto XRD model. Generates a Jupyter notebook containing
    helper code to train and index the model NOMAD.
    """

    m_def = Section(
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'name',
                    'datetime',
                    'lab_id',
                    'location',
                    'description',
                    'method',
                    'structure_files',
                    'notebook',
                    'trigger_generate_notebook',
                ],
                visible=Filter(
                    exclude=[
                        'inputs',
                        'query_for_inputs',
                        'steps',
                        'trigger_reset_inputs',
                    ],
                ),
            ),
        ),
    )
    description = Quantity(
        description='A description of the auto XRD model training.',
        a_eln=ELNAnnotation(
            component='RichTextEditQuantity',
            props=dict(height=500),
        ),
    )
    structure_files = Quantity(
        type=str,
        shape=['*'],
        description='Path to structure file (CIF) containing crystal structure.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.FileEditQuantity,
        ),
        a_browser=BrowserAnnotation(adaptor='RawFileAdaptor'),
    )
    trigger_generate_notebook = Quantity(
        default=True,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Generate Notebook',
        ),
    )
    outputs = SubSection(
        section_def=AutoXRDModelReference,
        repeats=True,
        description='An `AutoXRDModel` trained to predict phases in a given composition'
        'space.',
    )

    def write_predefined_cells(self, archive, logger):
        """
        Extends the `write_predefined_cells` method to add additional cells specific to
        the Auto XRD Training.
        """
        cells = super().write_predefined_cells(archive, logger)

        source = [
            '## Training Auto XRD Model\n',
            '\n',
            'For training the Auto XRD model, we need to simulate XRD patterns for\n',
            'different composition and phases covering an expected composition\n',
            'space. Once the training data is setup, we train a CNN model capable of\n',
            'phase identification from real XRD patterns.\n',
            '\n',
            'The workflow is managed by `nomad_auto_xrd.training` module which uses\n',
            'the [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer)\n',
            'package under the hood. `nomad_auto_xrd.training.train` takes\n',
            '`AutoXRDModel` NOMAD section as input. The section can be used to\n',
            'specify the settings for simulating XRD patterns and training the\n',
            'model.\n',
        ]
        cells.append(
            nbformat.v4.new_markdown_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            )
        )

        source = [
            'from nomad_auto_xrd.schema import (\n',
            '    AutoXRDModel,\n',
            '    SimulationSettings,\n',
            '    TrainingSettings,\n',
            ')\n',
            '\n',
            '# either specify or use the default settings\n',
            'training_settings = TrainingSettings(\n',
            '    num_epochs=2,\n',
            '    batch_size=32,\n',
            '    learning_rate=0.001,\n',
            '    seed=43,\n',
            ')\n',
            'simulation_settings = SimulationSettings()\n',
            'model = AutoXRDModel(\n',
            "    working_directory='.',\n",
            '    training_settings=training_settings,\n',
            '    simulation_settings=simulation_settings,\n',
            '    includes_pdf=True,\n',
            ')\n',
        ]
        cells.append(
            nbformat.v4.new_code_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            '## Training the Model\n',
            '\n',
            'Next, we connect the CIF files of the structures available in the\n',
            '`analysis` entry to be used as training data for the model.\n',
        ]
        cells.append(
            nbformat.v4.new_markdown_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            'import os\n',
            '\n',
            '# Specify the path to the input structures\n',
            'model.simulation_settings.structure_files = [\n',
            '    file_name\n',
            '    for file_name in analysis.structure_files\n',
            "    if file_name.endswith('.cif')\n",
            ']\n',
        ]
        cells.append(
            nbformat.v4.new_code_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            'Now, we import the training module and execute it for the model\n',
        ]
        cells.append(
            nbformat.v4.new_markdown_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            'from nomad_auto_xrd.training import train\n',
            '\n',
            'train(model)',
        ]
        cells.append(
            nbformat.v4.new_code_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            '## Saving the Model\n',
            '\n',
            'After completing the training, the `model.xrd_model` and\n',
            '`model.pdf_model`\n',
            'attributes contain the path to the trained model files. Additionally,\n',
            '`model.reference_files` provides a list of CIF files of the structures \n',
            'that the model can predict.\n',
            '\n',
            'Next, we will create an entry for this model in NOMAD, enabling its use\n',
            'for Auto XRD analysis. We will also link the model entry to the\n',
            '`analysis.outputs` for future reference.\n',
        ]
        cells.append(
            nbformat.v4.new_markdown_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            'from nomad_analysis.utils import create_entry_with_api\n',
            '\n',
            'file_name = (\n',
            "    os.path.basename(analysis.m_parent.metadata.mainfile).rsplit('.archive.', 1)[0]\n",  # noqa: E501
            "    + '_model.archive.json'\n",
            ')\n',
            "analysis.m_setdefault('outputs/0')\n",
            'analysis.outputs[0].reference = create_entry_with_api(\n',
            '    model,\n',
            '    base_url=analysis.m_context.installation_url,\n',
            '    upload_id=analysis.m_context.upload_id,\n',
            '    file_name=file_name,\n',
            ')\n',
        ]
        cells.append(
            nbformat.v4.new_code_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            '## Saving the Analysis Entry\n',
            '\n',
            'Finally, we save the analysis entry to update the changes in NOMAD.\n',
        ]
        cells.append(
            nbformat.v4.new_markdown_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            'analysis.save()\n',
        ]
        cells.append(
            nbformat.v4.new_code_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        return cells

    def normalize(self, archive, logger):
        """
        Normalizes the `AutoXRDAnalysis` entry.

        Args:
            archive (Archive): A NOMAD archive.
            logger (Logger): A structured logger.
        """
        self.method = 'Auto XRD Model Training'
        if self.description is None or self.description == '':
            self.description = """
            <p>
            This ELN comes with a Jupyter notebook that can be used to train an ML model
            for automatic phase identification from XRD data. The trained model can be
            indexed with `AutoXRDModel` entry which saves related metadata. </p> <p>

            To train the model, follow these steps:</p>
            <ol>
                <li>
                Upload the CIF files of the structures to be used for training in the
                <strong><em>structure_files</em></strong> quantity.
                </li>
                <li>
                From the <strong><em>notebook</em></strong> quantity, open the the
                Jupyter notebook and follow the steps mentioned in there to perform the
                training.
                </li>
            </ol>
            """
        super().normalize(archive, logger)


class AutoXRDAnalysis(JupyterAnalysis):
    """
    Schema for running an auto XRD analysis using an pre-trained Auto XRD model.
    Allows to attach Auto XRD model and XRD measurement entries and run the analysis
    using a pre-defined Jupyter notebook to identify the phases in the XRD data.
    """

    m_def = Section(
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'name',
                    'datetime',
                    'lab_id',
                    'location',
                    'description',
                    'method',
                    'query_for_inputs',
                    'notebook',
                    'trigger_generate_notebook',
                    'trigger_reset_inputs',
                    'analysis_settings',
                    'inputs',
                    'results',
                ],
                visible=Filter(
                    exclude=[
                        'steps',
                        'outputs',
                    ],
                ),
            ),
        ),
    )
    description = Quantity(
        type=str,
        description='A description of the auto XRD analysis.',
        a_eln=ELNAnnotation(
            component='RichTextEditQuantity',
            props=dict(height=500),
        ),
    )
    trigger_generate_notebook = Quantity(
        default=True,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.ActionEditQuantity,
            label='Generate Notebook',
        ),
    )
    analysis_settings = SubSection(
        section_def=AnalysisSettings,
        description='Settings for running the analysis.',
    )
    inputs = SubSection(
        section_def=AutoXRDMeasurementReference,
        repeats=True,
        description='A reference to an `XRayDiffraction` entry.',
    )
    results = SubSection(
        section_def=AutoXRDAnalysisResult,
        repeats=True,
        description='Results of the auto XRD analysis.',
    )

    def write_predefined_cells(self, archive, logger):
        cells = super().write_predefined_cells(archive, logger)

        source = [
            '## Running Auto XRD Analysis\n',
            '\n',
            'This workflow uses a pre-trained Auto XRD model to analyze multiple XRD ',
            'patterns simultaneously. The model and XRD measurements entries can be ',
            'connected in the analysis entry by creating references in the NOMAD GUI. ',
            'These will reflect under `analysis.analysis_settings.model` and ',
            '`analysis.inputs`.\n',
            '\n',
            'Once this is done, we can use the `nomad_auto_xrd.analysis.analyse` ',
            'routine to execute the analysis workflow on the `analysis` entry. This ',
            'will populate the `analysis.results`; one results sub-section is created ',
            'for each XRD measurement input.\n',
        ]
        cells.append(
            nbformat.v4.new_markdown_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            )
        )

        source = [
            'from nomad_auto_xrd.analysis import analyse\n',
            '\n',
            'analyse(analysis)\n',
            '\n',
            'analysis.results\n',
        ]
        cells.append(
            nbformat.v4.new_code_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            '## Saving the results\n',
            '\n',
            'Once the analysis results are generated, save the analysis entry to ',
            'update the results in NOMAD.\n',
        ]
        cells.append(
            nbformat.v4.new_markdown_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            'analysis.save()\n',
        ]
        cells.append(
            nbformat.v4.new_code_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        return cells

    def normalize(self, archive, logger):
        """
        Normalizes the `AutoXRDAnalysis` entry.

        Args:
            archive (Archive): A NOMAD archive.
            logger (Logger): A structured logger.
        """
        self.method = 'Auto XRD Analysis'
        if self.description is None or self.description == '':
            self.description = """
            <p>
            This ELN includes a Jupyter notebook designed to perform auto XRD analysis
            using a pre-trained ML model. Follow these steps to perform the
            analysis:</p>
            <ol>
                <li>
                Initialize the <strong><em>analysis_settings</em></strong> section and
                ensure that the
                <strong><em>analysis_settings.auto_xrd_model</em></strong>
                quantity references an <strong><em>AutoXRDModel</em></strong> entry.
                The selected model should be compatible with the sample's composition
                space.
                </li>
                <li>
                Review and adjust the default analysis settings in the
                <strong><em>analysis_settings</em></strong> section if necessary to
                match the requirements of your analysis.
                </li>
                <li>
                Use the <strong><em>analysis.inputs</em></strong> section to add the XRD
                measurement entries for which the analysis is to be performed.
                </li>
                <li>
                Open the Jupyter notebook from the <strong><em>notebook</em></strong>
                quantity and follow the provided instructions to execute the analysis.
                </li>
            </ol>
            """
        super().normalize(archive, logger)

        # validate the data in the referenced XRD entries
        if self.analysis_settings:
            for xrd_reference in self.inputs:
                if not xrd_reference.reference:
                    continue
                xrd = xrd_reference.reference
                if isinstance(xrd, MProxy):
                    xrd.m_proxy_resolve()
                if not isinstance(xrd, XRayDiffraction):
                    logger.error(
                        f'XRD entry "{xrd.name}" is not of type `XRayDiffraction`.'
                    )
                    continue
                try:
                    pattern = (
                        xrd.m_parent.results.properties.structural.diffraction_pattern[
                            0
                        ]
                    )
                    two_theta = pattern.two_theta_angles
                    intensity = pattern.intensity
                except AttributeError as e:
                    logger.error(f'Error accessing XRD entry "{xrd.name}". {e}')
                    continue
                if two_theta is None or intensity is None:
                    logger.error(
                        f'XRD entry {xrd.name} does not contain valid two theta '
                        'angles or intensity data.'
                    )
                    continue
                elif (
                    min(two_theta) > self.analysis_settings.min_angle
                    or max(two_theta) < self.analysis_settings.max_angle
                ):
                    logger.error(
                        f'Two theta range of XRD entry "{xrd.name}" [{min(two_theta)}, '
                        f'{max(two_theta)}] should be a super set of two theta range '
                        'specified in the analysis settings '
                        f'[{self.analysis_settings.min_angle}, '
                        f'{self.analysis_settings.max_angle}].'
                    )
                    continue


m_package.__init_metainfo__()
