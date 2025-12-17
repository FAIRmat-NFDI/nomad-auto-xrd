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
from ase.io import read as ase_io_read
from matid import SymmetryAnalyzer
from nomad.actions import manager
from nomad.datamodel import ArchiveSection
from nomad.datamodel.data import EntryDataCategory, Schema
from nomad.datamodel.metainfo.annotations import (
    BrowserAnnotation,
    ELNAnnotation,
    ELNComponentEnum,
    Filter,
    SectionProperties,
)
from nomad.datamodel.metainfo.basesections import Analysis, Entity, SectionReference
from nomad.datamodel.metainfo.plot import PlotlyFigure, PlotSection
from nomad.datamodel.results import DiffractionPattern, Material, SymmetryNew, System
from nomad.metainfo import (
    Category,
    MProxy,
    Quantity,
    SchemaPackage,
    Section,
    SubSection,
)
from nomad.normalizing.common import nomad_atoms_from_ase_atoms
from nomad.normalizing.topology import add_system, add_system_info
from nomad_analysis.actions.schema import Action
from nomad_analysis.jupyter.schema import JupyterAnalysis
from nomad_measurements.mapping.schema import MappingResult
from nomad_measurements.xrd.schema import XRayDiffraction, XRDResult, XRDResult1D
from pymatgen.io.cif import CifParser

from nomad_auto_xrd.actions.analysis.models import UserInput as AnalysisUserInput
from nomad_auto_xrd.actions.training.models import UserInput as TrainingUserInput
from nomad_auto_xrd.common.models import (
    AnalysisSettingsInput,
    AutoXRDModelInput,
    PatternAnalysisResult,
    Phase,
    PhasesPosition,
    SimulationSettingsInput,
    TrainingSettingsInput,
    XRDMeasurementEntry,
)
from nomad_auto_xrd.common.utils import (
    plot_identified_phases,
    plot_identified_phases_sample_position,
)

if TYPE_CHECKING:
    from nomad.datamodel.context import Context
    from structlog.stdlib import BoundLogger


def populate_material_topology_with_cifs(
    cif_files: list[str], context: 'Context'
) -> Material:
    """
    Returns a `nomad.datamodel.results.Material` section with elements and topology
    information from a list of CIF files.

    Args:
        cif_files (list[str]): A list of CIF file paths.
        context (Context): The NOMAD upload context. Required to read raw files.

    Returns:
        Material: A populated `nomad.datamodel.results.Material` section.
    """
    material = Material()

    # Read the cif files and convert them into ase atoms
    ase_atoms_list = []
    for cif_file in cif_files:
        if not cif_file or not cif_file.endswith('.cif'):
            raise ValueError(
                f'Cannot parse structure file: {cif_file}. Should be a "*.cif" file.'
            )
        with context.raw_file(cif_file) as file:
            ase_atoms_list.append(ase_io_read(file.name))

    # populate elements from a set of all the elements in ase_atoms
    elements = set()
    for ase_atoms in ase_atoms_list:
        elements.update(ase_atoms.get_chemical_symbols())
    material.elements = list(elements)

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
        label = f'{ase_atoms.get_chemical_formula()}-{symmetry.space_group_number}'
        labels.append(label)
        symmetry.crystal_system = symmetry_analyzer.get_crystal_system()
        symmetry.point_group = symmetry_analyzer.get_point_group()
        system = System(
            atoms=nomad_atoms_from_ase_atoms(ase_atoms),
            label=label,
            description='Structure generated from a CIF file.',
            structural_type='bulk',
            dimensionality='3D',
            symmetry=symmetry,
        )
        add_system_info(system, topology)
        add_system(system, topology)

    material.topology = list(topology.values())

    return material


m_package = SchemaPackage(aliases=['nomad_auto_xrd.schema'])


class Model(Entity, Schema):
    """
    Schema for a generic machine learning model. If the saved model file is from PyTorch
    or TensorFlow, the model can be loaded using the `load_model` method. These model
    file also normalized to populate the current model section.
    """


class AutoXRDCategory(EntryDataCategory):
    """
    Category for Auto XRD analysis, training, and model entries.
    """

    m_def = Category(
        label='Auto XRD Schemas',
        categories=[EntryDataCategory],
    )


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
        type=float,
        description='Maximum texture value for the simualtions.',
        default=0.5,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    min_domain_size = Quantity(
        type=float,
        description='Minimum domain size.',
        unit='nm',
        default=5.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    max_domain_size = Quantity(
        type=float,
        description='Maximum domain size.',
        unit='nm',
        default=30.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    max_strain = Quantity(
        type=float,
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
        type=float,
        description='Minimum angle value.',
        unit='deg',
        default=10.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    max_angle = Quantity(
        type=float,
        description='Maximum angle value.',
        unit='deg',
        default=80.0,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    max_shift = Quantity(
        type=float,
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
        type=float,
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
        type=float,
        description='Learning rate for training.',
        default=0.001,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    seed = Quantity(
        type=int,
        description='Seed for random number generator.',
        default=34,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    test_fraction = Quantity(
        type=float,
        description='Fraction of data used for testing.',
        default=0.2,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
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
        label='Auto XRD Model',
        description="""
        Based on the structure files (CIF files) added, XRD patterns are simulated
        for different phase compositions and structures. The simulated XRD patterns are
        then used to train a machine learning model to predict the phase composition
        and structure from the XRD data.""",
        categories=[AutoXRDCategory],
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'name',
                    'datetime',
                    'description',
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
                visible=Filter(
                    exclude=[
                        'lab_id',
                        'location',
                    ],
                ),
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
        description='Path to the trained PDF model file.',
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
            cif_files = [
                reference_structure.cif_file
                for reference_structure in self.reference_structures
            ]
            try:
                archive.m_setdefault('results/material')
                archive.results.material = populate_material_topology_with_cifs(
                    cif_files, archive.m_context
                )
                for i in range(len(archive.results.material.topology)):
                    self.reference_structures[
                        i
                    ].system = f'#/results/material/topology/{i}'
            except Exception:
                logger.error(
                    'Error in populating material topology from CIF files.',
                    exc_info=True,
                )


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

    m_def = Section(
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'auto_xrd_model',
                    'min_angle',
                    'max_angle',
                    'wavelength',
                    'max_phases',
                    'min_confidence',
                    'cutoff_intensity',
                    'include_pdf',
                    'parallel',
                    'simulated_reference_patterns',
                ]
            )
        )
    )
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
        default=3,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.NumberEditQuantity,
        ),
    )
    cutoff_intensity = Quantity(
        type=float,
        description='Intensity threshold (% of original maximum) below which phase '
        'identification stops, assuming remaining signal is noise.',
        default=10.0,
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
    include_pdf = Quantity(
        type=bool,
        description='Whether to include the PDF based model in the analysis.',
        default=True,
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.BoolEditQuantity,
        ),
    )
    parallel = Quantity(
        type=bool,
        description='Whether to run the analysis using a parallel processing pool.',
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
    simulated_reference_patterns = SubSection(
        section_def=XRDResult1D,
        repeats=True,
        description='The simulated XRD patterns for the reference phases predicted by'
        ' the model under the given analysis settings.',
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


class AutoXRDAnalysisResult(PlotSection):
    """
    Section for the results of the auto XRD analysis of a single pattern.
    """

    name = Quantity(
        type=str,
        description='The name of the analysis result.',
    )

    def generate_plots(self, logger: 'BoundLogger', **kwargs) -> list[PlotlyFigure]:
        """
        Creates plots for the analysis results.

        Returns:
            list[PlotlyFigure]: A list of Plotly figures for the analysis results.
        """
        logger.warning(
            '`generate_plots` method not implemented for `AutoXRDAnalysisResult` class.'
        )
        return []


class SinglePatternAnalysisResult(AutoXRDAnalysisResult):
    """
    Section for the results of the auto XRD analysis of a single pattern.
    """

    xrd_results = Quantity(
        type=XRDResult,
        description='The XRD measurement results used for analysis.',
    )
    identified_phases_plot = Quantity(
        type=str,
        description='Path to the plot showing the identified phases.',
        a_eln=ELNAnnotation(
            component=ELNComponentEnum.FileEditQuantity,
        ),
        a_browser=BrowserAnnotation(adaptor='RawFileAdaptor'),
    )
    identified_phases = SubSection(
        section_def=IdentifiedPhase,
        repeats=True,
        description='The identified phases in the XRD data.',
    )

    def generate_plots(
        self,
        logger: 'BoundLogger',
        **kwargs,
    ) -> list[PlotlyFigure]:
        """
        Creates plots for the analysis results, specifically the XRD pattern with
        identified phases.

        Args:
            logger (BoundLogger): The logger to use for logging messages.
        Kwargs:
            measured_pattern (DiffractionPattern): The measured XRD pattern.
            reference_phase_simulated_patterns (list[XRDResult1D]): The simulated XRD
                patterns for the reference phases.

        Returns:
            list[PlotlyFigure]: A list of Plotly figures for the analysis results.
        """
        figures = []
        measured_pattern: DiffractionPattern = kwargs.get('measured_pattern')
        reference_phase_simulated_patterns: list[XRDResult1D] = kwargs.get(
            'reference_phase_simulated_patterns', []
        )
        if not measured_pattern or not reference_phase_simulated_patterns:
            logger.warning(
                '`measured_pattern` or `reference_phase_simulated_patterns` not '
                'provided as kwargs to generate_plots method. Skipping plot generation.'
            )
            return figures
        if self.identified_phases:
            pattern_analysis_result = PatternAnalysisResult(
                two_theta=measured_pattern.two_theta_angles.to('deg').magnitude,
                intensity=measured_pattern.intensity,
                phases=[
                    Phase(
                        name=phase.name,
                        confidence=phase.confidence,
                        simulated_two_theta=next(
                            (
                                pattern.two_theta.to('deg').magnitude
                                for pattern in reference_phase_simulated_patterns
                                if pattern.name == phase.name
                            ),
                            None,
                        ),
                        simulated_intensity=next(
                            (
                                pattern.intensity.magnitude
                                for pattern in reference_phase_simulated_patterns
                                if pattern.name == phase.name
                            ),
                            None,
                        ),
                    )
                    for phase in self.identified_phases
                ],
            )
            for phase in pattern_analysis_result.phases:
                if (
                    phase.simulated_two_theta is None
                    or phase.simulated_intensity is None
                ):
                    logger.warning(
                        f'Simulated pattern not found for phase: {phase.name}. Unable '
                        'to generate the plot for identified phases.'
                    )
                    return figures
            plotly_json = plot_identified_phases(pattern_analysis_result)
            plotly_json['config'] = {'scrollZoom': False}
            figures.append(
                PlotlyFigure(
                    label='Identified phases in the XRD pattern',
                    index=0,
                    figure=plotly_json,
                )
            )
        return figures

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        if self.xrd_results and self.xrd_results.name:
            self.name = self.xrd_results.name


class MultiPatternAnalysisResult(AutoXRDAnalysisResult):
    """
    Section for the results of the auto XRD analysis of multiple patterns. For example,
    analysis of XRD patterns measured at different sample positions for a combinatorial
    library.
    """

    xrd_measurement = Quantity(
        type=XRayDiffraction,
        description='The XRD measurement used for analysis.',
    )
    single_pattern_results = SubSection(
        section_def=SinglePatternAnalysisResult,
        repeats=True,
        description='The results of the analysis for each XRD pattern.',
    )

    def generate_plots(self, logger: 'BoundLogger', **kwargs) -> list[PlotlyFigure]:
        """
        Creates plots for the analysis results, specifically a scatter plot showing
        the identified phases at their respective sample positions.

        Returns:
            list[PlotlyFigure]: A list of Plotly figures for the analysis results.
        """
        figures = []
        phases_position_list = []
        for result in self.single_pattern_results:
            if not result.identified_phases:
                continue
            try:
                assert isinstance(result.xrd_results, MappingResult)
                x_pos = result.xrd_results.x_absolute.to('millimeter').magnitude
                y_pos = result.xrd_results.y_absolute.to('millimeter').magnitude
                phases_position_list.append(
                    PhasesPosition(
                        x_position=x_pos,
                        y_position=y_pos,
                        x_unit='mm',
                        y_unit='mm',
                        phases=[
                            Phase(
                                name=phase.name,
                                confidence=phase.confidence,
                            )
                            for phase in result.identified_phases
                        ],
                    )
                )
            except Exception:
                logger.warning(
                    'Error in extracting sample position from measurement reference.',
                    exc_info=True,
                )
        if phases_position_list:
            plotly_json = plot_identified_phases_sample_position(phases_position_list)
            plotly_json['config'] = {'scrollZoom': False}
            figures.append(
                PlotlyFigure(
                    label='Primary identified phases for the Combinatorial library',
                    index=0,
                    figure=plotly_json,
                )
            )
        return figures

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
        label='Auto XRD Training',
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
            'package under the hood. `nomad_auto_xrd.training.train_nomad_model` '
            'takes\n',
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
            'from nomad_auto_xrd.schema_packages.schema import (\n',
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
            'Here, we ensure that the added CIFs are parsable by `pymatgen`.\n',
        ]
        cells.append(
            nbformat.v4.new_markdown_cell(
                source=source,
                metadata={'tags': ['nomad-analysis-predefined']},
            ),
        )

        source = [
            'import os\n',
            'from pymatgen.io.cif import CifParser\n',
            '\n',
            '# Specify the path to the input structures\n',
            'model.simulation_settings.structure_files = []\n',
            'for cif in analysis.structure_files:\n',
            '    parser = CifParser(cif)\n',
            '    try:\n',
            '        parser.get_structures()\n',
            '        model.simulation_settings.structure_files.append(cif)\n',
            '    except Exception as e:\n',
            '        print(f\'Error in {cif}: "{e}". Not using it for training.\')\n',
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
            'from nomad_auto_xrd.common.training import train_nomad_model\n',
            '\n',
            'train_nomad_model(model)',
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
        label='Auto XRD Analysis',
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
            'Once this is done, we can use the '
            '`nomad_auto_xrd.common.analysis.analyze` ',
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
            'from nomad_auto_xrd.common.analysis import analyze\n',
            '\n',
            'analyze(analysis)\n',
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

    def validate_inputs(self):
        """
        1. Ensures that the referenced entries are of type `XRayDiffraction`.
        2. Ensures that the two theta range of the XRD pattern is a super set of the
           two theta range specified in the analysis settings.

        Raises errors when the data in the referenced XRD entries is not valid. Updates
        the two theta range in analysis settings when required.
        """
        for xrd_reference in self.inputs:
            if not xrd_reference.reference:
                continue
            xrd = xrd_reference.reference
            if isinstance(xrd, MProxy):
                xrd.m_proxy_resolve()
            if not isinstance(xrd, XRayDiffraction):
                raise TypeError(
                    f'XRD entry "{xrd.name}" is not of type `XRayDiffraction`.'
                )
            pattern = xrd.m_parent.results.properties.structural.diffraction_pattern[0]
            two_theta = pattern.two_theta_angles
            intensity = pattern.intensity
            if two_theta is None or intensity is None:
                raise ValueError(
                    f'XRD entry "{xrd.name}" does not contain '
                    'valid two theta angles or intensity data.'
                )
            new_min = max(min(two_theta), self.analysis_settings.min_angle)
            new_max = min(max(two_theta), self.analysis_settings.max_angle)
            if new_min < new_max:
                self.analysis_settings.min_angle = new_min
                self.analysis_settings.max_angle = new_max
            else:
                raise ValueError(
                    'A valid two theta range for analysis settings could not be '
                    'determined for the given set of inputs. '
                    'The range in analysis setting should be a '
                    'sub-set of the two theta range of all the input measurements.'
                )

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
        self.m_setdefault('analysis_settings')
        try:
            self.validate_inputs()
        except Exception as e:
            logger.error(str(e))
        if (
            self.analysis_settings.min_angle.magnitude
            != AnalysisSettings.min_angle.default
            or self.analysis_settings.max_angle.magnitude
            != AnalysisSettings.max_angle.default
        ):
            logger.info(
                f'Based on the inputs, adjusted the two theta range '
                f'for analysis to [{self.analysis_settings.min_angle}, '
                f'{self.analysis_settings.max_angle}].'
            )

        super().normalize(archive, logger)


class AutoXRDTrainingAction(Action, Analysis, Schema):
    """
    Schema that uses actions to train an auto XRD model using specified simulation
    and training settings.
    """

    m_def = Section(
        label='Auto XRD Training Action',
        categories=[AutoXRDCategory],
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'name',
                    'datetime',
                    'trained_model_name',
                    'description',
                    'method',
                    'trigger_start_action',
                    'trigger_get_action_status',
                    'simulation_settings',
                    'training_settings',
                    'outputs',
                ],
                visible=Filter(
                    exclude=[
                        'location',
                        'inputs',
                        'steps',
                    ],
                ),
            ),
        ),
    )
    trained_model_name = Quantity(
        type=str,
        description='Name of the entry created for the trained Auto-XRD model. If '
        'not given, a default model name based on the composition space will be used.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )
    simulation_settings = SubSection(
        section_def=SimulationSettings,
        description='Settings for simulating XRD patterns.',
    )
    training_settings = SubSection(
        section_def=TrainingSettings,
        description='Settings for training the model.',
    )
    outputs = SubSection(
        section_def=AutoXRDModelReference,
        repeats=True,
        description='An `AutoXRDModel` trained to predict phases in a given composition'
        'space.',
    )

    def start_action(self, archive, logger) -> str:
        input_data = TrainingUserInput(
            upload_id=archive.metadata.upload_id,
            user_id=archive.metadata.authors[0].user_id,
            mainfile=archive.metadata.mainfile,
            trained_model_name=self.trained_model_name,
            simulation_settings=SimulationSettingsInput(
                structure_files=self.simulation_settings.structure_files,
                max_texture=float(self.simulation_settings.max_texture),
                min_domain_size=float(
                    self.simulation_settings.min_domain_size.magnitude
                ),
                max_domain_size=float(
                    self.simulation_settings.max_domain_size.magnitude
                ),
                max_strain=float(self.simulation_settings.max_strain),
                num_patterns=int(self.simulation_settings.num_patterns),
                min_angle=float(self.simulation_settings.min_angle.magnitude),
                max_angle=float(self.simulation_settings.max_angle.magnitude),
                max_shift=float(self.simulation_settings.max_shift.magnitude),
                separate=self.simulation_settings.separate,
                impur_amt=float(self.simulation_settings.impur_amt),
                skip_filter=self.simulation_settings.skip_filter,
                include_elems=self.simulation_settings.include_elems,
            ),
            training_settings=TrainingSettingsInput(
                num_epochs=int(self.training_settings.num_epochs),
                batch_size=int(self.training_settings.batch_size),
                learning_rate=float(self.training_settings.learning_rate),
                seed=int(self.training_settings.seed),
                test_fraction=float(self.training_settings.test_fraction),
            ),
        )
        action_instance_id = manager.start_action(
            'nomad_auto_xrd.actions.training:training_action', data=input_data
        )
        return action_instance_id

    def normalize(self, archive, logger):
        """
        Normalizes the `AutoXRDAnalysis` entry.

        Args:
            archive (Archive): A NOMAD archive.
            logger (Logger): A structured logger.
        """
        self.method = 'Auto XRD Model Training'

        if (
            not self.simulation_settings
            or not self.simulation_settings.structure_files
            or not self.training_settings
        ):
            self.trigger_start_action = False
            logger.warning(
                'Either simulation_settings or simulation_setting.structure_files '
                'or training_settings not set. These are requireed for running the '
                'training action.'
            )
        if self.simulation_settings and self.simulation_settings.structure_files:
            elements = set()
            for cif in self.simulation_settings.structure_files:
                try:
                    with archive.m_context.raw_file(cif) as file:
                        parser = CifParser(file.name)
                    structures = parser.get_structures()
                    elements.update(structures[0].chemical_system_set)
                except Exception:
                    self.trigger_start_action = False
                    logger.error(
                        f'Error in parsing {cif}. Cannot run the training.',
                        exec_info=True,
                    )
                    break
            elements_list = sorted(list(elements))
            if not self.trained_model_name:
                self.trained_model_name = '-'.join(elements_list) + ' Auto XRD Model'
        if self.trigger_start_action:
            if self.action_status == 'RUNNING':
                # if the updated status is still RUNNING, do not trigger a new run
                self.trigger_start_action = False
                logger.warning(
                    'The training action is already running. Please wait for it to '
                    'complete before running the training again.'
                )

        super().normalize(archive, logger)


class AutoXRDAnalysisAction(Action, Analysis, Schema):
    """
    Schema that uses actions to run an auto XRD analysis using a pre-trained Auto XRD
    model.
    """

    m_def = Section(
        label='Auto XRD Analysis Action',
        categories=[AutoXRDCategory],
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                order=[
                    'name',
                    'datetime',
                    'lab_id',
                    'description',
                    'method',
                    'trigger_start_action',
                    'trigger_get_action_status',
                    'analysis_settings',
                    'inputs',
                    'results',
                ],
                visible=Filter(
                    exclude=[
                        'location',
                        'outputs',
                        'steps',
                    ],
                ),
            ),
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

    def start_action(self, archive, logger) -> str:
        # reset results
        self.results = []
        self.analysis_settings.simulated_reference_patterns = []
        if archive.results and archive.results.material:
            archive.results.material = None

        xrd_measurement_entries = []
        for input_ref_section in self.inputs:
            xrd_measurement_entry = XRDMeasurementEntry(
                entry_id=input_ref_section.reference.m_parent.metadata.entry_id,
                upload_id=input_ref_section.reference.m_parent.metadata.upload_id,
            )
            xrd_measurement_entries.append(xrd_measurement_entry)
        model_entry = self.analysis_settings.auto_xrd_model
        model_input = AutoXRDModelInput(
            upload_id=model_entry.m_parent.metadata.upload_id,
            entry_id=model_entry.m_parent.metadata.entry_id,
            working_directory=model_entry.working_directory,
            includes_pdf=model_entry.includes_pdf,
            reference_structure_paths=[
                section.cif_file for section in model_entry.reference_structures
            ],
            xrd_model_path=model_entry.xrd_model,
            pdf_model_path=model_entry.pdf_model,
        )
        input_data = AnalysisUserInput(
            upload_id=archive.metadata.upload_id,
            user_id=archive.metadata.authors[0].user_id,
            mainfile=archive.metadata.mainfile,
            analysis_settings=AnalysisSettingsInput(
                auto_xrd_model=model_input,
                max_phases=self.analysis_settings.max_phases,
                cutoff_intensity=self.analysis_settings.cutoff_intensity,
                min_confidence=self.analysis_settings.min_confidence,
                include_pdf=self.analysis_settings.include_pdf,
                parallel=self.analysis_settings.parallel,
                wavelength=self.analysis_settings.wavelength.to('angstrom').magnitude,
                min_angle=self.analysis_settings.min_angle.to('degree').magnitude,
                max_angle=self.analysis_settings.max_angle.to('degree').magnitude,
            ),
            xrd_measurement_entries=xrd_measurement_entries,
        )
        action_instance_id = manager.start_action(
            'nomad_auto_xrd.actions.analysis:analysis_action', data=input_data
        )
        return action_instance_id

    def validate_inputs(self):
        """
        1. Ensures that the referenced entries are of type `XRayDiffraction`.
        2. Ensures that the two theta range of the XRD pattern is a super set of the
           two theta range specified in the analysis settings.

        Raises errors when the data in the referenced XRD entries is not valid. Updates
        the two theta range in analysis settings when required.
        """
        for xrd_reference in self.inputs:
            if not xrd_reference.reference:
                continue
            xrd = xrd_reference.reference
            if isinstance(xrd, MProxy):
                xrd.m_proxy_resolve()
            if not isinstance(xrd, XRayDiffraction):
                raise TypeError(
                    f'XRD entry "{xrd.name}" is not of type `XRayDiffraction`.'
                )
            pattern = xrd.m_parent.results.properties.structural.diffraction_pattern[0]
            two_theta = pattern.two_theta_angles
            intensity = pattern.intensity
            if two_theta is None or intensity is None:
                raise ValueError(
                    f'XRD entry "{xrd.name}" does not contain '
                    'valid two theta angles or intensity data.'
                )
            self.analysis_settings.min_angle = max(
                min(two_theta), self.analysis_settings.min_angle
            )
            self.analysis_settings.max_angle = min(
                max(two_theta), self.analysis_settings.max_angle
            )

    def populate_material_topology(self, archive, logger):
        """
        Populates the `archive.results.material.topology` of the analysis entry based
        on the identified phases in analysis results.

        Args:
            archive (Archive): A NOMAD archive.
            logger (Logger): A structured logger.
        """
        cif_files_set = set()
        for result in self.results:
            if isinstance(result, SinglePatternAnalysisResult):
                for phase in result.identified_phases:
                    cif_files_set.add(phase.reference_structure.cif_file)
            if isinstance(result, MultiPatternAnalysisResult):
                for pattern_result in result.single_pattern_results:
                    for phase in pattern_result.identified_phases:
                        cif_files_set.add(phase.reference_structure.cif_file)

        cif_files = list(cif_files_set)
        try:
            if not self.analysis_settings or not self.analysis_settings.auto_xrd_model:
                return
            model_context = self.analysis_settings.auto_xrd_model.m_context
            archive.m_setdefault('results/material')
            archive.results.material = populate_material_topology_with_cifs(
                cif_files, model_context
            )
        except Exception:
            logger.error(
                'Failed to populate material topology.',
                exc_info=True,
            )

    def generate_plots(self, logger):
        """
        Generates plots for the analysis results.
        """
        if not self.results:
            return
        try:
            for i, result in enumerate(self.results):
                if isinstance(result, SinglePatternAnalysisResult):
                    result.figures = result.generate_plots(
                        logger,
                        measured_pattern=(
                            result.xrd_results.m_parent.m_parent.results.properties.structural.diffraction_pattern[
                                0
                            ]
                        ),
                        reference_phase_simulated_patterns=(
                            self.analysis_settings.simulated_reference_patterns
                        ),
                    )
                if isinstance(result, MultiPatternAnalysisResult):
                    for j, pattern_result in enumerate(result.single_pattern_results):
                        pattern_result.figures = pattern_result.generate_plots(
                            logger,
                            measured_pattern=(
                                pattern_result.xrd_results.m_parent.m_parent.results.properties.structural.diffraction_pattern[
                                    j
                                ]
                            ),
                            reference_phase_simulated_patterns=(
                                self.analysis_settings.simulated_reference_patterns
                            ),
                        )
                    result.figures = result.generate_plots(logger)
        except Exception:
            logger.error(
                'Failed to generate plots for the analysis results.', exc_info=True
            )

    def normalize(self, archive, logger):
        """
        Normalizes the `AutoXRDAnalysis` entry.

        Args:
            archive (Archive): A NOMAD archive.
            logger (Logger): A structured logger.
        """
        self.method = 'Auto XRD Analysis'
        self.m_setdefault('analysis_settings')

        input_validation_failed = False
        try:
            self.validate_inputs()
        except Exception:
            input_validation_failed = True
            logger.error('Could not validate inputs.', exc_info=True)
        if (
            self.analysis_settings.min_angle.magnitude
            != AnalysisSettings.min_angle.default
            or self.analysis_settings.max_angle.magnitude
            != AnalysisSettings.max_angle.default
        ):
            logger.info(
                f'Based on the inputs, adjusted the two theta range '
                f'for analysis to [{self.analysis_settings.min_angle}, '
                f'{self.analysis_settings.max_angle}].'
            )

        if self.trigger_start_action:
            if not self.analysis_settings or not self.analysis_settings.auto_xrd_model:
                self.trigger_start_action = False
                logger.error(
                    'analysis_settings or analysis_settings.auto_xrd_model '
                    'is not set. Cannot run the analysis action.'
                )
            elif not self.inputs:
                self.trigger_start_action = False
                logger.error(
                    'No XRD measurements are provided for analysis. Cannot run the '
                    'analysis action.'
                )
            elif input_validation_failed:
                self.trigger_start_action = False
                logger.error(
                    'Validation of the XRD entries failed. Cannot run the analysis '
                    'action.'
                )

        self.populate_material_topology(archive, logger)
        self.generate_plots(logger)
        super().normalize(archive, logger)


m_package.__init_metainfo__()
