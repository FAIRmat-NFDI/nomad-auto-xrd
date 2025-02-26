from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    Filter,
    SectionProperties,
)
from nomad.datamodel.metainfo.basesections import SectionReference
from nomad.metainfo import (
    Quantity,
    SchemaPackage,
    Section,
    SubSection,
)
from nomad_analysis.general.schema import AnalysisResult
from nomad_analysis.jupyter.schema import ELNJupyterAnalysis
from nomad_measurements.xrd.schema import ELNXRayDiffraction

from nomad_auto_xrd.schema_packages.auto_xrd import AutoXRDModel

m_package = SchemaPackage()


class AutoXRDAnalysisInput(SectionReference):
    """
    Base class for all `AutoXRDAnalysis` inputs.
    """


class XRDMeasurement(AutoXRDAnalysisInput):
    reference = Quantity(
        type=ELNXRayDiffraction,
        description='A reference to an `ELNXRayDiffraction` entry.',
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )


class AutoXRDModelReference(AutoXRDAnalysisInput):
    reference = Quantity(
        type=AutoXRDModel,
        description='A reference to an `AutoXRDModel` entry.',
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )


class IdentifiedPhase(AnalysisResult):
    """
    Section for the identified phase.
    """

    phase = Quantity(
        type=str,
        description='The identified phase in the XRD data.',
    )
    reference_cif = Quantity(
        type=str,
        description='The reference CIF file.',
        a_eln=ELNAnnotation(
            component='FileEditQuantity',
        ),
    )
    probability = Quantity(
        type=float,
        description='The probability that the phase is present.',
    )


class AutoXRDAnalysis(ELNJupyterAnalysis):
    """
    Schema for running an auto XRD analysis using an pre-trained ML model.
    """

    m_def = Section(
        a_eln=ELNAnnotation(
            properties=SectionProperties(
                visible=Filter(
                    exclude=['input_entry_class', 'query_for_inputs'],
                ),
                order=[
                    'name',
                    'datetime',
                    'lab_id',
                    'location',
                    'notebook',
                    'reset_notebook',
                    'description',
                    'analysis_type',
                ],
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
    analysis_type = Quantity(
        type=str,
        default='Auto XRD',
        description=(
            'Based on the analysis type, code cells will be added to the Jupyter '
            'notebook. Code cells from **Generic** are always included.'
            """
            | Analysis Type       | Description                                     |
            |---------------------|-------------------------------------------------|
            | **Generic**         | Basic setup including connection \
                                    with entry data.                                |
            | **XRD**             | Adds XRD related analysis functions.            |
            | **Auto XRD**        | (Default) Analysis XRD patterns using machine \
                                    learning.                                       |
            """
        ),
    )
    inputs = SubSection(
        section_def=AutoXRDAnalysisInput,
        description='The input section for the auto XRD analysis.',
        repeats=True,
    )
    outputs = SubSection(
        section_def=IdentifiedPhase,
        description='The phases identified by the auto XRD analysis.',
        repeats=True,
    )

    def normalize(self, archive, logger):
        """
        Normalizes the `AutoXRDAnalysis` entry.

        Args:
            archive (Archive): A NOMAD archive.
            logger (Logger): A structured logger.
        """
        super().normalize(archive, logger)
        if self.description is None or self.description == '':
            self.description = """
            <p>
            This ELN comes with a Jupyter notebook that can be used to run an auto
            XRD analysis using a pre-trained ML model. To get started, do the
            following:</p> <p>

            1. In the <strong><em>inputs</em></strong> sub-section, use the
            <strong><em>AutoXRDModelReference</em></strong> section to reference an
            <strong><em>AutoXRDModel</em></strong> entry containing the pre-trained
            model.</p> <p>

            2. In the <strong><em>inputs</em></strong> sub-section, use the
            <strong><em>XRDMeasurement</em></strong> section to reference an
            <strong><em>ELNXRayDiffraction</em></strong> containing the XRD data
            you want to analyse.</p> <p>

            3. From the <strong><em>notebook</em></strong> quantity, open the the
            Jupyter notebook and follow the steps mentioned in there to perform the
            analysis.</p>
            """


m_package.__init_metainfo__()
