from nomad.datamodel.metainfo.annotations import ELNAnnotation
from nomad.datamodel.metainfo.basesections import SectionReference
from nomad.metainfo import Quantity, SubSection
from nomad_analysis.general.schema import AnalysisResult
from nomad_analysis.jupyter.schema import ELNJupyterAnalysis
from nomad_measurements.xrd.schema import ELNXRayDiffraction

from nomad_auto_xrd.schema_packages.auto_xrd import AutoXRDModel


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
        description='The identified phase.',
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
