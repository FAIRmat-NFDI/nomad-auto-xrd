from ase.data import chemical_symbols
from nomad_analysis.jupyter.schema import ELNJupyterAnalysis
from nomad.datamodel import ArchiveSection
from nomad.datamodel.metainfo.annotations import ELNAnnotation, MEnum
from nomad.datamodel.metainfo.basesections import SectionReference, ElementalComposition
from nomad.metainfo import Quantity, SubSection


class AutoXRDTrainingInput(ArchiveSection):
    """
    Base class for all `AutoXRDAnalysis` inputs.
    """


class CompositionSpace(AutoXRDTrainingInput):
    name = Quantity(
        type=str,
        description='A descriptor for the composition space.',
        a_eln=dict(component='StringEditQuantity'),
    )
    element_composition = SubSection(
        type=ElementalComposition,
        description='The elemental composition of the composition space.',
        repeats=True,
    )

    def normalize(self, archive, logger):
        super().normalize(archive, logger)
        if self.element_composition is not None:
            # construct a short description of the composition space
            elements = [c.element for c in self.element_composition]
            elements = sorted(elements)
            self.name = '-'.join(elements)


class CIFFile(AutoXRDTrainingInput):
    name = Quantity(
        type=str,
        description='The name of the phase described in the CIF file.',
        a_eln=dict(component='StringEditQuantity'),
    )
    cif_file = Quantity(
        type=str,
        description='The CIF file for a given phase.',
        a_eln=dict(component='FileEditQuantity'),
    )


class AutoXRDModelReference(SectionReference):
    reference = Quantity(
        type=SectionReference,
        description='A reference to an `AutoXRDModel` entry.',
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        ),
    )


class AutoXRDTraining(ELNJupyterAnalysis):
    inputs = SubSection(
        type=AutoXRDTrainingInput,
        description='The inputs for the training.',
        repeats=True,
    )
    outputs = SubSection(
        type=AutoXRDModelReference,
        description='Reference to the entry of the trained model.',
    )
