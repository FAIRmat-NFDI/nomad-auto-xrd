from nomad.config.models.plugins import SchemaPackageEntryPoint
from pydantic import Field


class NewSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_auto_xrd.schema_packages.model import m_package

        return m_package


auto_xrd = NewSchemaPackageEntryPoint(
    name='auto_xrd',
    description='New schema package entry point configuration.',
)


class XRDAnalaysisSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_auto_xrd.schema_packages.auto_xrd_analysis import m_package

        return m_package


auto_xrd_analysis = XRDAnalaysisSchemaPackageEntryPoint(
    name='auto_xrd_analysis',
    description='New schema package entry point configuration.',
)


class AnalysisSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_auto_xrd.schema_packages.analysis import m_package

        return m_package


class TrainingSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_auto_xrd.schema_packages.training import m_package

        return m_package


analysis_schema = AnalysisSchemaPackageEntryPoint(
    name='Auto XRD Inference Schema',
    description='Schema for performing Auto XRD analysis.',
)

training_schema = TrainingSchemaPackageEntryPoint(
    name='Auto XRD Training Schema',
    description='Schema for training Auto XRD models.',
)
