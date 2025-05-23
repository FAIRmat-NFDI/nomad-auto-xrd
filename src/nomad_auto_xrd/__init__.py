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
from nomad.config.models.plugins import SchemaPackageEntryPoint

try:
    from ._version import version as __version__
except ImportError:
    __version__ = ''


class AutoXRDSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    """
    Schema for training Auto XRD models and running Auto XRD analysis.
    """

    def load(self):
        from nomad_auto_xrd.schema import m_package

        return m_package


schema_entry_point = AutoXRDSchemaPackageEntryPoint(
    name='Auto XRD',
    description='Schema for training Auto XRD models and running Auto XRD analysis.',
)
