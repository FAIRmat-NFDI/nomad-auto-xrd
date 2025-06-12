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
from typing import TYPE_CHECKING

from nomad_analysis.utils import get_reference
from nomad_measurements.xrd.schema import XRayDiffraction

from nomad_auto_xrd.models import AnalysisInput

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger


def single_pattern_preprocessor(
    xrd_entries: list[XRayDiffraction], logger: 'BoundLogger' = None
) -> None:
    """
    Extract relevant data from `XRayDiffraction` entries for generation of .xy
    files. This preprocessor is suitable for `XRayDiffraction` entries that have not
    more than one diffraction pattern per entry with a single data file.

    Args:
        xrd_entries (list[XRayDiffraction]): List of XRD entries to preprocess.
        logger (BoundLogger | None): Optional logger for logging warnings.

    Returns:
        list[AnalysisInput]: List of processed data ready for analysis.
    """
    prepared_data = []
    for xrd in xrd_entries:
        if not isinstance(xrd, XRayDiffraction):
            continue
        try:
            filename = xrd.data_file
            pattern = xrd.m_parent.results.properties.structural.diffraction_pattern[0]
            two_theta = pattern.two_theta_angles
            intensity = pattern.intensity
        except AttributeError as e:
            (logger.warning if logger else print)(
                f'Encountered AttributeError: {e}. Skipping the XRD entry',
            )
            continue
        if two_theta is None or intensity is None:
            (logger.warning if logger else print)(
                'XRD data is missing. Skipping the XRD entry.'
            )
            continue
        prepared_data.append(
            AnalysisInput(
                filename=filename,
                two_theta=two_theta.to('degree').magnitude.tolist(),
                intensity=intensity.tolist(),
                entry_m_proxy=get_reference(
                    xrd.m_parent.metadata.upload_id,
                    xrd.m_parent.metadata.entry_id,
                ),
            )
        )
    return prepared_data
