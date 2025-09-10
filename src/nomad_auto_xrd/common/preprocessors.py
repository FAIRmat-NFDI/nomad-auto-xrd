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
import os
from typing import TYPE_CHECKING

from nomad_analysis.utils import get_reference
from nomad_measurements.xrd.schema import XRayDiffraction

from nomad_auto_xrd.common.models import AnalysisInput

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger


def single_pattern_preprocessor(
    xrd_sections: list[XRayDiffraction], logger: 'BoundLogger' = None
) -> list[AnalysisInput]:
    """
    Extract relevant data from `XRayDiffraction` sections into `AnalysisInput` objects.
    This preprocessor is suitable for `XRayDiffraction` sections that have not
    more than one diffraction pattern per entry with a single data file.

    Args:
        xrd_sections (list[XRayDiffraction]): List of XRD sections to preprocess.
        logger (BoundLogger | None): Optional logger for logging warnings.

    Returns:
        list[AnalysisInput]: List of processed data ready for analysis.
    """
    prepared_data = []
    for xrd in xrd_sections:
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
                filename=os.path.basename(filename),
                two_theta=two_theta.to('degree').magnitude.tolist(),
                intensity=intensity.tolist(),
                measurement_m_proxy=get_reference(
                    xrd.m_parent.metadata.upload_id,
                    xrd.m_parent.metadata.entry_id,
                    'data/results/0',
                ),
            )
        )
    return prepared_data


def multiple_patterns_preprocessor(
    xrd_sections: list[XRayDiffraction], logger: 'BoundLogger' = None
) -> list[AnalysisInput]:
    """
    Extract relevant data from `XRayDiffraction` section into `AnalysisInput` objects.
    This preprocessor is suitable for `XRayDiffraction` sections that have multiple
    diffraction patterns per entry with multiple data files.

    Args:
        xrd_sections (list[XRayDiffraction]): List of XRD sections to preprocess.
        logger (BoundLogger | None): Optional logger for logging warnings.

    Returns:
        list[AnalysisInput]: List of processed data ready for analysis.
    """
    prepared_data = []
    for xrd in xrd_sections:
        try:
            filenames = xrd.data_files
            patterns = xrd.m_parent.results.properties.structural.diffraction_pattern
        except AttributeError as e:
            (logger.warning if logger else print)(
                f'Encountered AttributeError: {e}. Skipping the XRD entry',
            )
            continue
        if not patterns:
            (logger.warning if logger else print)(
                'XRD data is missing. Skipping the XRD entry.'
            )
            continue
        for idx, (filename, pattern) in enumerate(zip(filenames, patterns)):
            two_theta = pattern.two_theta_angles
            intensity = pattern.intensity
            if two_theta is None or intensity is None:
                (logger.warning if logger else print)(
                    'XRD data is missing. Skipping the XRD entry.'
                )
                continue
            prepared_data.append(
                AnalysisInput(
                    filename=os.path.basename(filename),
                    two_theta=two_theta.to('degree').magnitude.tolist(),
                    intensity=intensity.tolist(),
                    measurement_m_proxy=get_reference(
                        xrd.m_parent.metadata.upload_id,
                        xrd.m_parent.metadata.entry_id,
                        f'data/results/{idx}',
                    ),
                )
            )
    return prepared_data
