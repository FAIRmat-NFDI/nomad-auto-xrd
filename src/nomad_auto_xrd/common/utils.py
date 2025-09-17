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

from nomad import infrastructure
from nomad.processing.data import PublicUploadFiles, StagingUploadFiles, Upload
from nomad_analysis.utils import get_reference

from nomad_auto_xrd.common.models import AnalysisInput

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger


def get_upload(upload_id: str, user_id: str) -> Upload | None:
    """
    Retrieve an upload by ID and check if the user is authorized to access it.
    Args:
        upload_id (str): The ID of the upload to retrieve.
        user_id (str): The ID of the user requesting access.
    Returns:
        Upload | None: The upload object if found and authorized, else None.
    """
    if infrastructure.mongo_client is None:
        infrastructure.setup_mongo()

    upload = Upload.get(upload_id)

    if upload is None:
        return None

    # Determine if user is authorized to get the upload.
    is_coauthor = isinstance(upload.coauthors, list) and user_id in upload.coauthors
    is_authorized = upload.main_author == user_id or is_coauthor

    # Raise error if not authorized
    if not is_authorized:
        raise PermissionError(
            f'User {user_id} is not authorized to access upload {upload_id}.'
        )

    return upload


def read_entry_archive(entry_id: str, upload_id: str, user_id: str) -> dict:
    """
    Read and return the archive of a specific entry from an upload after checking
    user authorization.

    Args:
        entry_id (str): The ID of the entry to read.
        upload_id (str): The ID of the upload containing the entry.
        user_id (str): The ID of the user requesting access.

    Returns:
        dict: The archive of the specified entry.
    """
    get_upload(upload_id, user_id)  # Check authorization

    # User is authorized, retrieve and return files
    if StagingUploadFiles.exists_for(upload_id):
        upload_files = StagingUploadFiles(upload_id)
    elif PublicUploadFiles.exists_for(upload_id):
        upload_files = PublicUploadFiles(upload_id)
    else:
        raise ValueError(f'Upload files for upload {upload_id} not found.')

    archive = upload_files.read_archive(entry_id)

    if archive is None:
        raise ValueError(f'Entry with id {entry_id} not found in upload {upload_id}.')

    return archive[entry_id]


def pattern_preprocessor(
    xrd_entry_archive: dict, logger: 'BoundLogger | None' = None
) -> list[AnalysisInput] | None:
    """
    Extract relevant data from `XRayDiffraction` sections into `AnalysisInput` objects.

    Args:
        xrd_entry_archive (dict): The XRD entry archive to process.
        logger (BoundLogger | None): Optional logger for logging warnings.

    Returns:
        list[AnalysisInput]: List of processed data ready for analysis.
    """
    if 'data_files' in xrd_entry_archive['data']:
        filenames = xrd_entry_archive['data']['data_files']
    elif 'data_file' in xrd_entry_archive['data']:
        filenames = [xrd_entry_archive['data']['data_file']]
    else:
        raise AttributeError('No data file(s) found in the XRD section.')
    patterns = xrd_entry_archive['results']['properties']['structural'][
        'diffraction_pattern'
    ]
    if len(filenames) != len(patterns):
        raise ValueError(
            'Number of data files does not match number of diffraction patterns.'
        )
    if not patterns:
        raise ValueError('No diffraction patterns found in the XRD section.')
    analysis_inputs = []
    for idx, (filename, pattern) in enumerate(zip(filenames, patterns)):
        two_theta = pattern['two_theta_angles']
        intensity = pattern['intensity']
        if two_theta is None or intensity is None:
            (logger.warning if logger else print)(
                f'Warning: Missing two_theta or intensity in pattern {idx} '
                f'of entry {xrd_entry_archive["metadata"].get("entry_id", None)}. '
                'Skipping.'
            )
            continue
        analysis_inputs.append(
            AnalysisInput(
                filename=os.path.basename(filename),
                two_theta=two_theta,
                intensity=intensity,
                measurement_m_proxy=get_reference(
                    xrd_entry_archive['metadata'].get('upload_id', None),
                    xrd_entry_archive['metadata'].get('entry_id', None),
                    f'data/results/{idx}',
                ),
            )
        )
    return analysis_inputs
