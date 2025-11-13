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
import time
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
from nomad import infrastructure
from nomad.processing.data import PublicUploadFiles, StagingUploadFiles, Upload
from nomad_analysis.utils import get_reference
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from nomad_auto_xrd.common.models import (
    AnalysisInput,
    PatternAnalysisResult,
    PhasesPosition,
)

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
    patterns = xrd_entry_archive['results']['properties']['structural'][
        'diffraction_pattern'
    ]
    if not patterns:
        raise ValueError('No diffraction patterns found in the XRD section.')
    analysis_inputs = []
    for idx, pattern in enumerate(patterns):
        two_theta = pattern['two_theta_angles']
        intensity = pattern['intensity']
        if two_theta is None or intensity is None:
            (logger.warning if logger else timestamped_print)(
                f'Warning: Missing two_theta or intensity in pattern {idx} '
                f'of entry {xrd_entry_archive["metadata"].get("entry_id", None)}. '
                'Skipping.'
            )
            continue
        analysis_inputs.append(
            AnalysisInput(
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


def simulate_pattern(
    cif: str, wavelength: float, two_theta_range: tuple[float, float]
) -> tuple[list[float], list[float]]:
    """
    Simulates an XRD pattern for a given phase and wavelength over a specified
    two-theta range.

    Args:
        cif (str): Path to the CIF file of the crystal structure.
        wavelength (float): The X-ray wavelength in Angstroms.
        two_theta_range (tuple[float, float]): The range of two-theta angles to
            simulate.

    Returns:
        tuple[list[float], list[float]]: Simulated two-theta angles and corresponding
        intensities.
    """

    structure = Structure.from_file(cif)
    calculator = XRDCalculator(wavelength=wavelength)
    pattern = calculator.get_pattern(structure, two_theta_range=two_theta_range)

    return pattern.x.tolist(), pattern.y.tolist()


def plot_identified_phases(data: PatternAnalysisResult) -> dict:
    """
    Generates a Plotly figure with the measured XRD pattern and simulated patterns of
    the identified phases.

    Args:
        data (PatternAnalysisResult): The analysis result containing measured and
            identified phases.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.two_theta,
            y=data.intensity,
            mode='lines',
            name='Measured Pattern',
        )
    )
    for phase in data.phases:
        fig.add_trace(
            go.Scatter(
                x=phase.simulated_two_theta,
                y=len(phase.simulated_intensity)
                * [min(data.intensity) / 2],  # baseline at y=1/2 of min intensity
                mode='markers',
                marker=dict(symbol='line-ns-open', size=10, line=dict(width=2)),
                name=f'{phase.name} (conf={phase.confidence:.1f})',
            )
        )
    fig.update_layout(
        title='Measured XRD pattern with simulated patterns of identified phases',
        xaxis_title='2<i>θ</i> / °',
        yaxis_title='Intensity',
        hovermode='closest',
        template='plotly_white',
        dragmode='zoom',
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False, type='log'),
        showlegend=True,
        legend=dict(
            orientation='h',  # Horizontal orientation
            yanchor='top',
            y=-0.30,  # Position below the plot
            xanchor='center',
            x=0.5,  # Center horizontally
        ),
    )
    return fig.to_plotly_json()


def plot_identified_phases_sample_position(
    data: list[PhasesPosition],
) -> dict:
    """
    Generates a Plotly scatter plot with x and y position on the axes and the phase with
    highest confidence as the marker.

    Args:
        data (list[PhasesPosition]): List of identified phases with sample positions.
    """
    df = pd.DataFrame(
        {
            'x_position': [d.x_position for d in data],
            'y_position': [d.y_position for d in data],
            'phase': [d.phases[0].name if d.phases else 'Unknown' for d in data],
            'confidence': [d.phases[0].confidence if d.phases else 0.0 for d in data],
        }
    )
    x_unit = data[0].x_unit if data else ''
    y_unit = data[0].y_unit if data else ''
    fig = px.scatter(
        df,
        x='x_position',
        y='y_position',
        color='phase',
        size='confidence',
    )
    fig.update_traces(
        marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))
    )
    fig.update_layout(
        title='Primary identified phases for the Combinatorial library',
        legend_title_text='',
        xaxis_title=f'x / {x_unit}',
        yaxis_title=f'y / {y_unit}',
        hovermode='closest',
        template='plotly_white',
        dragmode='zoom',
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False),
        showlegend=True,
    )

    return fig.to_plotly_json()


def get_total_memory_mb():
    """Get the total memory usage of the current process and its children."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    # Add memory of all child processes
    for child in process.children(recursive=True):
        try:
            mem += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return mem / 1024 / 1024  # Convert to MB


def timestamped_print(message: str):
    """Print a message with a timestamp."""
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}: {message}')
