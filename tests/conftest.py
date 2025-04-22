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
import logging
import os

import pytest
import structlog
from nomad.client import normalize_all, parse
from nomad.utils import structlogging
from structlog.testing import LogCapture

structlogging.ConsoleFormatter.short_format = True
setattr(logging, 'Formatter', structlogging.ConsoleFormatter)


@pytest.fixture(
    name='caplog',
    scope='function',
)
def fixture_caplog(request):
    """
    Extracts log messages from the logger and raises an assertion error if the specified
    log levels in the `request.param` are found.
    """
    caplog = LogCapture()
    processors = structlog.get_config()['processors']
    old_processors = processors.copy()

    try:
        processors.clear()
        processors.append(caplog)
        structlog.configure(processors=processors)
        yield caplog
        for record in caplog.entries:
            if record['log_level'] in request.param:
                raise AssertionError(
                    f"Log level '{record['log_level']}' found: {record}"
                )
    finally:
        processors.clear()
        processors.extend(old_processors)
        structlog.configure(processors=processors)


@pytest.fixture(
    name='parsed_measurement_archives',
    scope='function',
)
def fixture_parsed_measurement_archives(request):
    """
    Sets up data for testing and cleans up after the test. The data file is parsed,
    returning an `EntryArchive` object. It contains a reference to the `.archive.json`
    file created by plugin parsers for the measurement data. Parsing this
    `.archive.json` file returns the `EntryArchive` object for the measurement data,
    which is finally yeilded to the test function.

    Parameters:
        request.param (list): List of relative file paths to the measurement data files.
    """
    clean_up_extensions = ['.archive.json', '.nxs', '.h5']

    measurement_archives = []
    for rel_file_path in request.param:
        parse(rel_file_path)
        rel_measurement_archive_path = os.path.join(
            rel_file_path.rsplit('.', 1)[0] + '.archive.json'
        )
        measurement_archive = parse(rel_measurement_archive_path)[0]
        normalize_all(measurement_archive)
        measurement_archives.append(measurement_archive)

    yield measurement_archives

    # clean up
    for rel_file_path in request.param:
        for ext in clean_up_extensions:
            path = os.path.join(rel_file_path.rsplit('.', 1)[0] + ext)
            if os.path.exists(path):
                os.remove(path)
