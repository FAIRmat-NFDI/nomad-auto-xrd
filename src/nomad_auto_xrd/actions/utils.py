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

import contextlib
import threading
from contextvars import copy_context

from temporalio import activity


@contextlib.contextmanager
def with_activity_heartbeat(delay: float):
    """
    A context manager that sends temporal heartbeats in a background thread.

    This is used to wrap long-running, synchronous code blocks that are executed
    within a temporal activity. It prevents the activity from timing out by
    periodically sending heartbeats to the temporal server while the main thread
    is busy.

    The heartbeating stops automatically when the context is exited.

    Args:
        delay (float): The time in seconds between heartbeat calls. This should
            be set to a value less than the temporal activity's heartbeat
            timeout.
    """

    stop_event = threading.Event()
    # Copy context to preserve Temporal activity context vars in the new thread
    ctx = copy_context()

    def _heartbeat_loop():
        elapsed = 0
        # Check every 1s for responsiveness
        check_interval = 1
        while not stop_event.is_set():
            if elapsed >= delay:
                if activity.in_activity():
                    activity.heartbeat()
                elapsed = 0

            if stop_event.wait(timeout=check_interval):
                break
            elapsed += check_interval

    heartbeat_thread = threading.Thread(
        target=lambda: ctx.run(_heartbeat_loop), daemon=True
    )
    heartbeat_thread.start()

    try:
        yield
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=delay + 1)
