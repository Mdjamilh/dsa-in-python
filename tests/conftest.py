"""Shared pytest configuration.

Hypothesis profiles:

- ``dev`` (default): 100 examples per property, fast enough to run on every save.
- ``ci``: 400 examples per property, no per-example deadline (shared CI runners,
  especially Windows ones, have noisy timing), and the database is not written.

Select one with ``HYPOTHESIS_PROFILE=ci``.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=100)
settings.register_profile(
    "ci",
    max_examples=400,
    deadline=None,
    database=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
