"""Shared test configuration.

Force the non-interactive Agg backend so figure-saving helpers can be exercised
without a display and without writing to the user's matplotlib cache.
"""

import matplotlib

matplotlib.use("Agg")
