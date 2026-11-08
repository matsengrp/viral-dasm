"""Physical and numerical constants for DASM2 computations.

This module centralizes magic numbers and configuration constants to improve
maintainability and prevent duplication across the codebase.
"""

# Numerical constants for computation stability
BIG_VALUE = 1e9
"""Large value used for masking and numerical stability in log-space computations."""

SMALL_PROBABILITY = 1e-10
"""Small probability threshold to prevent log(0) and division by zero errors."""

# Optimization constants
MAX_BRACKET_ITERATIONS = 4000
"""Maximum iterations for bracket finding in optimization routines."""
