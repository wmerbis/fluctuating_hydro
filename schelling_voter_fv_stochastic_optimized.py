#!/usr/bin/env python3
"""Compatibility entry point for the optimized Schelling--Voter FV solver.

The optimized production implementation lives in :mod:`finite_volume_stochastic`.
This module preserves the requested historical filename for scripts that import
or execute ``schelling_voter_fv_stochastic_optimized.py`` directly.
"""

from finite_volume_stochastic import (  # noqa: F401
    Array,
    ModelParameters,
    SchellingVoterFVSolver,
    benchmark_fast_paths,
    main,
    make_random_initial_condition,
)


if __name__ == "__main__":
    main()
