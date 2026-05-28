"""Vehicle directory — one subpackage per vehicle type (spec §6, §7).

Each subdirectory follows the Vehicle Module Contract:
    params.py     — Params pytree
    state.py      — State shape conventions
    presets.py    — Factory functions (preset airframes)
    tier{0,1,2}.py — Tier-specific step functions

Today: fixed_wing. Future: quadrotor, tailsitter, morphing_wing.
"""
