"""Physical constants (vehicle-agnostic).

After the §18 / §A.5 restructure, only true environmental physical constants
live here. All airframe-specific values (mass, inertia, geometry, propulsion,
servo time constants, aero segment definitions) moved to
`vehicles/fixed_wing/presets.py`.
"""

G = 9.81      # m/s^2, standard gravity
RHO = 1.225   # kg/m^3, sea-level air density (ISA)

__all__ = ["G", "RHO"]
