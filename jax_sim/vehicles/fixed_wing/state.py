"""Fixed-wing state-shape conventions (spec §7.2).

Day-one state representation is a flat (17,) jnp.ndarray:

    state = [pos(3), vel(3), quat(4 w-x-y-z), omega(3), actuators(4)]

Indices:
    POS   = slice(0, 3)   NED position [m]
    VEL   = slice(3, 6)   NED velocity [m/s]
    QUAT  = slice(6, 10)  body->earth quaternion (w, x, y, z)
    OMEGA = slice(10, 13) body rates [rad/s]
    ACT   = slice(13, 17) actuators [aileron, elevator, rudder, throttle]
                          aileron/elevator/rudder in radians; throttle in [0, 1]

A `NamedTuple`-based FixedWingState is a planned migration (spec §7.2 ideal
form). It is deferred because flipping the state shape ripples through env,
controllers, and every script. For now, the flat-array convention is the
authority.
"""

POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)
ACT = slice(13, 17)
STATE_SIZE = 17

__all__ = ["POS", "VEL", "QUAT", "OMEGA", "ACT", "STATE_SIZE"]
