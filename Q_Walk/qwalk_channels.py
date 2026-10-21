"""Detector channel metadata used by the Q-walk/t0 comparison."""

UPSTREAM_MODULE_BASE = 128
DOWNSTREAM_MODULE_BASE = 224
TRIGGER_CHANNEL = 192
FIXED_DOWNSTREAM_PAIR = (234, 243)

UPSTREAM_BAR_CHANNELS = {
    0: (141, 145),
    1: (143, 144),
    2: (142, 146),
    3: (138, 147),
    4: (140, 148),
    5: (136, 151),
    6: (139, 149),
    7: (133, 154),
    8: (137, 150),
    9: (135, 155),
    10: (134, 156),
    11: (130, 159),
    12: (131, 158),
    13: (128, 152),
    14: (129, 153),
    15: (132, 157),
}

DOWNSTREAM_BAR_CHANNELS = {
    bar: (ch1 + (DOWNSTREAM_MODULE_BASE - UPSTREAM_MODULE_BASE),
          ch2 + (DOWNSTREAM_MODULE_BASE - UPSTREAM_MODULE_BASE))
    for bar, (ch1, ch2) in UPSTREAM_BAR_CHANNELS.items()
}

REFERENCE_CHANNELS = (133, 137, 150, 154)

# Coefficients are ordered as (a, b, c) for a + b/Q + c/Q**2, in ns.
CMSNOTE_WALK_COEFFICIENTS_NS = {
    133: (0.0796, 238.3, -21560.0),
    154: (-0.5155, 422.6, -55830.0),
    137: (0.1218, 198.5, -15230.0),
    150: (-0.2292, 388.9, -51320.0),
}


def reference_topology(bar):
    """Return the exact channel topology for an upstream reference bar."""
    return frozenset(
        (*UPSTREAM_BAR_CHANNELS[int(bar)], *FIXED_DOWNSTREAM_PAIR, TRIGGER_CHANNEL)
    )

def exact_four_channel_topology(upstream_bar: int, downstream_bar: int) -> frozenset[int]:
    return frozenset(
        (*UPSTREAM_BAR_CHANNELS[upstream_bar],
         *DOWNSTREAM_BAR_CHANNELS[downstream_bar],
         TRIGGER_CHANNEL)
    )


def bar_pair_for_channel(channel: int) -> tuple[int, int]:
    """Return the two-ended detector bar containing an upstream/downstream channel."""
    channel = int(channel)
    for pairs in (UPSTREAM_BAR_CHANNELS, DOWNSTREAM_BAR_CHANNELS):
        for pair in pairs.values():
            if channel in pair:
                return pair
    raise ValueError(f"Channel {channel} is not part of a mapped two-ended bar")
