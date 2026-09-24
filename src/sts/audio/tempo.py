"""Pitch-preserving speed-up for speech, used when spoken playback has fallen
behind and needs to catch up (see Pipeline._catch_up).

WSOLA (waveform-similarity overlap-add): the clip is cut into short
overlapping frames that are laid down closer together than they were taken,
and each frame is nudged (within a few ms) to wherever it lines up best with
the previous one, so the joins don't smear the waveform. That keeps the
voice's pitch and timbre while shortening it - unlike resampling, which
would raise the pitch - and suits speech better than a phase vocoder
(librosa's time_stretch), which makes voices sound phasey/metallic.
"""

import numpy as np

FRAME_MS = 30.0
TOLERANCE_MS = 10.0


def time_stretch(audio: np.ndarray, sample_rate: int, rate: float) -> np.ndarray:
    """`audio` played `rate` times faster at the same pitch (rate > 1 is
    shorter). Returned unchanged for a rate of 1 or a clip too short to
    stretch."""
    audio = np.asarray(audio, dtype=np.float32)
    frame = int(sample_rate * FRAME_MS / 1000) // 2 * 2
    if abs(rate - 1.0) < 1e-3 or len(audio) < 2 * frame:
        return audio
    synthesis_hop = frame // 2
    analysis_hop = synthesis_hop * rate
    tolerance = int(sample_rate * TOLERANCE_MS / 1000)
    # A periodic Hann window at 50% overlap sums to exactly 1, so overlapped
    # frames keep the clip's level.
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame) / frame)).astype(np.float32)

    # Padded so every search region and the natural-continuation template
    # below stay in bounds; positions are in padded coordinates.
    padded = np.concatenate([np.zeros(tolerance, np.float32), audio, np.zeros(frame + 2 * tolerance, np.float32)])
    out_len = int(round(len(audio) / rate))
    frames = out_len // synthesis_hop + 1
    out = np.zeros(frames * synthesis_hop + frame, dtype=np.float32)
    # The first frame isn't faded in (nothing overlaps its first half), so
    # the clip starts at full level - a fade there would be a dip where a
    # stretched chunk continues one already played. What the windows add up
    # to at each output sample is divided out at the end, which keeps any
    # partly overlapped stretch at the tail at full level too.
    first_window = window.copy()
    first_window[: frame // 2] = 1.0
    weight = np.zeros_like(out)

    position = tolerance  # where frame 0 is taken from: the clip's own start
    for k in range(frames):
        if k:
            nominal = tolerance + int(round(k * analysis_hop))
            if nominal + frame > len(padded) - tolerance:
                break
            # What would have followed the previous frame had nothing been
            # skipped - the new frame should look as much like it as possible.
            template = padded[position + synthesis_hop : position + synthesis_hop + frame]
            region = padded[nominal - tolerance : nominal + tolerance + frame]
            scores = np.correlate(region, template, mode="valid")
            position = nominal - tolerance + int(np.argmax(scores))
        w = first_window if k == 0 else window
        out[k * synthesis_hop : k * synthesis_hop + frame] += padded[position : position + frame] * w
        weight[k * synthesis_hop : k * synthesis_hop + frame] += w
    out = out[:out_len]
    weight = weight[:out_len]
    covered = weight > 0.1
    out[covered] /= weight[covered]
    return out
