# tamper-detect (SC-2 camera-tampering sidecar)

CPU-only, no GPU/VLM. Flags camera tampering (spray, cover, blind, defocus) by
watching Laplacian-variance, edge-density, and histogram-entropy collapse over
a rolling window.

## Install
    python -m venv .venv && .venv\Scripts\Activate.ps1   # (or source .venv/bin/activate)
    pip install -r requirements.txt

## Run against a file
    python tamper_detect.py path/to/video.mp4

## Run against a live camera (RTSP)
    python tamper_detect.py "rtsp://<host>:<port>/<path>"

Prints the first-tamper frame + second, or nulls if none.

## Tuning (defaults chosen for the POC clips)
- baseline_frames: 30
- drop_ratio: 0.4
- min_signals: 2
- window: 8

## Scope
POC sanity build validated on an idealized full-occlusion (spray-to-black) clip
and confirmed silent on normal pedestrian/doorway footage. Partial or gradual
tampering (a thin cloth, slow defocus, small angle shift) will need threshold
tuning on real footage — noted as a follow-up, not handled here.
