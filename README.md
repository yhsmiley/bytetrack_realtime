# ByteTrack

## Introduction

A realtime adaptation of [ByteTrack](https://github.com/ifzhang/ByteTrack).

## Dependencies

- Python
- Numpy, `pip install numpy`

## Install

- clone this repo & install bytetrack-realtime as a python package using `pip` or as an editable package if you like (`-e` flag)
```bash
cd bytetrack_realtime && pip3 install .
```

## Run

Refer to an example [here](https://github.com/yhsmiley/streamlit_cv/blob/a80b0f49cfa26d39f846047dbe90a29787b1a584/misc/utils.py#L41) to use ScaledYOLOv4 detector with ByteTrack. You can also use any other detectors as long as the input format to the tracker is the same.

Example usage:
```python
from bytetrack_realtime.byte_tracker import ByteTracker
tracker = ByteTracker(track_thresh=0.6, track_buffer=30, match_thresh=0.9)
bbs = object_detector.detect(frame)
tracks = tracker.update(detections=bbs)
for track in tracks:
   track_id = track.track_id
   ltrb = track.ltrb
   score = track.score
   det_class = track.det_class
```
