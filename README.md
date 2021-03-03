# chordapp
This project implements techniques for the extraction of two new musical features in an interactive application for quick visualisation.

### Features extracted:
- **harmoniousness**: measures how close notes are to the central vector (sum of all chromagram components, sorted according to the circle of fifths, when viewed as a circular distribution) on the circle of fifths
- **coharmoniousness**: measures how *"above"* or *"below"* notes with respect to the central vector on the circle of fifths. Corresponds to major (>0) or minor (<0) quality.

### Files:
- **"chordapp..."** is the main application.
- **"HStats"** contains functions for extracting those two musical features as well as some chromagram manipulation functions (mainly sorting the chromagram according to the circle of fifths). Used in ProcessAudio.
- **"conveniently"** contains some multithreading decorators.
- **"ProcessAudio"** contains some feature extraction and chromagram related functions.