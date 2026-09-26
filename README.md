# Web app descriptors

## What is it ?
This is a software for musical analysis. It computes harmonic descriptors on audio signal of musical pieces or musical extracts. 

The principle is as follows. The user imports one or several audio file(s), and for each audio file a segmentation file in txt format. A next section will explain how to construct the segmentation file. Static and dynamic harmonic descriptors are then computed, and display on a two-dimensional graph, depending on the two descriptors mapped to the x and y axis.

<img width="1907" height="986" alt="image" src="https://github.com/user-attachments/assets/a6a24b26-c479-4cc6-9c6f-b4566ce4699f" />


## Explanations

### Segmentation
The analysis methodology, which lies at the intersection of the signal and the symbolic, requires the audio signal to be segmented temporally. Harmonic descriptors are calculated based on the time segments, or verticalities, obtained in this way. For a given musical excerpt, multiple segmentations may be relevant, depending on the level of granularity and the purpose of the analysis.

The segmentation must have the extension .txt, and follow the following template: 
```
10.634739229	New Point
16.091428571	New Point
26.447528345	New Point
32.484716553	New Point
37.593106576	New Point
42.051337868	New Point
```
A direct way to produce the segmentation file is to use the software [Sonic Visualiser](https://www.sonicvisualiser.org/). To do so, import your audio, create a Time Instant Layer corresponding to your desired segmentation, and export it.

<img width="1018" height="735" alt="image" src="https://github.com/user-attachments/assets/174daf8a-90ae-41e4-9d9f-c4ee0132566a" />

### Spectral analysis
The spectral analysis uses a CQT ([Constant-Q Transform](https://librosa.org/doc/0.11.0/generated/librosa.cqt.html)). This transformation, compared to the classical STFT (Short Time Fourier Transform), has the advantage of preserving the pitch accuracy (at the expense of the time accuracy).

### Harmonic descriptors
Two types of harmonic descriptors are used, static and dynamic. Static descriptors describe the simultaneous (synchronic) spectral interactions inside a verticality, while dynamic descriptors describe the successive (diachronic) spectral interactions between two consecutive verticalities. In musical terms, static descriptors analyse isolated chords, while dynamic ones analyse transitions from a chord to another. 

#### Static descriptors
- Roughness
- Harmonicity
- Inverse harmonicity
- Concordance (need separated audio streams)
- Total concordance (need separated audio streams)

#### Dynamic descriptors
- Harmonic change
- Differential concordance
- Differential roughness

For definitions and further details on harmonic descriptors, refer to [Les descripteurs harmoniques: étude théorique et applications musicologiques](https://shs.hal.science/tel-03360582/).

### Multiple files
There is the possibility to import several audio files. This is useful to compare different musical excerpts and represent them on the same map. 
<img width="897" height="698" alt="image" src="https://github.com/user-attachments/assets/291b4a3f-6433-48b4-a572-8ec12c7b8735" />

## Requirements

In order to use this project, run the following command on your machine:

```bash
# For any operating system
pip3 install -r requirements.txt
```
or install the dependencies in a virtual python environment. 

## How to use

1. Clone the repository on your machine and install the requirements.
2. Go to the repo on your terminal, and run the command `python3 app.py`. You should see someting like `Dash is running on http://127.0.0.1:8050/`.
3. Copy paste this address in your browser. You should see the app page.
4. Import the audio and segmentation files.
5. Choose the type of descriptors you are interested in (static, dynamic or both), then click on "Compute". The computation may take few seconds.



## Further reading
- My PhD Thesis: [Les descripteurs harmoniques: étude théorique et applications musicologiques](https://shs.hal.science/tel-03360582/).
