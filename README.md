# Web app descriptors

## What is it ?
This is a software for musical analysis. It computes harmonic descriptors on audio signal of musical pieces or musical extracts. 

The principle is as follows. The user imports one or several audio file(s), and for each audio file a segmentation file in txt format. A next section will explain how to construct the segmentation file. Static and dynamic harmonic descriptors are then computed, and display on a two-dimensional graph, depending on the two descriptors mapped to the x and y axis.

For further explanations, please refer to the reading section.

![](images/screen.png)

## Requirements

In order to be able to do whatever you want with this project, you need run the following command on your machine:

```bash
# For any operating system
pip3 install -r requirements.txt
```

You need to have Max/Msp installed on your machine. 

Before opening the patch, please open MaxMsp, go to `Options -> File Preferences` and add the path to `HarmonicSpaces/images`, activating the subfolders option. Close Max/Msp in order to validate this step.

## How to use

To open the patch, click on `HarmonicSpaceProject/HarmonicSpaceProject.maxproj`. Before being able to hear anything, you need to load a sound. 

1. Drag and drop a sound. It needs to be monophonic in wav format. Don't use any space in the name. Once the sound is loaded, the analysis can take few seconds. It is done when the spectrum, the map and the sound name appear. Then you can play. 
2. Play on the map. There are two way to make sound, you can either click on the chord you want to hear on the map, and draw trajectories on the map, or plug a midi instrument. If you plug a midi polyphonic instrument, it is recommanded to use a MPE instrulment (Midi Polyphonic Expression), in order to have a continuous pitch control and to be able to access all the chords on the map. Such instruments include the [Osmose](https://www.expressivee.com/) or the [Haken Continuum](https://www.hakenaudio.com/).

## Further reading
- My PhD Thesis: [Les descripteurs harmoniques: étude théorique et applications musicologiques](https://shs.hal.science/tel-03360582/).
