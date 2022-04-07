import base64
import datetime
import io
import sys
import os
import json

import dash
from dash import dcc, html, dash_table, Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
from operator import itemgetter, attrgetter, truediv
from numpy import linalg as LA

#####################################################################################
# Calcul

descrNames = {'roughness':'Roughness', 'harmonicity':'Harmonicity', 'concordance':'Concordance', 'concordanceTot':'Concordance Totale', 'harmonicChange':'Harmonic Change', 'diffConcordance':'Differential Concordance','diffRoughness':'Differential Roughness'}
descrList = {'static': ['roughness', 'harmonicity', 'concordance', 'concordanceTot'],'staticOneSound': ['roughness', 'harmonicity'] , 'dynamic': ['harmonicChange', 'diffConcordance', 'diffRoughness']}
list_window = ['hann', 'hamming', 'blackman', 'flattop', 'boxcar', 'triang']

BINS_PER_OCTAVE = 12*8
FILTER_SCALE = 1
STEP = 512
cmap = 'gray_r'
seuil_activation = 0.01
simpl = True
#Roughness
type_rug_signal = True # If true, roughness is computed between all couple of partials. If False, just between partials from different notes
rug_simpl = True # If True, computed on simplified specra (pic spectra). If False, on non-simplified spectra
#Harmonicity:
σ = 12 * 4 # in octave divisions
κ = 15
decr = 0
#Affichage
plot_descr = True


class SignalSepare:
    """ Prend en entrée en signal et le signal des pistes audio séparées. """

    def __init__(self, signal, sr, pistes, onsets_txt, window, hpss, hpss_margin, Notemin, Notemax):
        self.y = signal
        self.pistes = pistes
        self.sr = sr
        self.n_pistes = len(pistes)
        self.Notemin = Notemin
        self.Notemax = Notemax
        self.window = window
        self.hpss = hpss
        self.hpss_margin = hpss_margin
        self.n_bins = 0
        self.fmin = 0
        self.fmax = 0
        self.n_bins = 0
        self.n_frames = 0
        self.N_sample = []
        self.Onset_given = True
        self.onsets_txt = onsets_txt
        self.onset_times = []
        self.onset_frames = []
        self.ChromDB_reloc = []
        self.Chrom = []
        self.chromSync = []
        self.chromSyncDB = []
        self.chromPistesSync = []
        self.chromSyncSimpl = []
        self.chromPistesSyncSimpl = []
        self.ChromNoHpss = []
        self.energy = []
        self.energyPistes = []
        self.activation =  []
        self.n_notes = []
        self.chrom_concordance = []
        self.concordance = []
        self.chrom_concordanceTot = []
        self.concordanceTot = []
        self.chrom_concordance3 = []
        self.concordance3 = []
        self.tension = []
        self.roughness = []
        self.chrom_harmonicity = []
        self.chrom_harmonicityInv = []
        self.liste_partials = []
        self.tensionSignal = []
        self.chrom_roughness = []
        self.roughnessSignal = []
        self.chrom_harmonicChange = []
        self.harmonicChange = []
        self.chrom_diffConcordance = []
        self.diffRoughness = []
        self.chrom_diffRoughness = []
        self.diffConcordance = []
        self.harmonicity = []
        self.virtualPitch = []




    def OnsetFrames(self):
        onsets = []
        with open(self.onsets_txt,'r') as f:
            for line in f:
                l = line.split()
                onsets.append(float(l[0]))
        self.onset_times = np.asarray(onsets)
        self.onset_frames = librosa.time_to_frames(self.onset_times, sr=self.sr, hop_length = STEP)




    def GlobalSpectralAnalyis(self):

        """ - Compute global CQT transform on signal,
            - Detect onsets if not given,
            - Synchronise global spectrum by temporal average on the timespans between two consecutive onsets """

        # CQT Constant-Q Transform
        self.fmin = librosa.note_to_hz(self.Notemin)
        self.fmax = librosa.note_to_hz(self.Notemax)
        self.n_bins = int((librosa.note_to_midi(self.Notemax) - librosa.note_to_midi(self.Notemin))*BINS_PER_OCTAVE/12)
        self.Chrom = np.abs(librosa.cqt(y=self.y, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins, window=self.window, filter_scale = FILTER_SCALE))
        self.N = self.Chrom.shape[1] # Number of frames
        self.times = librosa.frames_to_time(np.arange(self.N), sr=self.sr, hop_length=STEP)  # Frame times

        # CALCUL DES ONSETS (pour onset précalculé, le rentrer dans self.onset_frames à l'initialisation).......
        self.OnsetFrames()
        self.onset_frames = librosa.util.fix_frames(self.onset_frames, x_min=0, x_max=self.Chrom.shape[1]-1)
        self.onset_times = librosa.frames_to_time(self.onset_frames, sr=self.sr, hop_length = STEP)
        self.n_frames = len(self.onset_frames)-1
        self.n_notes = np.ones(self.n_frames)

        # SPECTRUM RELOCALISATION
        freq_analyse = [self.fmin*2**(k/BINS_PER_OCTAVE) for k in range(self.n_bins)]
        R = [round(self.sr * FILTER_SCALE/(f*(2**(1/BINS_PER_OCTAVE)-1))) for f in freq_analyse]
        self.N_sample = [round(n/STEP) for n in R]
        Chrom_copy = np.copy(self.Chrom)
        for k in range(self.n_bins):
            for n in reversed(range(self.N)):
                if n <= self.N_sample[k]: self.Chrom[k,n] = Chrom_copy[k,n]
                else: self.Chrom[k,n] = Chrom_copy[k,n-int(self.N_sample[k]/2)]


        # HARMONIC / PERCUSSIVE DECOMPOSITION
        if self.hpss:
            self.ChromNoHpss = np.copy(self.Chrom)
            self.Chrom = librosa.decompose.hpss(self.Chrom, margin=self.hpss_margin)[0]

        self.ChromDB = librosa.amplitude_to_db(self.Chrom, ref=np.max)


        # SPECTRUM SYNCHRONISATION on the interval between two onsets
        self.chromSync = np.zeros((self.n_bins,self.n_frames))

        ## Delay at the beginning and at the end of the synchronisation interval, depending on the frequency
        Δmin = 0.1 # en secondes
        for i in range(self.n_bins):
            f = self.fmin*2**(i/BINS_PER_OCTAVE)
            T_ret = 1.5 / (f * (2**(1.0/(12*4)) - 1))
            for j in range(self.n_frames):
                if T_ret < (self.onset_times[j+1] - self.onset_times[j+1]) - Δmin:
                    self.chromSync[i,j] = np.median(self.Chrom[i][(self.onset_frames[j]+int(librosa.time_to_frames(T_ret, sr=self.sr, hop_length = STEP))):(self.onset_frames[j+1])])
                else:
                    self.chromSync[i,j] = np.median(self.Chrom[i][(self.onset_frames[j+1]-int(librosa.time_to_frames(Δmin, sr=self.sr, hop_length = STEP))):(self.onset_frames[j+1])])


        self.chromSync[np.isnan(self.chromSync)] = 0
        ## We don't take into account first and last frames
        self.chromSync[:,0] = np.zeros(self.n_bins)
        self.chromSync[:,-1] = np.zeros(self.n_bins)
        self.chromSyncDB = librosa.amplitude_to_db(self.chromSync, ref=np.max)


        # Energy computation
        for t in range(self.n_frames):
            self.energy.append(LA.norm(self.chromSync[:,t])**2)
        self.energy[0]= sys.float_info.epsilon # Pour que la division par 0 ne donne pas une erreur
        self.energy[-1]= sys.float_info.epsilon


    def TracksSpectralAnalyis(self):
        """ Découpe et synchronise les pistes séparées sur les ONSETS, stoque le spectrogramme
        synchronisé en construisant self.chromPistesSync"""

        # if separate tracks are given
        if self.n_pistes != 0:

            # Construction of track spectra
            ChromPistes = []
            for k, voice in enumerate(self.pistes):
                if self.hpss:
                    ChromPistes.append(np.nan_to_num(librosa.decompose.hpss(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)), margin=self.hpss_margin)[0],False))
                else: ChromPistes.append(np.nan_to_num(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)),False))

            # SPECTRUM RELOCALISATION
            ChromPiste_copy = np.copy(ChromPistes)
            for k in range(self.n_pistes):
                for i in range(self.n_bins):
                    for j in reversed(range(self.N)):
                        if j <= self.N_sample[i]: ChromPistes[k][i,j] = ChromPiste_copy[k][i,j]
                        else: ChromPistes[k][i,j] = ChromPiste_copy[k][i,j-int(self.N_sample[i]/2)]

            # SYNCHRONISED SPECTRUM
            for k in range(self.n_pistes):
                self.chromPistesSync.append(np.zeros((self.n_bins,self.n_frames)))

            ## Delay at the beginning and at the end of the synchronisation interval, depending on the frequency
            Δmin = 0.1 # en secondes
            for i in range(self.n_bins):
                f = self.fmin*2**(i/BINS_PER_OCTAVE)
                T_ret = 1.5 / (f * (2**(1.0/(12*4)) - 1))
                for k in range(self.n_pistes):
                    for j in range(self.n_frames):
                        if T_ret < (self.onset_times[j+1] - self.onset_times[j+1]) - Δmin:
                            self.chromPistesSync[k][i,j] = np.median(ChromPistes[k][i][(self.onset_frames[j]+int(librosa.time_to_frames(T_ret, sr=self.sr, hop_length = STEP))):(self.onset_frames[j+1])])
                        else:
                            self.chromPistesSync[k][i,j] = np.median(ChromPistes[k][i][(self.onset_frames[j+1]-int(librosa.time_to_frames(Δmin, sr=self.sr, hop_length = STEP))):(self.onset_frames[j+1])])


            # Tracks energy computation
            self.energyPistes = np.zeros((self.n_pistes, self.n_frames))
            for t in range(self.n_frames):
                for k in range(self.n_pistes):
                    self.energyPistes[k,t] = np.sum(np.multiply(self.chromPistesSync[k][:,t], self.chromPistesSync[k][:,t]))


            # ACTIVATION MATRICE : this matrice indicates which tracks contain a note on a given frame
            # 1 indicates a note, 0 no note. Under a certain threshold of the track's maximum energy, the track is considered silent.
            self.activation = np.ones((self.n_pistes, self.n_frames))
            max_energy = np.amax(self.energyPistes, axis = 1)
            for k in range(self.n_pistes):
                for t in range(self.n_frames):
                    if (self.energyPistes[k,t] < seuil_activation * max_energy[k]):
                        self.activation[k,t] = 0
                        self.chromPistesSync[k][:,t] = 0
            self.activation[:,0] = 0
            self.activation[:,self.n_frames-1] = 0


            # Number of notes
            self.n_notes = np.sum(self.activation, axis=0)
            self.n_notes[0] = 0
            self.n_notes[-1] = 0



    def SimplifySpectrum(self):
        self.chromSyncSimpl = np.zeros(self.chromSync.shape)
        self.chromPistesSyncSimpl= np.copy(self.chromPistesSync)
        δ = 2 #int, en subdivisions spectrales
        if self.n_pistes >= 2: # Si plusieurs pistes
            for t in range(self.n_frames):
                for i in range(10, self.n_bins - 10):
                    for p in range(len(self.pistes)):
                        if self.chromPistesSync[p][i,t] < np.max(self.chromPistesSync[p][i-δ:i+δ+1,t]): self.chromPistesSyncSimpl[p][i,t] = 0
                        else: self.chromPistesSyncSimpl[p][i,t] = np.sum(self.chromPistesSync[p][i-δ:i+δ+1,t])
                    # Global
                    if self.chromSync[i,t] < np.max(self.chromSync[i-δ:i+δ+1,t]): self.chromSyncSimpl[i,t] = 0
                    else: self.chromSyncSimpl[i,t] = np.sum(self.chromSync[i-δ:i+δ+1,t])


        else: # Si une seule piste (configuration enregistrement)
            for t in range(self.n_frames):
                for i in range(10, self.n_bins - 10):
                    # Global
                    if self.chromSync[i,t] < np.max(self.chromSync[i-δ:i+δ+1,t]): self.chromSyncSimpl[i,t] = 0
                    else: self.chromSyncSimpl[i,t] = np.sum(self.chromSync[i-δ:i+δ+1,t])
    #
    #
        # Liste des partiels de self.chromSyncSimpl
        self.liste_partials = []
        for t in range(self.n_frames):
            self.liste_partials.append([])
            for k in range(self.n_bins):
                if self.chromSyncSimpl[k,t] > 0: self.liste_partials[t].append(k)


    def Concordance(self):
        """Multiplie les spectres (cqt) des différentes pistes pour créer le spectre de concordance,
        et calcule la concordance en sommant sur les fréquences"""

        self.chrom_concordance = np.zeros((self.n_bins,self.n_frames))
        for k in range(self.n_pistes-1):
            for l in range(k+1, self.n_pistes):
                self.chrom_concordance += np.multiply(self.chromPistesSync[k], self.chromPistesSync[l])

        # Normalisation par l'énergie et par le nombre de notes
        for t in range(self.n_frames):
            if self.n_notes[t] >= 2:
                self.chrom_concordance[:,t] *= (self.n_notes[t]**2/(self.n_notes[t]*(self.n_notes[t]-1)/2.)) / self.energy[t]

        self.chrom_concordance[:,0] = 0
        self.chrom_concordance[:,self.n_frames-1] = 0
        self.concordance = self.chrom_concordance.sum(axis=0)
        self.concordance = self.concordance[1:-1]



    def ConcordanceTot(self):
        """Multiplie les spectres (cqt) des différentes pistes pour créer le spectre de concordance,
        et calcule la concordance en sommant sur les fréquences"""

        self.chrom_concordanceTot = np.ones((self.n_bins,self.n_frames))
        for t in range(self.n_frames):
            for k in range(self.n_pistes):
                if self.activation[k,t]:
                    self.chrom_concordanceTot[:,t] = np.multiply(self.chrom_concordanceTot[:,t], self.chromPistesSync[k][:,t])
            if self.n_notes[t]>=1:
                self.chrom_concordanceTot[:,t] = np.divide((self.n_notes[t]**self.n_notes[t]) * self.chrom_concordanceTot[:,t], LA.norm(self.chromSync[:,t], self.n_notes[t])**self.n_notes[t])
            self.concordanceTot.append(self.chrom_concordanceTot[:,t].sum(axis=0))#**(1./self.n_notes[t]))


        self.chrom_concordanceTot[:,0] = 0
        self.chrom_concordanceTot[:,self.n_frames-1] = 0
        self.concordanceTot = self.concordanceTot[1:-1]


    def Concordance3(self):
        self.chrom_concordance3 = np.zeros((self.n_bins,self.n_frames))
        for k in range(self.n_pistes-2):
            for l in range(k+1, self.n_pistes-1):
                for m in range(l+1, self.n_pistes):
                    self.chrom_concordance3 += np.multiply(np.multiply(self.chromPistesSync[k], self.chromPistesSync[l]), self.chromPistesSync[m])

        # Normalisation par la norme 3 et le nombre de notes
        for t in range(self.n_frames):
            if self.n_notes[t] >= 3:
                self.chrom_concordance3[:,t] *= (self.n_notes[t]**3/(self.n_notes[t]*(self.n_notes[t]-1)*(self.n_notes[t]-2)/6)) / LA.norm(self.chromSync[:,t],ord=3)**3
        self.chrom_concordance3[:,0] = 0
        self.chrom_concordance3[:,self.n_frames-1] = 0
        self.concordance3 = self.chrom_concordance3.sum(axis=0)
        self.concordance3 = self.concordance3[1:-1]


    def Roughness(self):

        self.chrom_roughness = np.zeros((self.n_bins,self.n_frames))
        self.roughness = np.zeros(self.n_frames)
        β1 = 3.5
        β2 = 5.75

        for b1 in range(self.n_bins-1):
            for b2 in range(b1+1,self.n_bins):
                # Modèle de Sethares
                f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                freq = [f1, f2]
                freq.sort()
                # Roughness calculation, chose between two models. By default, Sethares implementation
                s = 0.24/(0.021*freq[0] + 19)
                rug = np.exp(-β1*s*(freq[1]-freq[0]))-np.exp(-β2*s*(freq[1]-freq[0]))

                if not type_rug_signal:
                    for p1 in range(self.n_pistes-1):
                        for p2 in range(p1+1, self.n_pistes):
                            if rug_simpl:
                                self.chrom_roughness[b1] += (self.chromPistesSyncSimpl[p1][b1] * self.chromPistesSyncSimpl[p2][b2] + self.chromPistesSyncSimpl[p1][b2] * self.chromPistesSyncSimpl[p2][b1]) * rug/2
                                self.chrom_roughness[b2] += (self.chromPistesSyncSimpl[p1][b1] * self.chromPistesSyncSimpl[p2][b2] + self.chromPistesSyncSimpl[p1][b2] * self.chromPistesSyncSimpl[p2][b1]) * rug/2
                            else:
                                self.chrom_roughness[b1] += (self.chromPistesSync[p1][b1] * self.chromPistesSync[p2][b2] + self.chromPistesSync[p1][b2] * self.chromPistesSync[p2][b1]) * rug/2
                                self.chrom_roughness[b2] += (self.chromPistesSync[p1][b1] * self.chromPistesSync[p2][b2] + self.chromPistesSync[p1][b2] * self.chromPistesSync[p2][b1]) * rug/2
                else:
                    if rug_simpl:
                        self.chrom_roughness[b1] += (self.chromSyncSimpl[b1] * self.chromSyncSimpl[b2]) * rug/2
                        self.chrom_roughness[b2] += (self.chromSyncSimpl[b1] * self.chromSyncSimpl[b2]) * rug/2
                    else:
                        self.chrom_roughness[b1] += (self.chromSync[b1] * self.chromSync[b2]) * rug/2
                        self.chrom_roughness[b2] += (self.chromSync[b1] * self.chromSync[b2]) * rug/2


        # Normalisation par l'énergie et par le nombre de n_notes
        for t in range(self.n_frames):
            if not type_rug_signal:
                if self.n_notes[t] >= 2:
                    self.chrom_roughness[:,t] *= (self.n_notes[t]**2 / (self.n_notes[t]*(self.n_notes[t]-1)/2.0)) / self.energy[t]
            else:
                self.chrom_roughness[:,t] /= self.energy[t]
        self.chrom_roughness[:,0] = 0
        self.roughness = self.chrom_roughness.sum(axis=0)
        self.roughness=self.roughness[1:-1]




    def Harmonicity(self):
        # HARMONIC SPECTRUM CONSTRUCTION
        dec = BINS_PER_OCTAVE/6 # décalage d'un ton pour tenir compte de l'épaisseur des gaussiennes
        epaiss = int(np.rint(BINS_PER_OCTAVE/(2*σ)))
        SpecHarm = np.zeros(2*int(dec) + int(np.rint(BINS_PER_OCTAVE * np.log2(κ))))
        for k in range(κ):
            pic =  int(dec + np.rint(BINS_PER_OCTAVE * np.log2(k+1)))
            for i in range(-epaiss, epaiss+1):
                SpecHarm[pic + i] = 1/(k+1)**decr
        len_corr = self.n_bins + len(SpecHarm) - 1

        # CORRELATION WITH REAL SPECTRUM
        self.chrom_harmonicity = np.zeros((len_corr,self.n_frames))
        self.harmonicity = []
        norm_harmonicity = 1
        for t in range(self.n_frames):
            self.chrom_harmonicity[:,t] = np.correlate(np.power(self.chromSync[:,t],norm_harmonicity), SpecHarm,"full") / self.energy[t]**(norm_harmonicity/2.)
            self.harmonicity.append(np.exp(max(self.chrom_harmonicity[:,t])))
            # Virtual Pitch
            self.virtualPitch.append(np.argmax(self.chrom_harmonicity[:,t]))

        virtualNotes = librosa.hz_to_note([self.fmin * (2**((i-len(SpecHarm)+dec+1)/BINS_PER_OCTAVE)) for i in self.virtualPitch] , cents = False)
        global f_corr_min
        f_corr_min =  self.fmin * (2**((-len(SpecHarm)+dec+1)/BINS_PER_OCTAVE))
        print(virtualNotes[1:self.n_frames-1])
        self.chrom_harmonicity[:,0] = 0
        self.chrom_harmonicity[:,self.n_frames-1] = 0
        self.harmonicity = self.harmonicity[1:-1]



    def HarmonicChange(self):
        self.chrom_harmonicChange = np.zeros((self.n_bins,self.n_frames-1))
        for t in range(self.n_frames-1):
            self.chrom_harmonicChange[:,t] = (self.chromSync[:,t+1] - self.chromSync[:,t]) / (self.energy[t+1]*self.energy[t])**(1.0/4)

        self.harmonicChange = np.sum(np.power(np.abs(self.chrom_harmonicChange),1), axis=0)
        self.harmonicChange = self.harmonicChange[1:-1]


    def DiffConcordance(self):
        self.chrom_diffConcordance = np.zeros((self.n_bins,self.n_frames-1))

        for t in range(self.n_frames-1):
            self.chrom_diffConcordance[:,t] = np.multiply(self.chromSync[:,t], self.chromSync[:,t+1])
            self.chrom_diffConcordance[:,t] /= np.sqrt(self.energy[t] * self.energy[t+1])

        self.diffConcordance = self.chrom_diffConcordance.sum(axis=0)
        self.diffConcordance = self.diffConcordance[1:-1]


    def DiffRoughness(self):
        self.chrom_diffRoughness = np.zeros((self.n_bins,self.n_frames-1))
        β1 = 3.5
        β2 = 5.75

        for b1 in range(self.n_bins):
            for b2 in range(self.n_bins):
                f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                freq = [f1, f2]
                freq.sort()
                s = 0.24/(0.021*freq[0] + 19)
                rug = np.exp(-β1*s*(freq[1]-freq[0]))-np.exp(-β2*s*(freq[1]-freq[0]))


                for t in range(self.n_frames-1):
                    if rug_simpl:
                        self.chrom_diffRoughness[b1,t] += (self.chromSyncSimpl[b1,t] * self.chromSyncSimpl[b2,t+1]) * rug / 2
                        self.chrom_diffRoughness[b2,t] += (self.chromSyncSimpl[b1,t] * self.chromSyncSimpl[b2,t+1]) * rug / 2
                    else:
                        self.chrom_diffRoughness[b1,t] += (self.chromSync[b1,t] * self.chromSync[b2,t+1]) * rug / 2
                        self.chrom_diffRoughness[b2,t] += (self.chromSync[b1,t] * self.chromSync[b2,t+1]) * rug / 2


        for t in range(self.n_frames-1):
            self.chrom_diffRoughness[:,t] = np.divide(self.chrom_diffRoughness[:,t], np.sqrt(self.energy[t]*self.energy[t+1]))


        self.diffRoughness = self.chrom_diffRoughness.sum(axis=0)
        self.diffRoughness = self.diffRoughness[1:-1]
    #
    #
    #
    def ComputeDescripteurs(self, space = ['roughness']):
        """Compute descriptors present in 'space'."""

        if 'concordance' in space: self.Concordance()
        if 'concordanceTot' in space: self.ConcordanceTot()
        if 'roughness' in space: self.Roughness()
        if 'harmonicity' in space: self.Harmonicity()
        if 'harmonicChange' in space: self.HarmonicChange()
        if 'diffConcordance' in space: self.DiffConcordance()
        if 'diffRoughness' in space: self.DiffRoughness()


# ##########################################################################################################################################################################
# ##########################################################################################################################################################################


# Interface et Application

# external_stylesheets = ['assets/bWLwgP.css']
# external_stylesheets=[dbc.themes.BOOTSTRAP]
# external_stylesheets=[dbc.themes.BOOTSTRAP, "assets/segmentation-style.css"]
external_stylesheets=[dbc.themes.QUARTZ]
# external_stylesheets=[dbc.themes.SUPERHERO]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title='Harmonic Descriptors Implementation'


# Inputs
input = dbc.Card(
    id='input_box',
    children=[
        dbc.CardHeader(html.H5("Inputs")),
        dbc.CardBody([
            dbc.Tabs(id='tabs', children=[]),
            dbc.Row([
                dbc.Col(dbc.Button(id='add_audio_input', n_clicks=0, children='Add Audio input'), width = 2),
                dbc.Col(dbc.Button(id='delete_audio_input', n_clicks=0, children='Delete Audio input'), width = 2)
            ], justify = 'start')
        ])
    ]
)


signal_box = dbc.Card(
    id="signal_box",
    children=[
        dbc.CardHeader(html.H5("Signal Processing Parameters")),
        dbc.CardBody([
            html.Details([
                html.Summary('Advanced settings'),
                dbc.Col(
                    id='parameters_analysis',
                    children=[
                        html.Br(),

                        dbc.Row([
                            dbc.Col([
                                html.Div('Minimal pitch'),
                                dbc.Select(
                                    id='notemin',
                                    options=[{'label': note, 'value': note} for note in librosa.midi_to_note(range(12,132))],
                                    value='C1',
                                )
                            ], width = 4),
                            dbc.Col([
                                html.Div('Maximal pitch'),
                                dbc.Select(
                                    id='notemax',
                                    options=[{'label': note, 'value': note} for note in librosa.midi_to_note(range(12,132))],
                                    value='C9',
                                )
                            ], width = 4),

                        ]),

                        dbc.Row([
                            dbc.Col([
                                html.Div([html.A('More details', href='https://librosa.org/doc/main/generated/librosa.note_to_hz.html#', target='_blank')]),
                                html.Br(),

                                html.Div('Window shape'),
                                dbc.Select(
                                    id='window',
                                    options=[{'label': fun[0].upper()+fun[1:], 'value': fun} for fun in list_window],
                                    value='hann',
                                    size = 'lg',
                                    style={'width':'50%'}
                                ),
                                html.Div([html.A('More details', href='https://docs.scipy.org/doc/scipy/reference/signal.windows.html', target='_blank')]),
                                html.Br(),
                                dbc.Switch(id='hpss', value=False,label = 'Reduction of percussive part'),
                                html.Br(),
                                html.Div(
                                    id = 'hpss_show',
                                    children=[
                                        html.Div("Margin"),
                                        dcc.Slider(
                                            id='hpss_margin',
                                            min=1, max=20, step=1, value=1,
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ],
                                ),
                                html.Div([html.A('More details', href='https://librosa.org/doc/main/generated/librosa.decompose.hpss.html', target='_blank')]),
                            ])
                        ])

                    ]
                ),
                # dcc.Store(id='mem_window'),
                # dcc.Store(id='mem_hpss'),
                # dcc.Store(id='mem_hpss_margin')
            ])
        ])
]
)

descriptors_box = dbc.Card(
    id= 'descriptors_box',
    children=[
        dbc.CardHeader(html.H5("Descriptors")),
        dbc.CardBody([
            html.Div(
                id='compute',
                children=[
                    dbc.RadioItems(
                        id='compute_type',
                        options=[
                            {'label': 'Static descriptors', 'value': 'static'},
                            {'label': 'Dynamic descriptors', 'value': 'dynamic'},
                            {'label': 'All descriptors', 'value': 'all_descriptors'}
                        ],
                        value='static'
                    ),
                    html.Br(),
                    dbc.Row(
                        dbc.Col(dbc.Button(id='compute_button', n_clicks=0, children='Compute descriptors', size='lg'), width=2),
                        justify='around'
                    )
                ]
            )
        ])
    ]
)


visualisation_option_box = dbc.Card(
    id= 'visualisation_option_box',
    children=[
        dbc.CardHeader(html.H5("Visualisation options")),
        dbc.CardBody([
            html.Div(
                id='vis_options',
                children=[
                    dbc.Switch(id='vis_trajectories', value=True, label = 'Visualise trajectories'),
                    html.Br(),
                    html.Div(
                        id='vis_descr_type',
                        children=[
                            html.Big('Type of descriptors'),
                            dbc.RadioItems(
                                id='vis_descr_type_radio',
                                options=[
                                    {'label':'Static','value':'static'},
                                    {'label':'Dynamic','value':'dynamic'}
                                ],
                                value='static',
                                labelStyle={'display':'inline-block'}
                            )
                        ],
                        style={'display':'none'}
                    ),
                    html.Div(
                        id='vis_descr',
                        children=[
                            dbc.Select(
                                id='selected_descr1',
                                options=[{'label': descrNames[descr], 'value': descr} for descr in descrList['static']],
                                size = 'lg',
                                placeholder="Select a {} descriptor for axe x".format('static')
                            ),
                            html.Br(),
                            dbc.Select(
                                id='selected_descr2',
                                options=[{'label': descrNames[descr], 'value': descr} for descr in descrList['static']],
                                size = 'lg',
                                placeholder="Select a {} descriptor for axe y".format('static')
                            )
                        ]
                    ),
                    html.Br(),
                    dbc.Switch(id='normalisation', value=False, label = 'Rescale axes for every audio'),
                    html.Br(),
                    html.Details([
                        html.Summary('Advanced options'),
                        dbc.Tabs(id='tabs_visualise', children=[])
                    ])
                ]
            ),
        ])
    ]
)


parameters = [
    signal_box,
    html.Br(),
    descriptors_box,
    html.Br(),
    visualisation_option_box,
]

loadings = [
    dcc.Loading(
        id="loading-1",
        children=html.Div(id='hidden_compute_descr', style={'display':'none'})
    ),
    dcc.Loading(
        id="loading-2",
        children=html.Div(id='hidden_new_class', style={'display':'none'})
    ),
    dcc.Loading(
        id="loading-3",
        children=html.Div(id='hidden_chords_and_trans', style={'display':'none'})
    )

]


graph = [html.Div(id='visualisation'), html.Div(id='hidden_dataframe',style={'display':'none'})]


app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Img(src=app.get_asset_url("iremus-logo.png"), className="logo"),width=3),
            dbc.Col(html.H2("Harmonic descriptors app",style={'color':'white'}), align='center',width=5),
        ], justify='start'),
        dbc.Row(dbc.Col(input)),
        html.Br(),
        dbc.Row(
            id= 'main_row',
            children = [dbc.Col(parameters, md=4), dbc.Col(loadings + graph, md=8)]
        ),
        html.Br(),
    ], fluid=True)
])



# Add or delete audio
@app.callback(
    Output('tabs','children'),
    Output('tabs','active_tab'),
    Output('tabs_visualise','children'),
    Output('tabs_visualise','active_tab'),
    Output('add_audio_input', 'n_clicks'),
    Input('add_audio_input', 'n_clicks'),
    Input('delete_audio_input', 'n_clicks'),
    State('tabs','children'),
    State('tabs_visualise','children'))
def add_del_tab(add_audio, del_audio, children, children_vis):
    ctx = dash.callback_context
    new_tab = dbc.Tab(
        label='Audio {}'.format(add_audio + 1),
        tab_id='Audio {}'.format(add_audio + 1),
        id={'type': 'audio', 'index': add_audio + 1},
        # children='Input audio n°{}'.format(add_audio + 1)
        children=[
            dbc.Container([
                # dbc.Row(Input(placeholder = 'Ici va le nom de input')),
                dbc.Row(
                    [
                        dbc.Col(
                            [html.Br(), html.P('Change the title:')],
                            width = 'auto',
                            align='below',
                            id={'type': 'change_name_bis', 'index': add_audio + 1},
                            style={'display':'none'}
                        ),

                        # dbc.Col([html.Div('hozufzoffougef')]),
                        # dbc.Col([html.Div('hozufzoffougef')]),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Input(id={'type': 'name_audio', 'index': add_audio + 1}, size='lg'),
                            ],
                            width = 3,
                            id={'type': 'change_name', 'index': add_audio + 1},
                            style={'display':'none'},
                            align='center'
                        )
                    ]
                ),
                dbc.Row([
                    dbc.Col([
                        html.Big('Main sound file {}'.format(add_audio + 1), style={'textAlign': 'center'}),
                        dcc.Upload(
                            id={'type': 'main_sound', 'index': add_audio + 1},
                            children=html.Div(['Drag and drop or ', html.A(html.B(html.U('Select File')))]),
                            style={'height': '40px','lineHeight': '30px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px'},
                            multiple=False,
                            filename=''
                        ),
                        html.Div(id={'type': 'input1', 'index': add_audio + 1}),
                        html.Div(id={'type': 'sound', 'index': add_audio + 1})
                    ], width=4),
                    dbc.Col([
                        html.Big('Separated audio tracks'.format(add_audio + 1)),
                        dcc.Upload(
                            id={'type': 'tracks', 'index': add_audio + 1},
                            children=html.Div(['Drag and drop or ', html.A(html.B(html.U('Select File')))]),
                            style={'height': '40px','lineHeight': '30px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px'},
                            # Allow multiple files to be uploaded
                            multiple=True
                        ),
                        html.Div(id={'type': 'input2', 'index': add_audio + 1}),
                    ], width=4),
                    dbc.Col([
                        html.Big('Onsets'.format(add_audio + 1)),
                        dcc.Upload(
                            id={'type': 'onsets', 'index': add_audio + 1},
                            children=html.Div(['Drag and drop or ', html.A(html.B(html.U('Select File')))]),
                            style={'height': '40px','lineHeight': '30px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px'},
                            # Allow multiple files to be uploaded
                            multiple=False
                        ),
                        html.Div(id={'type': 'input3', 'index': add_audio + 1})
                    ], width=4),
                ]),

            ], fluid = True),
        ]
    )

    new_tab_vis = dbc.Tab(
        label='Audio {}'.format(add_audio + 1),
        tab_id='Audio {}'.format(add_audio + 1),
        id={'type': 'audio_chords', 'index': add_audio + 1}
    )

    if add_audio==0 or ctx.triggered[0]['prop_id']=='add_audio_input.n_clicks':
        children.append(new_tab)
        children_vis.append(new_tab_vis)
        return children, 'Audio {}'.format(add_audio + 1), children_vis, 'Audio 1', add_audio,
    else:
        children.pop()
        children_vis.pop()
        return children, 'Audio {}'.format(add_audio), children_vis, 'Audio 1', (add_audio - 1)


# Add or delete audio BIS
@app.callback(
    Output('delete_audio_input','style'),
    Input('add_audio_input', 'n_clicks'))
def cache_button(add_audio):
    if add_audio==0:
        return {'display':'none'}
    else:
        return {'display':'inline-block'}

# Input main_sound
@app.callback(
    Output({'type': 'input1', 'index': MATCH}, 'children'),
    Output({'type': 'sound', 'index': MATCH},'children'),
    Output({'type': 'name_audio', 'index': MATCH},'value'),
    Output({'type': 'change_name', 'index': MATCH},'style'),
    Output({'type': 'change_name_bis', 'index': MATCH},'style'),
    Input({'type': 'main_sound', 'index': MATCH}, 'filename'))
def set_name_main(filename):
    if isinstance(filename, str) and len(filename)>4:
        return html.I(filename), [html.Audio(src='assets/temp.wav', controls=True), html.Br()], filename.split('.')[0], {'display':'inline-block'}, {'display':'inline-block'}
    else:
        return None, None, None, {'display':'none'}, {'display':'none'}


# Input tracks
@app.callback(
    Output({'type': 'input2', 'index': MATCH}, 'children'),
    Input({'type': 'tracks', 'index': MATCH}, 'filename'))
def set_name_input2(filename):
    if filename==None: return ''
    return [html.Div(html.I(name)) for name in filename]

# Input onsets
@app.callback(
    Output({'type': 'input3', 'index': MATCH}, 'children'),
    Input({'type': 'onsets', 'index': MATCH}, 'filename'))
def set_name_input3(filename):
    return html.I(filename)


# Descriptors inputs
@app.callback(
    Output('vis_descr_type', 'style'),
    Input('compute_type', 'value'))
def set_type_descr(value):
    if value=='all_descriptors':
        return {'display':'inline-block'}
    else:
        return {'display':'none'}

@app.callback(
    Output('vis_descr_type_radio', 'value'),
    Input('compute_type', 'value'),
    Input('vis_descr_type_radio', 'value'))
def set_type_descr_2(type, val):
    if type!='all_descriptors':
        return type
    else:
        return val


# Descriptors inputs
@app.callback(
    Output('selected_descr1', 'options'),
    Output('selected_descr2', 'options'),
    Output('selected_descr1', 'placeholder'),
    Output('selected_descr2', 'placeholder'),
    Output('selected_descr1', 'value'),
    Output('selected_descr2', 'value'),
    Input('vis_descr_type_radio', 'value'),
    Input({'type': 'tracks', 'index': ALL},'contents'))
def set_type_descr_3(type_d, list_tracks):
    if type_d=='static' and (list_tracks is None or None in list_tracks):
        type = 'staticOneSound'
        l = [{'label': 'Concordance', 'value': 'concordance', 'disabled': True}, {'label': 'Concordance Totale', 'value': 'concordanceTot', 'disabled': True}]
    else:
        type = type_d
        l=[]

    global space
    space = descrList[type]
    print('Space settled')
    print(space)

    return [{'label': descrNames[descr], 'value': descr} for descr in descrList[type]]+l, [{'label': descrNames[descr], 'value': descr} for descr in descrList[type]]+l, "Select a {} descriptor for axe x".format(type_d), "Select a {} descriptor for axe x".format(type_d), None, None


# Hpss_margin
@app.callback(
    Output('hpss_show','style'),
    Input('hpss', 'value'))
def set_margin(hpss):
    if hpss:
        return {'display':'inline-block','width':'50%'}
    else:
        return {'display':'none','width':'50%'}



# Instanciation de classe
@app.callback(
    Output({'type': 'audio_chords', 'index': ALL}, 'children'),
    Output('hidden_new_class', 'children'),
    Input('compute_button', 'n_clicks'),
    State({'type': 'main_sound', 'index': ALL},'contents'),
    State({'type': 'tracks', 'index': ALL},'contents'),
    State({'type': 'onsets', 'index': ALL},'filename'),
    State('window','value'),
    State('hpss','value'),
    State('hpss_margin','value'),
    State('notemin','value'),
    State('notemax','value'))
def set_class_instance(n_clicks, list_main, list_sep_tracks, list_onsets, window, hpss, hpss_margin, Notemin, Notemax):
    global S
    S=[]
    if n_clicks==0:
        raise PreventUpdate
    else:
        # global S
        ctx = dash.callback_context
        for i, (main, sep_tracks, onsets) in enumerate(zip(list_main, list_sep_tracks, list_onsets)):
            # Calcul duration
            onsets_list = []
            with open('assets/'+onsets,'r') as f:
                for line in f:
                    l = line.split()
                    onsets_list.append(float(l[0]))
            duration = onsets_list[-1]+0.05

            # Load main sound
            content_type1, content_string1 = main.split(",")
            decoded1 = base64.b64decode(content_string1)
            wav_file = open("assets/temp.wav", "wb")
            wav_file.write(decoded1)
            y, sr = librosa.load('assets/temp.wav', sr=None, duration = duration)

            # Load tracks
            if sep_tracks is not None:
                l=[]
                #List of separated tracks
                for k, content in enumerate(sep_tracks):
                    content_type2, content_string2 = content.split(",")
                    decoded2 = base64.b64decode(content_string2)
                    wav_file = open("assets/temp{}.wav".format(k), "wb")
                    wav_file.write(decoded2)
                    y_temp, sr = librosa.load('assets/temp{}.wav'.format(k), sr=None, duration = duration)
                    l.append(y_temp)


            # Instance class
            if sep_tracks is None:
                inst = SignalSepare(y, sr, [], 'assets/'+onsets, window, hpss, hpss_margin, Notemin, Notemax)
            else:
                print(len(l))
                inst = SignalSepare(y, sr, l, 'assets/'+onsets, window, hpss, hpss_margin, Notemin, Notemax)
            # Analyse spectrale et segmentation
            inst.GlobalSpectralAnalyis()
            inst.TracksSpectralAnalyis()

            # Ajout de l'instance de classe à S
            S.append(inst)



        # # Entrée pour les accords et les transitions
        liste_children = [
            [
                dbc.Container(
                    [dbc.Row(dbc.Col([html.Br(), html.Big('List of verticalities (chords):'), html.Br()]))] + [
                        dbc.Row([
                            dbc.Col(
                                dbc.Input(
                                    id={'type': 'name_chord', 'index': i+1, 'temp':k+1}, value='{}'.format(k+1), size='lg'
                                )
                            ),
                            dbc.Col(
                                dbc.RadioItems(
                                    id={'type': 'show_chord', 'index': i+1,'temp': k+1},
                                    options=[
                                        {'label': 'Show', 'value': 1},
                                        {'label': 'Hide', 'value': 0},
                                    ],
                                    value=1,
                                    inline=True
                                )
                            )
                        ])
                        for k in range(S[i].n_frames - 2)
                    ], id={'type': 'liste_chords', 'index': i+1}),

                dbc.Container(
                    [dbc.Row(dbc.Col([html.Br(), html.Big('List of transitions:'), html.Br()]))] + [
                        dbc.Row([
                            dbc.Col(
                                dbc.Input(
                                    id={'type': 'name_trans', 'index': i+1, 'temp':k+1}, value='{}'.format(k+1), size='lg'
                                )
                            ),
                            dbc.Col(
                                dbc.RadioItems(
                                    id={'type': 'show_trans', 'index': i+1,'temp': k+1},
                                    options=[
                                        {'label': 'Show', 'value': 1},
                                        {'label': 'Hide', 'value': 0},
                                    ],
                                    value=1,
                                    inline=True
                                )
                            )
                        ])
                        for k in range(S[i].n_frames - 3)
                    ], id={'type': 'liste_trans', 'index': i+1}),


            ]
            for i in range(len(S))
        ]

        print('Instanciation ok')
        return liste_children, None



# Affichage des accords ou des transitions
@app.callback(
    Output({'type': 'liste_chords', 'index': ALL}, 'style'),
    Output({'type': 'liste_trans', 'index': ALL}, 'style'),
    Output('hidden_chords_and_trans', 'children'),
    Input('vis_descr_type_radio', 'value'),
    Input('hidden_new_class', 'children'),
    State('compute_button', 'n_clicks'))
def name_chords_trans(type, hidden, n_clicks):
    if n_clicks==0:
        raise PreventUpdate
    else:
        global S
        if type=='static':
            displ_stat, displ_dyn = {'display':'inline-block'}, {'display':'none'}
        else:
            displ_stat, displ_dyn = {'display':'none'}, {'display':'inline-block'}
        print('accords_trans')
        return [displ_stat for i in range(len(S))], [displ_dyn for i in range(len(S))], None






# Descriptors computation
@app.callback(
    Output('hidden_compute_descr', 'children'),
    Input('compute_button', 'n_clicks'),
    Input('hidden_new_class', 'children'),
    State('hidden_chords_and_trans', 'children'),
    State('compute_type','value'),
    State({'type': 'tracks', 'index': ALL},'contents'))
def compute_descriptors(n_clicks, hidden1, hidden2, type, list_tracks):
    if n_clicks == 0:
        raise PreventUpdate
    else:
        # List of descriptors to compute
        if type=='static' and None in list_tracks:
            space_compute = descrList['staticOneSound']
        elif type=='all_descriptors' and None in list_tracks:
            space_compute = descrList['dynamic'] + descrList['staticOneSound']
        elif type=='all_descriptors' and None not in list_tracks:
            space_compute = descrList['dynamic'] + descrList['static']
        else:
            space_compute = descrList[type]

        # Do the computation
        global S, simpl
        for i in range(len(S)):
            if simpl: S[i].SimplifySpectrum()
            S[i].ComputeDescripteurs(space = space_compute)
        print('computation')


# Calcul de DataFrame
@app.callback(
    Output('hidden_dataframe', 'children'),
    Input('compute_button', 'n_clicks'),
    Input('vis_descr_type_radio','value'),
    Input({'type': 'name_audio', 'index': ALL}, 'value'),
    Input({'type': 'name_chord', 'index': ALL, 'temp':ALL},'value'),
    Input({'type': 'name_trans', 'index': ALL, 'temp':ALL},'value'),
    Input({'type': 'show_chord', 'index': ALL, 'temp':ALL},'value'),
    Input({'type': 'show_trans', 'index': ALL, 'temp':ALL},'value'),
    Input('selected_descr1', 'options'),
    Input('hidden_compute_descr', 'children'))
def compute_dataframe(n_clicks, type, names_audio, name_chords, name_trans, show_chords, show_trans, hidden1, hidden2):
    global df, S
    if n_clicks == 0:
        df = None
    else:
        # List of descriptors to compute
        global space
        print('Space:')
        print(space)
        frames = []
        # Dataframe of tracks
        j = 0
        for i in range(len(S)):
            dict={}
            for descr in space:
                dict[descr]=getattr(S[i], descr)

            # Number of verticalities or transitions
            L = len(dict[space[0]])
            if type=='static':
                dict['index'] = name_chords[j:j+L]
                dict['show'] = show_chords[j:j+L]
            if type=='dynamic':
                dict['index'] = name_trans[j:j+L]
                dict['show'] = show_trans[j:j+L]
            j = j+L
            dict['audio']=[names_audio[i] for k in range(L)]
            df_track = pd.DataFrame(dict)
            frames.append(df_track)

        # Concatenate
        df = pd.concat(frames)
        print(df)
        print('Dataframe computed')
        return 'Dataframe computed'

# Name_chord
@app.callback(
    Output({'type': 'name_chord', 'index': MATCH, 'temp':MATCH},'disabled'),
    Input({'type': 'show_chord', 'index': MATCH, 'temp':MATCH},'value'))
def disabled_name_chord(value):
    if value==0:
        return True
    else: return False


# Visualisation
@app.callback(
    Output('visualisation','children'),
    Input('compute_button', 'n_clicks'),
    Input('selected_descr1','value'),
    Input('selected_descr2','value'),
    Input('vis_trajectories', 'value'),
    Input('normalisation', 'value'),
    Input('vis_descr', 'children'),
    Input('hidden_dataframe', 'children'),
    Input('hidden_compute_descr', 'children'))
def set_visualisation(n_clicks, descr1, descr2, traj, norm, hidden, hidden2, hidden3):
    if n_clicks is None:
        raise PreventUpdate
    else:
        if (descr1 is not None) and (descr2 is not None):
            global df
            global space

            # Sélection des descripteurs à représenter et normalisation des descripteurs

            if not norm:
                df_norm = df.copy()
                df_norm = df_norm[df_norm['show']==1]
                for descr in space:
                    max = df_norm[descr].max()
                    min = df_norm[descr].min()
                    if (max-min)!= 0:
                        df_norm[descr] = (df_norm[descr] - min) / (max-min)

            else:
                audios = df['audio'].to_list()
                audios = list(set(audios))
                frames = []
                for i in audios:
                    fr = df[(df['audio']==i) & df['show']==1]
                    for descr in space:
                        max = fr[descr].max()
                        min = fr[descr].min()
                        if (max-min)!= 0:
                            fr[descr] = (fr[descr] - min) / (max-min)
                    frames.append(fr)
                df_norm = pd.concat(frames)


            # Représentations

            if traj:
                fig = px.line(
                    df_norm, x=descr1, y=descr2, text='index', color='audio',
                    hover_data={
                        'audio':False,
                        descr1:':.2f',
                        descr2:':.2f'
                    }
                )
            else:
                fig = px.scatter(
                    df_norm, x=descr1, y=descr2, text='index', color='audio',
                    hover_data={
                        'audio':False,
                        descr1:':.2f',
                        descr2:':.2f'
                    }
                )
            fig.update_traces(textposition="bottom right")
            fig.update_xaxes(title_text=descrNames[descr1])
            fig.update_yaxes(title_text=descrNames[descr2])
            fig.update_layout(
                autosize=True,
                width=900,
                height= 700,
                margin=dict(l=0, r=0, b=0, t=0,),
            )

            print('Figure Layout : ')
            print(fig.layout.width)
            print(fig.layout.height)



            return dcc.Graph(
                id='example-graph',
                figure=fig,
                # style={
                #     'width': '600px',
                #     'height': '500px',
                #     'lineHeight': '30px'
                # },
            )





if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)
