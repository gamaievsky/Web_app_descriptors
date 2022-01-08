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
# import dash_bootstrap_components as dbc
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
                        if j <= self.N_sample[i]: ChromPiste[k][i,j] = ChromPiste_copy[k][i,j]
                        else: ChromPiste[k][i,j] = ChromPiste_copy[k][i,j-int(self.N_sample[i]/2)]

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

external_stylesheets = ['assets/bWLwgP.css']
# external_stylesheets=[dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title='Harmonic Descriptors Implementation'



header = html.Div(
    id="app-header",
    children=[
        html.Img(src=app.get_asset_url("iremus-logo.png"), className="logo"),
        html.H1("Harmonic descriptors")
    ]
)
app.layout = html.Div([
        header,
        html.Hr(),

# Inputs
        html.H3("Input files"),
        html.Br(),
        html.Div(id='tabs_input_files'),
        dcc.Tabs(id='tabs', children=[]),
        html.Button(id='add_audio_input', n_clicks=0, children='Add Audio input'),
        html.Button(id='delete_audio_input', n_clicks=0, children='Delete Audio input'),
        html.Hr(),

# Paramètres de l'analyse
        html.Div(
            children=[
                html.H5("Signal Processing Parameters"),
                html.Details([
                    html.Summary('Advanced settings'),
                    html.Div(
                        id='parameters_analysis',
                        children=[
                            html.Hr(),
                            html.Div('Minimal pitch'),
                            dcc.Dropdown(
                                id='notemin',
                                options=[{'label': note, 'value': note} for note in librosa.midi_to_note(range(12,132))],
                                value='C1',
                                style={'width':'50%'}
                            ),
                            html.Div('Maximal pitch'),
                            dcc.Dropdown(
                                id='notemax',
                                options=[{'label': note, 'value': note} for note in librosa.midi_to_note(range(12,132))],
                                value='C9',
                                style={'width':'50%'}
                            ),
                            html.Div([html.A('More details', href='https://librosa.org/doc/main/generated/librosa.note_to_hz.html#', target='_blank')]),
                            html.Br(),

                            html.Div('Window shape'),
                            dcc.Dropdown(
                                id='window',
                                options=[{'label': fun[0].upper()+fun[1:], 'value': fun} for fun in list_window],
                                value='hann',
                                style={'width':'50%'}
                            ),
                            html.Div([html.A('More details', href='https://docs.scipy.org/doc/scipy/reference/signal.windows.html', target='_blank')]),
                            html.Br(),
                            html.Div('Reduction of percussive part'),
                            daq.BooleanSwitch(id='hpss', on=False, style={'float':'left'}),
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
                            html.Hr(),
                        ]
                    ),
                    # dcc.Store(id='mem_window'),
                    # dcc.Store(id='mem_hpss'),
                    # dcc.Store(id='mem_hpss_margin')
                ]),
            ]
        ),

# Paramètres des descripteurs
        html.Div(
            id='parameters_descriptors',
            children=[
                html.H5("Descriptors Parameters"),
                html.Details([
                    html.Summary('Advanced settings'),
                    html.Div('Contents')
                ]),
            ]
        ),
        html.Hr(),

# Calcul des descripteurs
        html.Div(
            id='compute',
            children=[
                html.H3("Descriptors"),
                dcc.RadioItems(
                    id='compute_type',
                    options=[
                        {'label': 'Static descriptors', 'value': 'static'},
                        {'label': 'Dynamic descriptors', 'value': 'dynamic'},
                        {'label': 'All descriptors', 'value': 'all_descriptors'}
                    ],
                    value='static'
                ),
                html.Br(),
                html.Button(id='compute_button', n_clicks=0, children='Compute descriptors'),
                dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=html.Div(id='hidden_compute_descr', style={'display':'none'})
                ),
                dcc.Loading(
                    id="loading-2",
                    type="default",
                    children=html.Div(id='hidden_new_class', style={'display':'none'})
                )
            ]
        ),
        html.Hr(),

# Options de visulisation
        html.Div(
            id='vis_options',
            children=[
                html.H3("Visualisation options"),
                html.Div([
                    html.Span('Visualise Trajectories'),
                    daq.BooleanSwitch(id='vis_trajectories', on=True, style={'float':'left'})
                    ],
                    id='vis_trajectories_display'),
                html.Br(),
                html.Div(
                    id='vis_descr_type',
                    children=[
                        html.Big('Type of descriptors'),
                        dcc.RadioItems(
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
                        dcc.Dropdown(
                            id='selected_descr1',
                            options=[{'label': descrNames[descr], 'value': descr} for descr in descrList['static']],
                            placeholder="Select a {} descriptor for axe x".format('static'),
                            style={'width':'50%'}
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id='selected_descr2',
                            options=[{'label': descrNames[descr], 'value': descr} for descr in descrList['static']],
                            placeholder="Select a {} descriptor for axe y".format('static'),
                            style={'width':'50%'}
                        )
                    ]
                ),
                html.Br(),
                html.Details([
                    html.Summary('Advanced options'),
                    dcc.Tabs(id='tabs_visualise', children=[])

                    # html.Div([
                    #     html.Div([
                    #         dcc.RadioItems(
                    #             id={'type': 'show', 'index': i}
                    #             options=[
                    #                 {'label': 'Show', 'value': 'show'},
                    #                 {'label': 'Hide', 'value': 'hide'},
                    #             ],
                    #             value='show',
                    #             labelStyle={'display': 'inline-block'}
                    #         ),
                    #         dcc.Input(
                    #             id={'type': 'name_elt', 'index': i},
                    #         )
                    #         for i in range
                    #     ])
                    # ])
                ])
            ]
        ),
        html.Div(id='hidden_dataframe',style={'display':'none'}),#, style={‘display’:‘none’})
        dcc.Store(id='dic_chords_trans',),
        html.Hr(),
        html.Div(id='visualisation')
    ]
)

# Add or delete audio
@app.callback(
    Output('tabs','children'),
    Output('tabs','value'),
    Output('tabs_visualise','children'),
    Output('tabs_visualise','value'),
    Output('add_audio_input', 'n_clicks'),
    Input('add_audio_input', 'n_clicks'),
    Input('delete_audio_input', 'n_clicks'),
    State('tabs','children'),
    State('tabs_visualise','children'))
def add_del_tab(add_audio, del_audio, children, children_vis):
    ctx = dash.callback_context
    new_tab = dcc.Tab(
        label='Audio {}'.format(add_audio + 1),
        value='Audio {}'.format(add_audio + 1),
        id={'type': 'audio', 'index': add_audio + 1},
        # children='Input audio n°{}'.format(add_audio + 1)
        children=[
            html.Div([
                html.Div(id={'type': 'change_name', 'index': add_audio + 1}, style={'display':'none'}, children=[
                    html.Div('Change the title:'),
                    dcc.Input(id={'type': 'name', 'index': add_audio + 1}, type='text')]),
                html.P(),
                html.Big('Main sound file {}'.format(add_audio + 1)),
                dcc.Upload(
                    id={'type': 'main_sound', 'index': add_audio + 1},
                    children=html.Div(['Drag and drop or ', html.A('Select File')]),
                    style={'width': '30%','height': '40px','lineHeight': '30px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px','display': 'inline-block'},

                    multiple=False,
                    filename=''
                ),
                html.Div(id={'type': 'input1', 'index': add_audio + 1})
            ]),

            html.Div([
                html.Big('Separated audio tracks'),
                dcc.Upload(
                    id={'type': 'tracks', 'index': add_audio + 1},
                    children=html.Div(['Drag and drop or ', html.A('Select File')]),
                    style={'width': '30%','height': '40px','lineHeight': '30px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px','display': 'inline-block'},
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.Div(id={'type': 'input2', 'index': add_audio + 1})
            ]),

            html.Div([
                html.Big('Onsets'),
                dcc.Upload(
                    id={'type': 'onsets', 'index': add_audio + 1},
                    children=html.Div(['Drag and drop or ', html.A('Select File')]),
                    style={'width': '30%','height': '40px','lineHeight': '30px','borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px','textAlign': 'center','margin': '10px','display': 'inline-block'},
                    multiple=False
                ),
                html.Div(id={'type': 'input3', 'index': add_audio + 1})
            ]),
            html.Div(id={'type': 'sound', 'index': add_audio + 1})
        ]
    )

    new_tab_vis = dcc.Tab(
        label='Audio {}'.format(add_audio + 1),
        value='Audio {}'.format(add_audio + 1),
        id={'type': 'audio_chords', 'index': add_audio + 1},
        children = []
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
    Output({'type': 'name', 'index': MATCH},'value'),
    Output({'type': 'change_name', 'index': MATCH},'style'),
    Input({'type': 'main_sound', 'index': MATCH}, 'filename'))
def set_name_main(filename):
    if isinstance(filename, str) and len(filename)>4:
        return html.I(filename), html.Audio(src='assets/temp.wav', controls=True),filename.split('.')[0], {'display':'inline-block'}
    else:
        return None, None, None, {'display':'none'}


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

    return [{'label': descrNames[descr], 'value': descr} for descr in descrList[type]]+l, [{'label': descrNames[descr], 'value': descr} for descr in descrList[type]]+l, "Select a {} descriptor for axe x".format(type_d), "Select a {} descriptor for axe x".format(type_d), None, None


# Hpss_margin
@app.callback(
    Output('hpss_show','style'),
    Input('hpss', 'on'))
def set_margin(hpss):
    if hpss:
        return {'display':'inline-block','width':'50%'}
    else:
        return {'display':'none','width':'50%'}



# Instanciation de classe
@app.callback(
    Output('dic_chords_trans','data'),
    Output('hidden_new_class', 'children'),
    Input('compute_button', 'n_clicks'),
    State({'type': 'main_sound', 'index': ALL},'contents'),
    State({'type': 'tracks', 'index': ALL},'contents'),
    State({'type': 'onsets', 'index': ALL},'filename'),
    State('window','value'),
    State('hpss','on'),
    State('hpss_margin','value'),
    State('notemin','value'),
    State('notemax','value'))
def set_class_instance(n_clicks, list_main, list_sep_tracks, list_onsets, window, hpss, hpss_margin, Notemin, Notemax):
    if n_clicks is None:
        raise PreventUpdate
    else:
        global S
        S = []
        dic={'static':{}, 'dynamic':{}}
        # dic={'static':{}, 'dynamic':{}}
        for i, (main, sep_tracks,onsets) in enumerate(zip(list_main, list_sep_tracks, list_onsets)):
            # Load main sound
            content_type1, content_string1 = main.split(",")
            decoded1 = base64.b64decode(content_string1)
            wav_file = open("assets/temp.wav", "wb")
            wav_file.write(decoded1)
            y, sr = librosa.load('assets/temp.wav', sr=None)

            # Load tracks
            if sep_tracks is not None:
                k=0 #Number of separated tracks
                l=[] #List of separated tracks
                for content in sep_tracks:
                    k+=1
                    content_type2, content_string2 = content.split(",")
                    decoded2 = base64.b64decode(content_string2)
                    wav_file = open("assets/temp{}.wav".format(k), "wb")
                    wav_file.write(decoded2)
                    y_temp, sr = librosa.load('assets/temp{}.wav'.format(i), sr=None)


            # Instance class
            if sep_tracks is None:
                inst = SignalSepare(y, sr, [], 'assets/'+onsets, window, hpss, hpss_margin, Notemin, Notemax,)
            else:
                inst = SignalSepare(y, sr, l, 'assets/'+onsets, window, hpss, hpss_margin, Notemin, Notemax,)
            # Analyse spectrale et segmentation
            inst.GlobalSpectralAnalyis()
            inst.TracksSpectralAnalyis()
            # Ajout des noms au dictionnaire
            print('nombre frames : {}'.format(inst.n_frames))
            dic['static']['Audio {}'.format(i+1)] = [k for k in range(1, inst.n_frames - 1)]
            dic['dynamic']['Audio {}'.format(i+1)] = [k for k in range(1, inst.n_frames - 1)]
            # Ajout de l'instance de classe à S
            S.append(inst)
        print(dic)


        return dic, None





# Descriptors computation
@app.callback(
    Output('hidden_compute_descr', 'children'),
    Input('compute_button', 'n_clicks'),
    Input('hidden_new_class', 'children'),
    State('compute_type','value'),
    State({'type': 'tracks', 'index': ALL},'contents'))
def compute_descriptors(n_clicks, newClass, type, list_tracks):
    global mem_space
    if n_clicks == 0:
        mem_space = None
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
        mem_space = space_compute

        # Do the computation
        global S, simpl
        for i in range(len(S)):
            if simpl: S[i].SimplifySpectrum()
            S[i].ComputeDescripteurs(space = space_compute)


# Calcul de DataFrame
@app.callback(
    Output('hidden_dataframe', 'children'),
    Input('compute_button', 'n_clicks'),
    Input('vis_descr_type_radio','value'),
    Input({'type': 'name', 'index': ALL}, 'value'),
    Input('hidden_compute_descr', 'children'))
def compute_dataframe(n_clicks, type, names, hidden):
    global df, S
    if n_clicks == 0:
        df = None
    else:
        # List of descriptors to compute
        global space
        frames = []
        # Dataframe of tracks
        for i in range(len(S)):
            dict={}
            for descr in space:
                dict[descr]=getattr(S[i], descr)
            # Number of verticalities or transitions
            L = len(dict[space[0]])
            dict['index']=range(1, L+1)
            dict['audio']=[names[i] for k in range(L)]
            df_track = pd.DataFrame(dict)
            frames.append(df_track)
        # Concatenate
        df = pd.concat(frames)
        print(df)
        return 'Dataframe computed'



# # Visualisation
@app.callback(
    Output('visualisation','children'),
    Input('compute_button', 'n_clicks'),
    Input('selected_descr1','value'),
    Input('selected_descr2','value'),
    Input('vis_trajectories', 'on'),
    Input('vis_descr', 'children'),
    Input('hidden_dataframe', 'children'),
    Input('hidden_compute_descr', 'children'))
def set_visualisation(n_clicks, descr1, descr2, traj, hidden, hidden2, hidden3):
    if n_clicks is None:
        raise PreventUpdate
    else:
        if (descr1 is not None) and (descr2 is not None):
            global df
            if traj:
                fig = px.line(df, x=descr1, y=descr2, text='index', color='audio')
            else:
                fig = px.scatter(df, x=descr1, y=descr2, text='index', color='audio')
            fig.update_traces(textposition="bottom right")
            fig.update_xaxes(title_text=descrNames[descr1])
            fig.update_yaxes(title_text=descrNames[descr2])

            return dcc.Graph(
                id='example-graph',
                figure=fig,
                style={
                    'width': '600px',
                    'height': '500px',
                    'lineHeight': '30px'
                },
            )





if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)
