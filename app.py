import base64
import datetime
import io
import sys
import os
import json

import dash
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_table
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
descrList = {'static': ['roughness', 'harmonicity', 'concordance', 'concordanceTot'], 'dynamic': ['harmonicChange', 'diffConcordance', 'diffRoughness']}


WINDOW = np.hanning
BINS_PER_OCTAVE = 12*8
FILTER_SCALE = 1
STEP = 512
cmap = 'gray_r'
decompo_hpss = True
margin = 10
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

    def __init__(self, signal, sr, pistes, onsets_txt, Notemin  = 'D3', Notemax = 'D9'):
        self.y = signal
        self.pistes = pistes
        self.sr = sr
        self.n_pistes = len(pistes)
        self.Notemin = Notemin
        self.Notemax = Notemax
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
        self.Chrom = np.abs(librosa.cqt(y=self.y, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins, window=WINDOW, filter_scale = FILTER_SCALE))
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
        if decompo_hpss:
            self.ChromNoHpss = np.copy(self.Chrom)
            self.Chrom = librosa.decompose.hpss(self.Chrom, margin=margin)[0]

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
                if decompo_hpss:
                    ChromPistes.append(np.nan_to_num(librosa.decompose.hpss(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)), margin=margin)[0],False))
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
        self.concordanceTot[0]=0
        self.concordanceTot[self.n_frames-1]=0


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
        self.roughness[0]=0
        self.roughness[self.n_frames-1]=0



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
        self.harmonicity[0]=0
        self.harmonicity[self.n_frames-1]=0



    def HarmonicChange(self):
        self.chrom_harmonicChange = np.zeros((self.n_bins,self.n_frames-1))
        for t in range(self.n_frames-1):
            self.chrom_harmonicChange[:,t] = (self.chromSync[:,t+1] - self.chromSync[:,t]) / (self.energy[t+1]*self.energy[t])**(1.0/4)

        self.harmonicChange = np.sum(np.power(np.abs(self.chrom_harmonicChange),1), axis=0)
        self.harmonicChange[0]=0
        self.harmonicChange[-1]=0


    def DiffConcordance(self):
        self.chrom_diffConcordance = np.zeros((self.n_bins,self.n_frames-1))

        for t in range(self.n_frames-1):
            self.chrom_diffConcordance[:,t] = np.multiply(self.chromSync[:,t], self.chromSync[:,t+1])
            self.chrom_diffConcordance[:,t] /= np.sqrt(self.energy[t] * self.energy[t+1])

        self.diffConcordance = self.chrom_diffConcordance.sum(axis=0)
        self.diffConcordance[0]=0
        self.diffConcordance[self.n_frames-2]=0


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
        self.diffRoughness[0]=0
        self.diffRoughness[self.n_frames-2]=0
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





    def Affichage(self, space = ['concordance', 'concordanceTot'], begin = "first", end = "last", vis_type='temporal', traj=False, color_abstr_numbers = 'black', color_abstr='b'):

        # Horizontal position of chords
        self.onset_times_graph = self.onset_times



        # if parametres.plot_onsets:
        #     # fig, ax = plt.subplots(figsize=(13, 7))
        #     # # Partition
        #     # if parametres.plot_score & (len(self.score)!=0):
        #     #     img=mpimg.imread(self.score)
        #     #     score = plt.subplot(2,1,1)
        #     #     plt.axis('off')
        #     #     score.imshow(img)
        #     #     p = 1
        #     # else: p=0
        #     #
        #     # ax = plt.subplot(p+1,1,p+1)
        #     #
        #     # img = librosa.display.specshow(self.chromSyncDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph, cmap=cmap)
        #     # # img = librosa.display.specshow(librosa.amplitude_to_db(self.Chrom, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph, cmap=cmap)
        #     # # plt.title('Synchronised spectrum, β = {}'.format(parametres.margin))
        #     # plt.title('Synchronised spectrum'.format(parametres.margin))
        #     # for t in self.onset_times_graph:
        #     #     ax.axvline(t, color = 'k',alpha=0.5, ls='--')
        #     # plt.axis('tight')
        #     # ax.get_xaxis().set_visible(False)
        #     # plt.tight_layout()
        #
        #
        #     plt.figure(1,figsize=(13, 7))
        #     ax1 = plt.subplot(3, 1, 1)
        #     librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times,cmap=cmap)
        #     plt.title('CQT spectrogram')
        #
        #     plt.subplot(3, 1, 2, sharex=ax1)
        #     plt.vlines(self.times[self.onset_frames], 0, 1, color='r', alpha=0.9, linestyle='--', label='Onsets')
        #
        #     plt.axis('tight')
        #     plt.legend(frameon=True, framealpha=0.75)
        #
        #     plt.subplot(3, 1, 3, sharex=ax1)
        #     librosa.display.specshow(self.chromSyncDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph, cmap=cmap)
        #     plt.tight_layout()
        #
        # #Plot de la décomposition en partie harmonique / partie percussive
        # if parametres.plot_decompo_hpss & parametres.decompo_hpss:
        #     plt.figure(2,figsize=(13, 7))
        #     plt.subplot(3, 1, 1)
        #     librosa.display.specshow(librosa.amplitude_to_db(self.ChromNoHpss,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
        #     plt.title('Full cqt transform')
        #
        #     plt.subplot(3, 1, 2)
        #     librosa.display.specshow(librosa.amplitude_to_db(self.Chrom,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
        #     plt.title('Harmonic part')
        #
        #     plt.subplot(3, 1, 3)
        #     librosa.display.specshow(librosa.amplitude_to_db(self.ChromNoHpss - self.Chrom,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
        #     plt.title('Percussive part')
        #     plt.tight_layout()
        #
        # #Plot des pistes
        # if parametres.plot_pistes:
        #     plt.figure(3,figsize=(13, 7.5))
        #     ax1 = plt.subplot(self.n_pistes,1,1)
        #     librosa.display.specshow(librosa.amplitude_to_db(self.chromPistesSyncSimpl[0], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph,cmap=cmap)
        #     for k in range(1, self.n_pistes):
        #         plt.subplot(self.n_pistes, 1, k+1, sharex=ax1)
        #         librosa.display.specshow(librosa.amplitude_to_db(self.chromPistesSyncSimpl[k], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph,cmap=cmap)
        #     plt.tight_layout()
        #
        #
        # #Plot des spectres simplifiés
        # if parametres.plot_simple:
        #     fig, ax = plt.subplots(figsize=(13, 7.5))
        #     img = librosa.display.specshow(librosa.amplitude_to_db(self.chromSyncSimpl, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph, cmap=cmap)
        #     plt.title('Partial detection on synchronised spectrum, β = {}, with delay, δ = {}'.format(parametres.margin, parametres.δ))
        #     for t in self.onset_times_graph:
        #         ax.axvline(t, color = 'k',alpha=0.5, ls='--')
        #     plt.axis('tight')
        #     plt.tight_layout()
        #     plt.show()
        #
        # #Plot les spectrogrammes
        # if parametres.plot_chromDescr:
        #
        #     #Construction de la liste des descripteurs avec Chrom
        #     spaceChrom = []
        #     for descr in space:
        #         if descr in ['concordance','concordance3','concordanceTot','roughness','harmonicChange','diffConcordance']: spaceChrom.append(descr)
        #
        #
        #     dimChrom = len(space)
        #     times_plotChromDyn = [self.onset_times_graph[0]] + [t-0.25 for t in self.onset_times_graph[2:self.n_frames-1]] + [t+0.25 for t in self.onset_times_graph[2:self.n_frames-1]] + [self.onset_times_graph[self.n_frames]]
        #     times_plotChromDyn.sort()
        #
        #     plt.figure(5,figsize=(13, 7.5))
        #     # Partition
        #     if parametres.plot_score & (len(self.score)!=0):
        #         #plt.subplot(dim+1+s,1,s)
        #         img=mpimg.imread(self.score)
        #         score = plt.subplot(dimChrom+1,1,1)
        #         plt.axis('off')
        #         score.imshow(img)
        #         plt.title(title +' '+instrument)
        #
        #     else:
        #         ax1 = plt.subplot(dimChrom+1,1,1)
        #         librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times, cmap=cmap)
        #         plt.title(title +' '+instrument)
        #
        #     for k, descr in enumerate(spaceChrom):
        #         if (k==0) & parametres.plot_score & (len(self.score)!=0):
        #             ax1 = plt.subplot(dimChrom+1,1,2)
        #         else: plt.subplot(dimChrom+1, 1, k+2, sharex=ax1)
        #
        #             # Descripteurs statiques
        #         if len(getattr(self, descr)) == self.n_frames:
        #             if descr in ['roughness']:
        #                 librosa.display.specshow(getattr(self, 'chrom_'+descr), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap)
        #             else:
        #                 librosa.display.specshow(librosa.amplitude_to_db(getattr(self, 'chrom_'+descr), ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap)
        #
        #             # Descripteurs dynamiques
        #         else:
        #             Max  = np.amax(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2])
        #             librosa.display.specshow(librosa.amplitude_to_db(np.insert(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2]/Max, range(self.n_frames-2),1, axis=1), ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=np.asarray(times_plotChromDyn), cmap=cmap)
        #         plt.title('Spectre de ' + descr)
        #
        #     plt.tight_layout()
        #
        #Plot les descripteurs harmoniques
        if vis_type=='temporal':
            dim = len(space)
            fig = plt.figure(6,figsize=(13, 7.5))
            ax1 = plt.subplot(dim+1,1,1)
            librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times,cmap=cmap)

            for k, descr in enumerate(space):
                ax = plt.subplot(dim+1, 1, k+2, sharex=ax1)
                ax.get_xaxis().set_visible(False)
                # Je remplace les valeurs nan par 0
                for i,val in enumerate(getattr(self,descr)):
                    if np.isnan(val): getattr(self,descr)[i] = 0

                if len(getattr(self, descr)) == self.n_frames:
                    plt.vlines(self.onset_times_graph[1:self.n_frames], min(getattr(self, descr)), max(getattr(self, descr)[1:(self.n_frames-1)]), color='k', alpha=0.9, linestyle='--')
                else:
                    plt.vlines(self.onset_times_graph[1:self.n_frames-1], min(getattr(self, descr)), max(getattr(self, descr)[1:(self.n_frames-1)]), color='k', alpha=0.9, linestyle='--')
                plt.xlim(self.onset_times_graph[0],self.onset_times_graph[-1])
                if not all(x>=0 for x in getattr(self, descr)[1:(self.n_frames-1)]):
                    plt.hlines(0,self.onset_times_graph[0], self.onset_times_graph[self.n_frames], alpha=0.5, linestyle = ':')


                # Descripteurs statiques
                if len(getattr(self, descr)) == self.n_frames:
                    plt.hlines(getattr(self, descr)[1:(self.n_frames-1)], self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames],color=['b','r','g','c','m','y','b','r','g'][k] , label=descr[0].upper() + descr[1:])
                # Descripteurs dynamiques
                elif len(getattr(self, descr)) == (self.n_frames-1):
                    plt.plot(self.onset_times_graph[2:(self.n_frames-1)], getattr(self, descr)[1:(self.n_frames-2)],['b','r','g','c','m','y','b','r','g'][k]+'o', label=(descr[0].upper() + descr[1:]))
                    plt.hlines(getattr(self, descr)[1:(self.n_frames-2)], [t-0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], [t+0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], color=['b','r','g','c','m','y','b','r','g'][k], alpha=0.9, linestyle=':'  )

                plt.legend(frameon=True, framealpha=0.75)

            plt.tight_layout()

        #Plot descriptogramme + valeur numérique
        # if parametres.plot_OneDescr:
        #
        #     descr = space[0]
        #     plt.figure(7,figsize=(10, 7.5))
        #
        #     ############################
        #
        #     # Partition
        #     if parametres.plot_score & (len(self.score)!=0):
        #         img=mpimg.imread(self.score)
        #         score = plt.subplot(3,1,1)
        #         plt.axis('off')
        #         score.imshow(img)
        #         p = 1
        #     else: p=0
        #
        #     ax1 = plt.subplot(p+2,1,p+1)
        #
        #     # Descripteurs statiques
        #     if len(getattr(self, descr)) == self.n_frames:
        #         if descr in ['concordanceTot', 'concordance3','roughness']:
        #             librosa.display.specshow(getattr(self, 'chrom_'+descr), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap)
        #         elif descr == 'harmonicity':
        #             librosa.display.specshow(np.power(getattr(self, 'chrom_'+descr),4)[0:4*BINS_PER_OCTAVE], bins_per_octave=BINS_PER_OCTAVE, fmin=f_corr_min, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap)
        #         else:
        #             librosa.display.specshow(librosa.amplitude_to_db(getattr(self, 'chrom_'+descr)[0:int(5*self.n_bins/6),:], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap,sr = self.sr)
        #
        #         # Descripteurs dynamiques
        #     else:
        #         times_plotChromDyn = [self.onset_times_graph[0]] + [t-0.75 for t in self.onset_times_graph[2:self.n_frames-1]] + [t+0.75 for t in self.onset_times_graph[2:self.n_frames-1]] + [self.onset_times_graph[self.n_frames]]
        #         times_plotChromDyn.sort()
        #         Max  = np.amax(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2])
        #         librosa.display.specshow(librosa.amplitude_to_db(np.insert(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2]/Max, range(self.n_frames-2),1, axis=1), ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=np.asarray(times_plotChromDyn),cmap=cmap)
        #
        #     for t in self.onset_times_graph:
        #         ax1.axvline(t, color = 'k',alpha=0.5, ls='--')
        #     if descr == 'harmonicity':
        #         plt.title('Virtual pitch spectrum')
        #     else:
        #         # plt.title('DiffRoughness Spectrum')
        #         plt.title(descr[0].upper()+descr[1:]+' spectrum')
        #     plt.xlim(self.onset_times_graph[0],self.onset_times_graph[-1])
        #     ax1.get_xaxis().set_visible(False)
        #
        #
        #     # Plot Descr
        #     ax2 = plt.subplot(p+2, 1, p+2)
        #     if len(getattr(self, descr)) == self.n_frames:
        #         plt.vlines(self.onset_times_graph[1:self.n_frames], min(getattr(self, descr)), max(getattr(self, descr)), color='k', alpha=0.9, linestyle='--')
        #     else:
        #         plt.vlines(self.onset_times_graph[1:self.n_frames-1], min(getattr(self, descr)),max(getattr(self, descr)), color='k', alpha=0.9, linestyle='--')
        #     if not all(x>=0 for x in getattr(self, descr)):
        #         plt.hlines(0,self.onset_times_graph[0], self.onset_times_graph[self.n_frames], alpha=0.5, linestyle = ':')
        #
        #     # Legend
        #
        #     norm = ''
        #     par = ''
        #     if parametres.plot_norm and (descr in parametres.dic_norm ): norm = '\n' + parametres.dic_norm[descr]
        #
        #     if descr in ['harmonicity']:
        #         par = '\n{} partials'.format(parametres.κ)
        #
        #         # Descripteurs statiques
        #     if len(getattr(self, descr)) == self.n_frames:
        #         plt.hlines(getattr(self, descr)[1:(self.n_frames-1)], self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][1], label= descr[0].upper() + descr[1:] + norm + par)
        #         # Descripteurs dynamiques
        #     elif len(getattr(self, descr)) == (self.n_frames-1):
        #         plt.plot(self.onset_times_graph[2:(self.n_frames-1)], getattr(self, descr)[1:(self.n_frames-2)],['b','r','g','c','m','y','b','r','g'][0]+'o')
        #         plt.hlines(getattr(self, descr)[1:(self.n_frames-2)], [t-0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], [t+0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], color=['b','r','g','c','m','y','b','r','g'][0], alpha=0.9, linestyle=':',label = descr[0].upper() + descr[1:] + norm)
        #     plt.xlim(self.onset_times_graph[0],self.onset_times_graph[-1])
        #     # plt.ylim(bottom=0)
        #     ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        #     ax2.get_xaxis().set_visible(False)
        #     plt.legend(frameon=True, framealpha=0.75)
        #     plt.tight_layout()

        # Plot représentations abstraites
        elif vis_type=='abstract':
            if len(space)==2 :
                color = color_abstr
                l1 = getattr(self, space[0])[1:len(getattr(self, space[0]))-1]
                l2 = getattr(self, space[1])[1:len(getattr(self, space[1]))-1]

                #Si un descripteur statique et un descripteur dynamique
                if len(l1)<len(l2) : l2.pop(0)
                elif len(l1)>len(l2) : l1.pop(0)

                #Tronquage
                if isinstance(end,int):
                    l1= l1[0:end]
                    l2= l2[0:end]

                plt.figure(8)
                ax = plt.subplot()
                if traj: plt.plot(l1, l2, color+'--')
                plt.plot(l1, l2, color+'o')
                for i in range(len(l1)):
                    ax.annotate(' {}'.format(i+1), (l1[i], l2[i]), color=color_abstr_numbers)
                plt.xlabel(space[0][0].upper() + space[0][1:])
                plt.ylabel(space[1][0].upper() + space[1][1:])

            else:
                color = color_abstr
                l1 = getattr(self, space[0])[1:len(getattr(self, space[0]))-1]
                l2 = getattr(self, space[1])[1:len(getattr(self, space[0]))-1]
                l3 = getattr(self, space[2])[1:len(getattr(self, space[0]))-1]
                fig = plt.figure(9)
                ax = fig.add_subplot(111, projection='3d')
                if traj: plt.plot(l1, l2, l3, color+'--')
                for i in range(len(l1)):
                    ax.scatter(l1[i], l2[i], l3[i], c=color, marker='o')
                    ax.text(l1[i], l2[i], l3[i], i+1, color=color_abstr_numbers)
                ax.set_xlabel(space[0][0].upper() + space[0][1:])
                ax.set_ylabel(space[1][0].upper() + space[1][1:])
                ax.set_zlabel(space[2][0].upper() + space[2][1:])

        plt.savefig('assets/temp_figure.png')
        print('Figure saved !')
        # plt.show()

#
#
# ##########################################################################################################################################################################
# ##########################################################################################################################################################################

duration = 16.0
Notemin = 'C1'
Notemax = 'B9'
# main_sound = '/Users/manuel/Github/API/Dash/assets/Schnittke_Smith_1.wav'
# onsets_txt = '/Users/manuel/Github/API/Dash/assets/Onset_given_Schnittke_Smith_1.txt'
# # main_sound = 'assets/temp.wav'
# # onsets_txt = 'assets/tempOnsets.txt'
#
# #
# y, sr = librosa.load(main_sound, duration = duration, sr=None)
# S = SignalSepare(y, sr, [], onsets_txt, Notemin, Notemax)
# S.GlobalSpectralAnalyis()
# S.TracksSpectralAnalyis()
#
#
# if simpl: S.SimplifySpectrum()
# space = ['roughness']
# S.ComputeDescripteurs(space)
# S.Affichage(space = space, end = duration)

#
#
#


##########################################################################################################################################################################
##########################################################################################################################################################################

# Interface et Application

external_stylesheets = ['assets/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title='Harmonic Descriptors Implementation'
mem_input1, mem_input2, mem_input3 = None, None, None


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
        html.Div(
            id="input_files",
            children=[
                html.H3("Input files"),
                html.Div([
                    html.Big('Main sound file'),
                    dcc.Upload(
                        id='main_sound',
                        children=html.Div([
                            'Drag and drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '30%',
                            'height': '40px',
                            'lineHeight': '30px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'display': 'inline-block'
                        },
                        multiple=False,
                        filename=''
                    ),
                    html.Div(id='input1')
                ]),

                html.Div([
                    html.Big('Separated audio tracks'),
                    dcc.Upload(
                        id='separated_tracks',
                        children=html.Div([
                            'Drag and drop or ',
                            html.A('Select File')
                        ]),
                        style={
                            'width': '30%',
                            'height': '40px',
                            'lineHeight': '30px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'display': 'inline-block'
                        },
                        # Allow multiple files to be uploaded
                        multiple=True
                    ),
                    html.Div(id='input2')
                ]),

                html.Div([
                    html.Big('Onsets'),
                    dcc.Upload(
                        id='onsets',
                        children=html.Div([
                            'Drag and drop or ',
                            html.A('Select File')
                        ]),
                        style={
                            'width': '30%',
                            'height': '40px',
                            'lineHeight': '30px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'display': 'inline-block'
                        },
                        multiple=False
                    ),
                    html.Div(id='input3')
                ]),
                html.Div(id='sound')
            ]
        ),
        html.Hr(),
        html.Div(
            id='parameters_signal',
            children=[
                html.H3("Signal Processing Parameters"),
                html.Div('Default for now')
            ]
        ),
        html.Hr(),
        html.Div(
            id='parameters_descriptors',
            children=[
                html.H3("Descriptors Parameters"),
                html.Div('Default for now')
            ]
        ),
        html.Hr(),
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
                    value='static',
                    style={}
                ),
                html.Button(id='compute_button', n_clicks=0, children='Compute descriptors'),
                html.Div(id='compute_hidden')
            ]
        ),

        html.Hr(),
        html.Div(
            id='vis_options',
            children=[
                html.H3("Visualisation options"),
                html.Div([
                    html.Span('Visualise Trajectories'),
                    daq.BooleanSwitch(id='vis_trajectories', on=False, style={'float':'left'})
                    ],
                    id='vis_trajectories_display'),
                html.Br(),
                html.Div(
                    id='vis_descr_type',
                ),
                html.Div(
                    id='vis_descr'
                )
            ]
        ),


        html.Hr(),
        html.Div(
            id='visualisation_option',
            children=[
                html.H3("Visualisation Options"),
                html.Big('Type of visualisation'),

                dcc.RadioItems(
                    id='vis_type',
                    options=[
                        {'label':'Temporal','value':'temporal'},
                        {'label':'Abstract','value':'abstract'}
                    ],
                    value='temporal',
                    labelStyle={'display':'inline-block'}
                ),

                html.Big('Descriptors'),
                dcc.RadioItems(
                    id='descr_type',
                    options=[
                        {'label':'Static','value':'static'},
                        {'label':'Dynamic','value':'dynamic'}
                    ],
                    value='static',
                    labelStyle={'display':'inline-block'}
                ),
                html.Div(
                    id='descriptors'
                )

            ]
        ),
        html.Hr(),
        html.Div([
            html.Button(id='submit_button', n_clicks=0, children='submit'),
        ]),
        html.Hr(),
        html.Div(
        id='dessin'
        ),
        html.Div(id='hidden'),
        html.Div(id='new_class'),
        html.Div(id='new_descr_values'),#, style={‘display’:‘none’})
        html.Div(id='hidden_type'),
        html.Hr(),
        html.Div(id='visualisation')
    ]
)

# Input main_sound
@app.callback(
    Output('input1', 'children'),
    Output('sound','children'),
    Input('main_sound', 'filename'))
def set_name_input1(filename):
    if isinstance(filename, str) and len(filename)>4:
        return html.I(filename), html.Audio(src='assets/temp.wav', controls=True)
    else:
        return None, None


# Input separated_tracks
@app.callback(
    Output('input2', 'children'),
    Input('separated_tracks', 'filename'))
def set_name_input2(filename):
    if filename==None: return ''
    return [html.Div(html.I(name)) for name in filename]

# Input onsets
@app.callback(
    Output('input3', 'children'),
    Input('onsets', 'filename'))
def set_name_input3(filename):
    return html.I(filename)


# Descriptors inputs
@app.callback(
    Output('vis_descr_type', 'children'),
    Input('compute_type', 'value'))
def set_type_descr(value):
    if value=='all_descriptors':
        return html.Div(
            id='vis_descr_type_bis',
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
            ]
        ),

# Descriptors type
@app.callback(
    Output('hidden_type', 'children'),
    Input('vis_descr_type_radio', 'value'))
def set_type_descr_hidden(value):
    return value

# Descriptors inputs
@app.callback(
    Output('vis_descr', 'children'),
    Input('compute_type', 'value'))
    # Input('vis_descr_type_radio', 'value'))
def set_type_descr_bis(type):
    if type!='all_descriptors':
        return html.Div(
            children=[
                dcc.Dropdown(
                    id='selected_descr1',
                    options=[{'label': descrNames[descr], 'value': descr} for descr in descrList[type]],
                    placeholder="Select a {} descriptor for axe x".format(type),
                    style={'width':'50%'}
                ),
                html.Br(),
                dcc.Dropdown(
                    id={'type':'selected_descr', 'index':3},
                    options=[{'label': descrNames[descr], 'value': descr} for descr in descrList[type]],
                    placeholder="Select a {} descriptor for axe y".format(type),
                    style={'width':'50%'}
                )
            ]
        )





# Test
# @app.callback(
#     Output('dessin', 'children'),
#     Input('submit_button', 'n_clicks'),
#     State('vis_type', 'value'),
#     State('descr_type', 'value'))
# def set_draw(n_clicks,selected_vis_type, selected_descr_type):
#     if n_clicks>0:
#         return 'Draw a {} figure of {} descriptors, press {}'.format(selected_vis_type, selected_descr_type, n_clicks)




# Instanciation de classe
@app.callback(
    Output('new_class', 'children'),
    Input('submit_button', 'n_clicks'),
    State('main_sound','contents'),
    State('separated_tracks','contents'),
    State('onsets','filename'))
def set_class_instance(n_clicks, input1, input2, input3):
    global mem_input1,mem_input2, mem_input3
    if n_clicks==0:
        mem_input1, mem_input2 = None, None
    else:
        if mem_input1==input1 and mem_input2==input2 :
            rep = 'No change in class instance'
        else:
            rep = 'New class instanced'
            global duration
            # Load main sound
            content_type1, content_string1 = input1.split(",")
            decoded1 = base64.b64decode(content_string1)
            wav_file = open("assets/temp.wav", "wb")
            wav_file.write(decoded1)
            y, sr = librosa.load('assets/temp.wav', duration = duration, sr=None)

            # Load separated_tracks
            if input2 is not None:
                i=0 #Number of separated tracks
                l=[] #List of separated tracks
                for content in input2:
                    i+=1
                    content_type2, content_string2 = content.split(",")
                    decoded2 = base64.b64decode(content_string2)
                    wav_file = open("assets/temp{}.wav".format(i), "wb")
                    wav_file.write(decoded2)
                    y_temp, sr = librosa.load('assets/temp{}.wav'.format(i), duration = duration, sr=None)
                    l.append(y_temp)

            # Load onsets
            # content_type3, content_string3 = input3.split(",")
            # decoded3 = base64.b64decode(content_string3)
            # txt_file = open("assets/tempOnsets.txt", "wb")
            # txt_file.write(decoded3)



            # Instance class
            global S
            if input2 is None:
                S = SignalSepare(y, sr, [], 'assets/'+input3, Notemin, Notemax)
            else:
                S = SignalSepare(y, sr, l, 'assets/'+input3, Notemin, Notemax)
            # Analyse spectrale et segmentation
            S.GlobalSpectralAnalyis()
            S.TracksSpectralAnalyis()
            print('Class created')



        mem_input1, mem_input2 = input1, input2
        return rep

# Descriptors computation
@app.callback(
    Output('new_descr_values', 'children'),
    Input('submit_button', 'n_clicks'),
    Input('new_class', 'children'),
    State({'type': 'selected_descr', 'index': ALL}, 'value'),
    State('vis_type','value'))
def compute_descriptors(n_clicks, newClass, values, selected_vis_type):
    global mem_space
    if n_clicks == 0:
        mem_space = None
    else:
        # List of descriptors to compute
        global space
        space=[]
        if  selected_vis_type == 'temporal':
            for descr in values[0]:
                space.append(descr)
        else:
            space.append(values[0])
            space.append(values[1])



        # Do the computation whenever space is changed or new class is instanciated
        if space != mem_space or newClass == 'New class instanced':
            global S, simpl
            if simpl: S.SimplifySpectrum()
            S.ComputeDescripteurs(space = space)
            rep='New descriptors values'
        else: rep='No change in descriptors values'
        mem_space = space
        return rep


# Visualisation
# @app.callback(
#     Output('visualisation','children'),
#     Input('submit_button', 'n_clicks'),
#     Input('new_descr_values', 'children'),
#     Input('vis_trajectories_display', 'children'),
#     State('vis_type','value'),
#     State('trajectories','on'))
# def set_visualisation(n_clicks, newValuesDescr, fantome, vis_type, traj):
#     global mem_vis_type, mem_traj
#     if n_clicks==0:
#         mem_vis_type, mem_traj = vis_type, traj
#         return None
#     else:
#         global S,space
#         if vis_type!=mem_vis_type or traj!=mem_traj or newValuesDescr=='New descriptors values':
#             print('bientôt nouvelle figure')
#             S.Affichage(space = space, end = duration, vis_type=vis_type, traj=traj)
#             mem_vis_type, mem_traj = vis_type, traj
#             return html.Img(id='image',src=app.get_asset_url("temp_figure.png"), style={'width':'50%'})
#         return html.Img(id='image',src=app.get_asset_url("temp_figure.png"), style={'width':'50%'})






if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=False)
