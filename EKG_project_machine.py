#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:09:45 2018

@author: ClayElmore, ClaireLo
"""

import os
import wfdb
import numpy as np
from pywt import wavedec as wt
from pywt import waverec as wt_inv
#plt.close('all')


# this is the structure used to hold data for the EKG signals
class data_hold():
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                               Data Structure                            #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    # holds the data
    def __init__(self):
        
        # data that will train the machine
        self.name_train = [] # name of the data
        self.data_train = [] # the raw data
        self.freq_train = [] # the frequency the data is sampled at
        self.length_train = [] # length of the data sample
        self.type_train = [] # whether it is a good or a bad heart
        self.dwt_scale = 0 # scale at which wavelet transforms will be done
        self.dwt_train = [] # wavelet transform of the data
        self.dwt_mod_train = [] # thresholded wavelet
        self.vector_mod_train = [] # data transformed by wavelet
        self.heart_rate_train = [] # calculated heart rates
        self.steady_train = [] # calculated steadiness of beat
        self.time_first_train =[] # time at which first heart beat was detected
        self.qr_ratio_train =[] # calculated ratio of p and q heights
        self.qs_time_train =[] # calculated value of QRS signal time
        self.top_or_bottom_train = [] # whether or not the max is 1 or -1
        self.traits_train = [] # these are the traits that will be fed into ML
        self.rs_ratio_train = []
        
        # data that will be predicted (same descriptions as above)
        self.name_learn = []
        self.data_learn= []
        self.freq_learn = []
        self.length_learn = []
        self.type_learn = []
        self.dwt_learn = []
        self.dwt_mod_learn = []
        self.vector_mod_learn = []
        self.heart_rate_learn = []
        self.steady_learn = []
        self.time_first_learn =[]
        self.qr_ratio_learn =[] 
        self.qs_time_learn =[]
        self.top_or_bottom_learn = []
        self.traits_learn = []
        self.rs_ratio_learn = []
        
        # set up the machine learning objects
        self.fit1 = []
        self.fit2 = []
        self.fit3 = []
        self.fit4 = []
        self.fit5 = []
        self.predict1 = []
        self.predict2 = []
        self.predict3 = []
        self.predict4 = []
        self.predict5 = []
                
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                               Data Mining                               #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    def get_data(self,samps=12):
        # firgure out all of the directories that contain EKG signals
        signal_types = []
        cwd=os.getcwd()
        allFiles=os.listdir(os.getcwd())
        for i in allFiles:
            if '.' not in i:
                signal_types.append(i)
        
        # go into each of these directories and pull out the EKG signals and Hz
        for i in signal_types:
            # this denotates the training data
            if 'Unknown' not in i:
                    cwd_new = cwd+'/'+i
                    allSignals=os.listdir(cwd_new)
                    
                    # extract signals and fields from each file in the directory 
                    for j in allSignals:
                        if '.hea' in j:
                            name = j.replace(".hea","")
                            signals,fields = wfdb.rdsamp('/'+cwd_new+'/'+name\
                                                         ,sampfrom=0, sampto=2**samps)
                            
                            # store the values from signal and field from each file
                            self.name_train.append(name)
                            self.data_train.append(np.transpose(signals)[1])
                            self.freq_train.append(fields['fs'])
                            self.length_train.append(fields['sig_len'])
                            if 'Bad' in cwd_new:
                                self.type_train.append(0)
                            else:
                                self.type_train.append(1)                                
                            
            # this denotates the data that will be predicted
            if 'Unknown' in i:
                cwd_new = cwd+'/'+i
                allSignals=os.listdir(cwd_new)
                
                # extract signals and fields from each file in the directory 
                for j in allSignals:
                    if '.hea' in j:
                        name = j.replace(".hea","")
                        signals,fields = wfdb.rdsamp('/'+cwd_new+'/'+name\
                                                     ,sampfrom=0, sampto=2**samps)
                        
                        # store the values from signal and field from each file
                        self.name_learn.append(name)
                        self.data_learn.append(np.transpose(signals)[1])
                        self.freq_learn.append(fields['fs'])
                        self.length_learn.append(fields['sig_len'])
                        if 'Bad' in cwd_new:
                            self.type_learn.append(0)
                        else:
                            self.type_learn.append(1)
                
        # normalize all of the data and store whether it is a top or bottom peak:
        for vector in range(len(self.data_train)):
            self.data_train[vector] = self.data_train[vector] - np.mean(self.data_train[vector])
            self.data_train[vector] = self.data_train[vector] / max(abs(self.data_train[vector]))
            if max(self.data_train[vector]==1):
                self.top_or_bottom_train.append(1)
            else:
                self.top_or_bottom_train.append(0)
            
        for vector in range(len(self.data_learn)):
            self.data_learn[vector] = self.data_learn[vector] - np.mean(self.data_learn[vector])
            self.data_learn[vector] = self.data_learn[vector] / max(abs(self.data_learn[vector]))
            if max(self.data_learn[vector]==1):
                self.top_or_bottom_learn.append(1)
            else:
                self.top_or_bottom_learn.append(0)
                
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                               Wavelet Transform                         #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def dwt_analysis(self, scale = 5, wavelet = 'db4'):
        # goes through each data piece of the structure and wavelet transforms
        
        # training set
        for i in range(len(self.data_train)):
            vect = self.data_train[i]
            self.dwt_train.append(wt(vect,wavelet,level = scale))
            self.dwt_mod_train.append(wt(vect,wavelet,level = scale))
            
        # learning set
        for i in range(len(self.data_learn)):
            vect = self.data_learn[i]
            self.dwt_learn.append(wt(vect,wavelet,level = scale))
            self.dwt_mod_learn.append(wt(vect,wavelet,level = scale))
            
        self.dwt_scale = scale
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                           Wavelet Reconstruction                        #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def dwt_synthesis(self, wavelet = 'db4'):
        # goes through each data piece of the structure and reconstructs
        self.data_train = []
        for i in self.dwt_mod_train:
            self.data_train.append(wt_inv(i,wavelet))
        
        self.data_learn = []
        for i in self.dwt_mod_learn:
            self.data_learn.append(wt_inv(i,wavelet))
        
        # renormalize the data
        self.top_or_bottom_train = []
        for vector in range(len(self.data_train)):
            self.data_train[vector] = self.data_train[vector] - np.mean(self.data_train[vector])
            self.data_train[vector] = self.data_train[vector] / max(abs(self.data_train[vector]))
            if max(self.data_train[vector]==1):
                self.top_or_bottom_train.append(1)
            else:
                self.top_or_bottom_train.append(0)
        
        self.top_or_bottom_learn = []
        for vector in range(len(self.data_learn)):
            self.data_learn[vector] = self.data_learn[vector] - np.mean(self.data_learn[vector])
            self.data_learn[vector] = self.data_learn[vector] / max(abs(self.data_learn[vector]))
            if max(self.data_learn[vector]==1):
                self.top_or_bottom_learn.append(1)
            else:
                self.top_or_bottom_learn.append(0)
            
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                           Wavelet Modifications                         #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    # modifies the wavelet to have only one acting piece with or without 
    # the signal included
    def dwt_mod_single(self,detail_pick = 1, edit_signal = False):
        # training set modification
        for i in self.dwt_mod_train:
            for n in range(len(i)):
                if n == 0:
                    if edit_signal:
                        for k in range(len(i[n])):
                            i[n][k] = 0           
                    else:  
                        True
                elif n != detail_pick:
                    for p in range(len(i[n])):
                        i[n][p] = 0 
        
        # learning set modification
        for i in self.dwt_mod_learn:
            for n in range(len(i)):
                if n == 0:
                    if edit_signal:
                        for k in range(len(i[n])):
                            i[n][k] = 0           
                    else:  
                        True
                elif n != detail_pick:
                    for p in range(len(i[n])):
                        i[n][p] = 0 
    
    # does a threshold deletion of noise in the decomposition
    def dwt_mod_noise(self,threshold,signal_change = False):
        # training set
        for m in self.dwt_mod_train:
            for n in range(len(m)):
                if n == 0 and signal_change:
                    average = np.mean(m[n])
                    for k in range(len(m[n])):
                        if abs(m[n][k]-average) < threshold:
                            m[n][k] = 0        
                else :
                    for k in range(len(m[n])):
                        if abs(m[n][k]) < threshold:
                            m[n][k] = 0 
                            
        # learning set 
        for m in self.dwt_mod_learn:
            for n in range(len(m)):
                if n == 0 and signal_change:
                    average = np.mean(m[n])
                    for k in range(len(m[n])):
                        if abs(m[n][k]-average) < threshold:
                            m[n][k] = 0        
                else :
                    for k in range(len(m[n])):
                        if abs(m[n][k]) < threshold:
                            m[n][k] = 0 
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                           Local Minimum Finder                          #   
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def find_local_min (self, vector, start, direction = 1, array_size = 8):
        
        # work your way down the peak until you find a local minimum
        local_min = False
        count = start-5 # start a couple points lower
        while not local_min:
            local_min = True
            # check for a local minimum occuring (similar to get heart beat)
            array_below = vector[count-array_size:count]
            array_above = vector[count+1:count+array_size]
            below_bool = vector[count]<=array_below
            above_bool = vector[count]<=array_above
            
            for i in below_bool:
                if i == False:
                    local_min = False
            for i in above_bool:
                if i == False:
                    local_min = False
            if local_min:
                # once a local minimum has been found return the point and 
                # its value in the vector
                return count, vector[count]
                
            count += direction
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                           Local Maximum Finder                          #   
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def find_local_max (self, vector, start, direction = 1, array_size = 8):
    
        # work your way down the peak until you find a local minimum
        local_max = False
        count = start-5 # start a couple points lower
        while not local_max:
            local_max = True
            # check for a local minimum occuring (similar to get heart beat)
            array_below = vector[count-array_size:count]
            array_above = vector[count+1:count+array_size]
            below_bool = vector[count]>=array_below
            above_bool = vector[count]>=array_above
            
            for i in below_bool:
                if i == False:
                    local_max = False
            for i in above_bool:
                if i == False:
                    local_max = False
            if local_max:
                # once a local minimum has been found return the point and 
                # its value in the vector
                return count, vector[count]
                
            count += direction

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                           Heart Rate General                            #   
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def get_heart_rate(self,log_two,peak,dwt = False):
        
        # ------------------- Learning Set ---------------------------------- # 
        for mod_vec in range(len(self.data_learn)):
            # first determine if it is easier to use maxs or mins
            # (maxs)
            rate = 0
            rate_places = [0]
            if max(self.data_learn[mod_vec]) == 1:
                # now find local maximums:
                for point in range(len(self.data_learn[mod_vec])):
                    # need a small range to work with
                    if (point > 10) and (point < (len(self.data_learn[mod_vec]) - 12)):
                        array_below = self.data_learn[mod_vec][point-8:point]
                        array_above = self.data_learn[mod_vec][point+1:point+8]
                        beat_yes = 1
                        below_bool =self.data_learn[mod_vec][point]>=array_below
                        above_bool =self.data_learn[mod_vec][point]>=array_above
                        
                        for i in below_bool:
                            if i == False:
                                beat_yes = 0
                            else:
                                True
                        for i in above_bool:
                            if i == False:
                                beat_yes = 0
                            else:
                                True
                        if beat_yes:
                            # check to make sure this is a real beat
                            if self.data_learn[mod_vec][point]>(peak) and point>(rate_places[-1]+50):
                                rate+=1
                                if rate_places == [0]:
                                    self.time_first_learn.append(point)
                                rate_places.append(point)
                                
            #(mins)
            if min(self.data_learn[mod_vec]) == -1:
                # now find local minimums:
                rate = 0
                for point in range(len(self.data_learn[mod_vec])):
                    # need a small range to work with
                    if (point > 10) and (point < (len(self.data_learn[mod_vec]) - 12)):
                        array_below = self.data_learn[mod_vec][point-8:point]
                        array_above = self.data_learn[mod_vec][point+1:point+8]
                        beat_yes = 1
                        below_bool =self.data_learn[mod_vec][point]<=array_below
                        above_bool =self.data_learn[mod_vec][point]<=array_above
                        
                        for i in below_bool:
                            if i == False:
                                beat_yes = 0
                            else:
                                True
                        for i in above_bool:
                            if i == False:
                                beat_yes = 0
                            else:
                                True
                        if beat_yes:
                            # check to make sure this is a real beat
                            if self.data_learn[mod_vec][point]<(-peak) and point>(rate_places[-1]+50):
                                rate+=1
                                if rate_places == [0]:
                                    self.time_first_learn.append(point)
                                rate_places.append(point)                  
            rate = rate/(2**log_two/self.freq_learn[mod_vec])*60
            self.heart_rate_learn.append(rate)
            
            
            # calculate the steadiness of the beat
            diff = []
            for i in range(len(rate_places[1:])):
                diff.append(rate_places[i+1]-rate_places[i])
                
            diff_total = sum(abs(diff - np.mean(diff))) / rate / self.freq_learn[mod_vec]
            self.steady_learn.append(diff_total)
            
        # ------------------- Training Set ---------------------------------- # 
            
        for mod_vec in range(len(self.data_train)):
            # first determine if it is easier to use maxs or mins
            # (maxs)
            rate = 0
            rate_places = [0]
            if max(self.data_train[mod_vec]) == 1:
                # now find local maximums:
                for point in range(len(self.data_train[mod_vec])):
                    # need a small range to work with
                    if (point > 10) and (point < (len(self.data_train[mod_vec]) - 12)):
                        array_below = self.data_train[mod_vec][point-8:point]
                        array_above = self.data_train[mod_vec][point+1:point+8]
                        beat_yes = 1
                        below_bool =self.data_train[mod_vec][point]>=array_below
                        above_bool =self.data_train[mod_vec][point]>=array_above
                        
                        for i in below_bool:
                            if i == False:
                                beat_yes = 0
                            else:
                                True
                        for i in above_bool:
                            if i == False:
                                beat_yes = 0
                            else:
                                True
                        if beat_yes:
                            # check to make sure this is a real beat
                            
                            if self.data_train[mod_vec][point]>(peak) and point>(rate_places[-1]+50):
                                rate+=1
                                if rate_places == [0]:
                                    self.time_first_train.append(point)
                                rate_places.append(point)
                                
            #(mins)
            if min(self.data_train[mod_vec]) == -1:
                # now find local maximums:
                for point in range(len(self.data_train[mod_vec])):
                    # need a small range to work with
                    if (point > 10) and (point < (len(self.data_train[mod_vec]) - 12)):
                        array_below = self.data_train[mod_vec][point-8:point]
                        array_above = self.data_train[mod_vec][point+1:point+8]
                        beat_yes = 1
                        below_bool =self.data_train[mod_vec][point]<=array_below
                        above_bool =self.data_train[mod_vec][point]<=array_above
                        
                        for i in below_bool:
                            if i == False:
                                beat_yes = 0
                            else:
                                True
                        for i in above_bool:
                            if i == False:
                                beat_yes = 0
                            else:
                                True
                        if beat_yes:
                            # check to make sure this is a real beat
                            if self.data_train[mod_vec][point]<(-peak) and point>(rate_places[-1]+50):
                                rate+=1
                                if rate_places == [0]:
                                    self.time_first_train.append(point)
                                rate_places.append(point)
                    
            # calculate the rate of the hearts in bpm
            rate_new = rate/(2**log_two/self.freq_train[mod_vec])*60
            self.heart_rate_train.append(rate_new)
            
            # calculate the steadiness of the beat
            diff = []
            for i in range(len(rate_places[1:])):
                diff.append(rate_places[i+1]-rate_places[i])
            diff_total = sum(abs(diff - np.mean(diff))) / rate / self.freq_train[mod_vec]
            self.steady_train.append(diff_total)


    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                         QR Height Ratio Extraction                      #   
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def get_q_r_ratio(self,array_size):
        # this function will determine the ratio between the q and p heights
        
        # ------------------- Training Set ---------------------------------- # 
        
        # start by going through every signal
        for signal in range(len(self.data_train)):
            
            # Determine if the peaks were found at the top or the bottom for rate
            if max(self.data_train[signal])==1:
                # this is the case where the peaks were found at the top
                
                # start at the point where the first beat was determined:
                point = self.time_first_train[signal]
                start = point -5
                vector = self.data_train[signal]
                
                min_point,min_val = self.find_local_min(vector,start,-1)
                
                q_height = self.data_train[signal][min_point]
                p_height = self.data_train[signal][point]

                # now that the p and q heights are determined, you have to find 
                # the ratio. Do this by finding a new relative max. This is the 
                # value that is taken to be the "zero" point
                
                # start at the point where the local min was determined:
                start = min_point - 2
                vector = self.data_train[signal]
                
                # determine the local max
                max_point,max_val = self.find_local_max(vector,start,-1)                
                zero_height = self.data_train[signal][max_point]
                
            
            # this is the case where mins are taken for the beat
            else:
                # this is the case where the peaks were found at the bottom
                
                # start at the point where the first beat was determined:
                point = self.time_first_train[signal]
                start = point + 5
                vector = self.data_train[signal]
                # work your way up the peak until you find a local max
                max_point,max_val = self.find_local_max(vector,start,1)
                
                q_height = self.data_train[signal][max_point]
                p_height = self.data_train[signal][point]

                # now that the p and q heights are determined, you have to find 
                # the ratio. Do this by finding a new relative min. This is the 
                # value that is taken to be the "zero" point
                point = max_point
                start = point + 3
                vector = self.data_train[signal]
                
                min_point,min_val = self.find_local_min(vector,start,-1)
                zero_height = self.data_train[signal][min_point]

            # Finally determine and store the ratio of the heights and make sure 
            # not divide by 0
            if (q_height == zero_height):
                pq_ratio = 100
            elif (p_height == zero_height):
                pq_ratio = 0
            else:
                pq_ratio = (p_height-zero_height)/(zero_height-q_height)/p_height
            self.qr_ratio_train.append(pq_ratio)
                                                        
        # ------------------- Learning Set ---------------------------------- # 
        
        # Do the same thing for the learning set. See above for explanation
        for signal in range(len(self.data_learn)):
            if max(self.data_learn[signal])==1:
                point = self.time_first_learn[signal]
                start = point -5
                vector = self.data_learn[signal]                
                min_point,min_val = self.find_local_min(vector,start,-1)                
                q_height = self.data_learn[signal][min_point]
                p_height = self.data_learn[signal][point]
                start = min_point - 2
                vector = self.data_learn[signal]
                max_point,max_val = self.find_local_max(vector,start,-1)                
                zero_height = self.data_learn[signal][max_point]
            else:
                point = self.time_first_learn[signal]
                start = point + 5
                vector = self.data_learn[signal]
                max_point,max_val = self.find_local_max(vector,start,1)                
                q_height = self.data_learn[signal][max_point]
                p_height = self.data_learn[signal][point]
                point = max_point
                start = point + 3
                vector = self.data_learn[signal]                
                min_point,min_val = self.find_local_min(vector,start,-1)
                zero_height = self.data_learn[signal][min_point]
            if (q_height == zero_height):
                pq_ratio = 100
            elif (p_height == zero_height):
                pq_ratio = 0
            else:
                pq_ratio = (p_height-zero_height)/(zero_height-q_height)
            self.qr_ratio_learn.append(pq_ratio)

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                         RS Ratio Extraction                             #   
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def get_r_s_ratio(self,array_size):
        # this function will determine the ratio between the q and p heights
        
        # ------------------- Training Set ---------------------------------- # 
        
        # start by going through every signal
        for signal in range(len(self.data_train)):
            
            # Determine if the peaks were found at the top or the bottom for rate
            if max(self.data_train[signal])==1:
                # this is the case where the peaks were found at the top
                
                # start at the point where the first beat was determined:
                point = self.time_first_train[signal]
                start = point + 5
                vector = self.data_train[signal]
                
                min_point,min_val = self.find_local_min(vector,start,1)
                
                s_height = self.data_train[signal][min_point]
                p_height = self.data_train[signal][point]

                # now that the p and q heights are determined, you have to find 
                # the ratio. Do this by finding a new relative max. This is the 
                # value that is taken to be the "zero" point
                
                # start at the point where the local min was determined:
                start = min_point + 2
                vector = self.data_train[signal]
                
                # determine the local max
                max_point,max_val = self.find_local_max(vector,start,1)                
                zero_height = self.data_train[signal][max_point]
                
                # Finally determine and store the ratio of the heights and make sure 
                # not divide by 0
                if (s_height == zero_height):
                    sq_ratio = 100
                elif (p_height == zero_height):
                    sq_ratio = 0
                else:
                    sq_ratio = (p_height-zero_height)/(zero_height-s_height)/p_height
                self.rs_ratio_train.append(sq_ratio)
            
            # this is the case where mins are taken for the beat
            else:
                # this case is impossible to tell where you are, so a zero is 
                # returned for th ratio
                self.rs_ratio_train.append(0)
                                                        
        # ------------------- Learning Set ---------------------------------- # 
        
        # start by going through every signal
        for signal in range(len(self.data_learn)):
            
            # Determine if the peaks were found at the top or the bottom for rate
            if max(self.data_learn[signal])==1:
                # this is the case where the peaks were found at the top
                
                # start at the point where the first beat was determined:
                point = self.time_first_learn[signal]
                start = point + 5
                vector = self.data_learn[signal]
                
                min_point,min_val = self.find_local_min(vector,start,1)
                
                s_height = self.data_learn[signal][min_point]
                p_height = self.data_learn[signal][point]

                # now that the p and q heights are determined, you have to find 
                # the ratio. Do this by finding a new relative max. This is the 
                # value that is taken to be the "zero" point
                
                # start at the point where the local min was determined:
                start = min_point + 2
                vector = self.data_learn[signal]
                
                # determine the local max
                max_point,max_val = self.find_local_max(vector,start,1)                
                zero_height = self.data_learn[signal][max_point]
                
                # Finally determine and store the ratio of the heights and make sure 
                # not divide by 0
                if (s_height == zero_height):
                    sq_ratio = 100
                elif (p_height == zero_height):
                    sq_ratio = 0
                else:
                    sq_ratio = (p_height-zero_height)/(zero_height-s_height)/p_height
                self.rs_ratio_learn.append(sq_ratio)
            
            # this is the case where mins are taken for the beat
            else:
                # this case is impossible to tell where you are, so a zero is 
                # returned for th ratio
                self.rs_ratio_learn.append(0)
            
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                         Time Ratio Extraction                           #   
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- # 
    # this function will determine the time interval of the QRS signal
    
    def get_time_interval(self, array_size):

        # ------------------- Training Set ---------------------------------- # 
        # start by going through every signal
        for signal in range(len(self.data_train)):
            
            # start at the point where the first beat was determined:
            point = self.time_first_train[signal]
            start = point + 6
            vector = self.data_train[signal]
            # work your way up the peak until you find a local max and store it
            # as the s signal time
            max_point,max_val = self.find_local_max(vector,start,1)
            s_time = max_point

            # now find the q time by working the other way
            point = self.time_first_train[signal]
            start = point - 6
            vector = self.data_train[signal]
            
            # now go the opposite way and find the q_time. To do this you
            # first need to find a local max
            max_point,max_val = self.find_local_max(vector,start,-1)
            q_time = max_point
            
            # take into account if this was a bottom or top start signal
            if max(self.data_train[signal]==1):                
                time_int = (s_time - q_time) / self.freq_train[signal]
                self.qs_time_train.append(time_int)
            else:
                # the multiplication by two is the estimation factor used
                time_int = (s_time - q_time) / self.freq_train[signal] * 2 
                self.qs_time_train.append(time_int)
        # ------------------- Learning Set ---------------------------------- # 
        # See training set for explanaiton
        for signal in range(len(self.data_learn)):
            point = self.time_first_learn[signal]
            start = point + 5
            vector = self.data_learn[signal]
            max_point,max_val = self.find_local_max(vector,start,1)
            s_time = max_point
            point = max_point
            start = point + 5
            vector = self.data_learn[signal]
            max_point,max_val = self.find_local_max(vector,start,-1)
            q_time = max_point
            if max(self.data_learn[signal]==1):                
                time_int = (s_time - q_time) / self.freq_learn[signal]
                self.qs_time_learn.append(time_int)
            else:
                time_int = (s_time - q_time) / self.freq_learn[signal] * 2 
                self.qs_time_learn.append(time_int)
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                        Machine Learning Data Set Up                     #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def set_up_machine_learn(self):
        # first, the data has to be put into a language that KNeighbors can
        # understand ie X = [[],[],...,[]] and y =[]
        self.traits_train = []
        for signal in range(len(self.data_train)):
            traits = []
            traits.append(self.heart_rate_train[signal])
            traits.append(self.steady_train[signal])
            traits.append(self.qr_ratio_train[signal])
#            traits.append(self.qs_time_train[signal])
            traits.append(self.top_or_bottom_train[signal])
#            traits.append(self.rs_ratio_train[signal])
            self.traits_train.append(traits)
        
        self.traits_learn = []
        # the data has to be reshaped for the learning set as well
        for signal in range(len(self.data_learn)):
            traits_unknown = []
            traits_unknown.append(self.heart_rate_learn[signal])
            traits_unknown.append(self.steady_learn[signal])
            traits_unknown.append(self.qr_ratio_learn[signal])
#            traits_unknown.append(self.qs_time_learn[signal])
            traits_unknown.append(self.top_or_bottom_learn[signal])
#            traits_unknown.append(self.rs_ratio_learn[signal])
            self.traits_learn.append(traits_unknown)
    
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                        Machine Learning N-neighbors                     #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def machine_learn_N_neighbors(self, neighbors = 1):
        
        # import the machine learning class
        from sklearn.neighbors import KNeighborsClassifier 
        
        # This is the model used for the training
        knn = KNeighborsClassifier(n_neighbors = neighbors)

        # this is the training set fitting
        self.fit1 = knn.fit(self.traits_train,self.type_train)

        # predict results using the model
        self.predict1 = knn.predict(self.traits_learn)
        del self.fit1
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                        Machine Learning Logistic                        #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def machine_learn_Logistic(self):
        
        # import the machine learning class
        from sklearn.linear_model import LogisticRegression
        
        # This is the model used for the training
        logistic  = LogisticRegression()

        # this is the training set fitting
        self.fit2 = logistic.fit(self.traits_train,self.type_train)

        # predict results using the model
        self.predict2 = logistic.predict(self.traits_learn)
        del self.fit2

    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                        Machine Learning Naive Bayes                     #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def machine_learn_Naive_Bayes(self):
        
        # import the machine learning class
        from sklearn.naive_bayes import GaussianNB
        
        # This is the model used for the training
        gnb_solver = GaussianNB()

        # this is the training set fitting
        self.fit3 = gnb_solver.fit(self.traits_train,self.type_train)

        # predict results using the model
        self.predict3 = gnb_solver.predict(self.traits_learn)
        
        del self.fit3
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                        Machine Learning Decision Tree                   #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def machine_learn_Decision_Tree(self):
        
        # import the machine learning class
        from sklearn import tree
        
        # This is the model used for the training
        tree_solver = tree.DecisionTreeClassifier()

        # this is the training set fitting
        self.fit4 = tree_solver.fit(self.traits_train,self.type_train)

        # predict results using the model
        self.predict4 = tree_solver.predict(self.traits_learn)
        
        del self.fit4
        
        
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                        Machine Learning Neural Network                  #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def machine_learn_Neural_Network(self):
        
        # import the machine learning class
        from sklearn.neural_network import MLPClassifier
        
        # This is the model used for the training
        newtwork_solver = MLPClassifier()

        # this is the training set fitting
        self.fit5 = newtwork_solver.fit(self.traits_train,self.type_train)

        # predict results using the model
        self.predict5 = newtwork_solver.predict(self.traits_learn)
        
        del self.fit5
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    #                        Machine Learning Testing                         #
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    def run_machine_tests(self):
        # testing for decision trees
        self.machine_learn_Decision_Tree()
        # testing for n-nearest neighbors 
        for neighbors in [1,2,3,4,5]:
            self.machine_learn_N_neighbors(neighbors=neighbors)            
        # testing for logistic regression
        self.machine_learn_Logistic()        
        # testing for logistic regression
        self.machine_learn_Naive_Bayes()
        # testing for neutral network
#        self.machine_learn_Neural_Network()

