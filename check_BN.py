from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys,argparse,gc
import time
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Calculate the motion presence using the three models
# trained using private dataset and simulated data


def run_CNN(params):

    # predicts the motion presence
    # input : function parameters
    # output: motion-presence probability

    # Start timer
    nfolds = 3
    f_pred = []


    for f1 in np.arange(nfolds):

        try:                
            model = MyModel(params,f1)


        except:
            print('Model not found')


    return


def MyModel(params,f1):

    # load model per fold
    # imput params: function parameters
    #       f1: fold number
    # output: model     

    model_name = params.model_path+"MoCIDet_f"+str(f1)+".h5"

    print(model_name)
    model = load_model(model_name) 
    model.summary()
    
    return model




parser = argparse.ArgumentParser()

parser.add_argument('-data_path',
                    type=str,
                    default='test_dicom/',
                    help='path to directory containing input data')

parser.add_argument('-save_file',
                    type=str,
                    default='test_dicom.txt',
                    help='file to save the predicted data')

parser.add_argument('-model_path',
                    type=str,
                    default='/home/irene/MoCIDet/models/',
                    help='path to directory containing models')

parser.add_argument('-data_type',type=str, default = 'dicom',
                      help= 'type of data: nifti, dicom, multi-dicom,dicom-2D')

parser.add_argument('-save_slice',
                    action="store_true",
                    help='flag to save minimum and maximum prediction image')

parser.add_argument('-display', 
                    action="store_true",
                    help='flag to print prediction image and time spend')

params = parser.parse_args()

run_CNN(params)

