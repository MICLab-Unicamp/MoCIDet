#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:50:53 2020

@author: irene
"""

import numpy as np
import h5py

from tensorflow.keras.utils import to_categorical
import sklearn.feature_extraction.image as ski

def clip_slice(img,clip,pct):

    #print(img) 
    if clip == 'max':
        img = pct + img
        zzero = (np.argwhere(img==np.amin(img,axis=(0,1),keepdims=True)))
        img[tuple(zzero[0])] = -1.
        img[tuple(zzero[1])] = -1.
        img[tuple(zzero[2])] = -1.
        img -= -1
        img /= np.max(img,axis=(0,1))
        img -= 0.5
        img *= 2.
        
    elif clip == 'max2':
        zmax = (np.argwhere(img==np.amax(img,axis=(0,1),keepdims=True)))
        img[tuple(zmax[0])] = pct+1
        img[tuple(zmax[1])] = pct+1
        img[tuple(zmax[2])] = pct+1


    elif clip == 'min2':
        zzero = (np.argwhere(img==np.amin(img,axis=(0,1),keepdims=True)))
        img[tuple(zzero[0])] = -1 - pct
        img[tuple(zzero[1])] = -1 - pct
        img[tuple(zzero[2])] = -1 - pct


    elif clip == 'min':
        img = img - pct
        zmax = (np.argwhere(img==np.amax(img,axis=(0,1),keepdims=True)))
        img[tuple(zmax[0])] = 1.
        img[tuple(zmax[1])] = 1.
        img[tuple(zmax[2])] = 1.
        img -= -(1+pct)
        img /= np.max(img)
        img -= 0.5
        img *= 2.

    image  = np.clip(img,-1.,1.)


    return image

def norm_patches(x2):
    x2[:,:,:,:] -= np.min(x2[:,:,:,:],axis=(1,2),keepdims=True)
    x2[:,:,:,:] /= np.max(x2[:,:,:,:],axis=(1,2),keepdims=True)

    x2 -= .5
    x2 *= 2

    return x2

def create_patches(params,image):
    batch_img = []
    for ii in np.arange(image.shape[0]):
        batch_img += [ski.extract_patches_2d(image[ii,:,:],(params.psize[0],params.psize[1]),max_patches=1)]

    return batch_img

def generate_batches(files, idxt, params):

    nfiles = len(idxt)
    nn = len(idxt)
    counter = 0

    norm = ['min','max','min2','max2']

    bs2 = int(params.batch_size/3)

    # One file for each dataset and each orthogonal view
    h5_axial = []
    h5_coronal = []
    h5_sagittal = []
    nsamples = []

    for counter in np.arange(nfiles):
        h5_axial += [h5py.File(files[counter]+'axial.hdf5', "r")]
        h5_coronal += [ h5py.File(files[counter]+'coronal.hdf5', "r")]
        h5_sagittal += [ h5py.File(files[counter]+'sagittal.hdf5', "r")]
        nsamples += [idxt[counter].shape[0]]

    
    if not params.class_weight_working:

        for ii in np.arange(1,4,2):
            h5_axial += [h5_axial[ii]]
            h5_coronal += [h5_coronal[ii]]
            h5_sagittal += [h5_sagittal[ii]]
            nsamples += [idxt[ii].shape[0]]   
            idxt[nfiles] = idx_axial[ii]
            nfiles += 1
            
    
    while True:

        for nt in np.arange(params.ntrans):
            for index in np.arange(nfiles):
                np.random.shuffle(idxt[index])           

            for cbatch in np.arange(0,nsamples[0],bs2):

                vclip = np.random.randint(0,11,size=(params.batch_size))/100
                vnorm = np.random.randint(0,4,size=(params.batch_size))
                vflip1 = np.random.randint(0,2,size=(params.batch_size))

                #loop on dataset
                for inn in np.arange(nfiles):

                    batch_img = []
                    batch_y1 = []
                    batch_y2 = []

                    if inn > 3 and inn < 8:
                        cb = cbatch + nsamples[0]*(nt)
                    elif inn == 2:
                        cb = cbatch%nsamples[inn]
                    else: 
                        cb = cbatch                    

                    iidx = idxt[inn][cb:cb+bs2]
                    iidx.sort()
                    img = h5_axial[inn]["img"][iidx,:,:,:]                     
                    batch_y2 += [ np.full(bs2,0) ]
                    batch_y1 += [ (h5_axial[inn]["label"][iidx]) ]                      
                    batch_img += create_patches(params,img)

                    img = h5_coronal[inn]["img"][iidx,:,:,:]
                    batch_y2 += [ np.full(bs2,1) ]
                    batch_y1 += [ (h5_coronal[inn]["label"][iidx]) ]                        
                    batch_img += create_patches(params,img[:2*img.shape[0]//3,:,:])

                    img = h5_sagittal[inn]["img"][iidx,:,:,:]
                    batch_y2 += [ np.full(bs2,2) ]
                    batch_y1 += [ (h5_sagittal[inn]["label"][iidx]) ]                        
                    batch_img +=  create_patches(params,img[:2*img.shape[0]//3,:,:])
                   
                     
                x11 = norm_patches(np.vstack(batch_img))                       
                y2 = (np.hstack(batch_y2))

                if nt > 0:
                    for i1 in np.arange(params.batch_size):
                        x11[i1,:,:,:] = clip_slice(x11[i1,:,:,:],norm[vnorm[i1]],vclip[i1])
                    x11[vflip1==1,:,:,:] = x11[vflip1==1,:,::-1,:]

    
                y21 = to_categorical(np.hstack(batch_y1),num_classes=2)
                y22 = to_categorical(y2,num_classes=3)

           
                yield (x11,[y21,y22])


    for counter in np.arange(nn):
        h5_axial[counter].close()
        h5_coronal[counter].close()
        h5_sagittal[counter].close()


    return



