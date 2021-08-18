from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, argparse
os.environ['TF_KERAS'] = '1'

import numpy as np

from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,CSVLogger

from tensorflow.keras.optimizers import RMSprop, SGD, Adam

from inception_shallow import InceptionShallow

import keras.backend as K
from generator  import generate_batches

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.utils import class_weight

def createKFolds(params,f):
    # determine folds

    train_idx = []
    test_idx = []
    class_weights = []

    # private dataset:               train  val 
    #       24  motion-corrupted    D1 16    8
    #       24  motion-free         D1 16    8
    #       24  motion-corrupted    D2 16    8
    #       24  motion-free         D2 16    8
    #       24  geometric transform D1 16    8
    #       24  geometric transform D2 16    8
    
    # Simulated data is based on motion-free images. In order to restrict the test image 
    # to not be part of train data, simulated images will be the same as base motion-free images

    # The file contains motion-corrupted and motion-free acquisition list
    data = np.loadtxt(params.file,delimiter=',',dtype='U10')

    mskf = MultilabelStratifiedKFold(n_splits=params.nfolds, random_state=0)

    X = data[:,0]
    y = data[:,1:]


    for train_index, test_index in mskf.split(X, y):
        train_idx.append(train_index)
        test_idx.append(test_index)

    train_index = np.vstack(train_idx)[f]
    test_index = np.vstack(test_idx)[f]


    # 24 motion-corrupted real, 24 motion-free
    # As the label list is jointed and the acquisition files are distinct 
    # we need to adjust the indexes

    idx_train = dict()
    idx_val = dict()

    ndata_train = [0,16,32,48,72]
    ndata_val = [0,8,16,24,40]
    ndatar = [0,24,48,72,96]

    # read data 
    for ii in np.arange(len(ndatar)-1):
        idx_train[ii] = (train_index[ndata_train[ii]:ndata_train[ii+1]]-ndatar[ii])
        idx_val[ii] = (test_index[ndata_val[ii]:ndata_val[ii+1]]-ndatar[ii])

    # simulated data: using same indexes from motion-free
    idx = len(ndatar)-1
    idx_train[idx] = idx_train[1][:ndata_train[1]]
    idx_val[idx] = idx_val[1][:ndata_val[1]]

    idx_train[idx+1] = idx_train[3][:ndata_train[1]//2]
    idx_val[idx+1] = idx_val[3][:ndata_val[1]//2]

    #print(idx_train,idx_val,len(idx_train))

    # Number of acquisitions 
    ntrain = 0
    nval = 0
    for ii in np.arange(len(idx_train)):
        ntrain += len(idx_train[ii])
        nval += len(idx_val[ii])
    
    # duplicate the motion-free data to balance classes
    if not params.class_weight_working:
        ntrain += len(idx_train[1])
        nval += len(idx_val[1])    

    print(ntrain,nval)
    #print(idx_val)
    
    # create a index vector to each slice
    # real data have different number of sliced than simulated ones
    train_idx = dict()
    test_idx = dict()
    
    for ii in np.arange(idx+2):
        train_idx[ii] = duplicate_index(params,idx_train[ii])
        test_idx[ii] = duplicate_index(params,idx_val[ii])


    # If the version have class_weight for multiple outputs
    if params.class_weight_working:
        yy = []
        mfree_acq = len(idx_train[1]) + len(idx_train[3])
        mcorr_acq = len(idx_train[0]) + len(idx_train[2]) + len(idx_train[4]) + len(idx_train[5]) + len(idx_train[6]) + len(idx_train[7])

        yy.append(np.ones(mcorr_acq ,dtype=int))
        yy.append(np.zeros(mfree_acq ,dtype=int))
        yy = np.hstack(yy)
        
        class_weights = class_weight.compute_class_weight('balanced',np.unique(yy),yy)
        class_weights = dict(enumerate(class_weights))
        print('class',class_weights)

    #print(len(train_idx),test_idx)
    return train_idx, test_idx, class_weights, ntrain, nval
		


def duplicate_index(params, idx):
    # indexes for real data slices
    idx1 = idx.copy() * (params.nslices-params.nch)
    idx2 = (idx.copy() + 1) * (params.nslices-params.nch)
    iidx = []
    
    for ii in np.arange(len(idx)):
        iidx.append(np.arange(idx1[ii],idx2[ii]))
    return np.hstack(iidx)


def run_CNN(params):

    # Run the CNN training
    # There are 4 data files:
    #     motion-corrupted
    #     motion-free
    #     simulation mixlines combination
    #     simulation geometric transform

	for f in np.arange(params.nfolds):
		h5name_train = [
                     params.input + 'file_motion_corrupted_1', params.input + 'file_motion_free_1', 
                     params.input + 'file_motion_corrupted_2', params.input + 'file_motion_free_2',
		     params.input + 'file_motion_simulated_1',
                     params.input + 'file_motion_simulated_2',
                        ]

		idx_train, idx_val, class_weights, ntrain, nval = createKFolds(params,f)

		gen_train = generate_batches(files=h5name_train, idxt=idx_train,params=params)
		gen_val = generate_batches(files=h5name_train,idxt=idx_val, params=params)

        # Create model
		model,model_name,loadmdl = MyModel(params,f)

		if loadmdl == 0:
            # Transfer-learning -  training the classifier
		    print('sgd = SGD(lr=1e-2 decay=1e-6, momentum=0.9, nesterov=True)')
		    sgd = SGD(lr=5e-2, decay=1e-6, momentum=0.9, nesterov=True)

		    model = train_CNN(params,model,15,sgd,gen_train,gen_val,model_name,3,class_weights, ntrain, nval)
		    
		    
		model_name = params.models+params.architecture+'_2_'+str(f)
		
		if not os.path.isfile(model_name+'.hdf5'):
            # Fine-tunning -  training the whole architecture
		    print("[INFO] finetuning model...")

		    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)  

		    for layer in model.layers:
		        layer.trainable = True
		    

		    model = train_CNN(params,model,50,sgd,gen_train,gen_val,model_name,3,class_weights,ntrain, nval)

            # second Fine-tunning -  training the whole architecture
		    model_name = params.models+params.architecture+'_3_'+str(f)

		    print('sgd =  SGD(lr=1e-5)')
		    sgd = SGD(lr=1e-5, decay=1e-5, momentum=0.9, nesterov=True)  

		    model = train_CNN(params,model,30,sgd,gen_train,gen_val,model_name,3,class_weights,ntrain, nval)

		else:
		    print('Model exist',model_name)

	return

def MyModel(params,f):

    # create model or loading an existent one

    model_name = params.models+params.architecture+'_1_'+str(f)

    if not os.path.isfile(model_name+'.hdf5'):
        print("[INFO] Creating Model : ", model_name)

        if params.architecture == 'Inception':
            model = InceptionShallow(copy_weights=True, weights='imagenet', input_shape=(params.psize[0], params.psize[1],params.nch))
            for layer in model.layers[:-4]:
               layer.trainable = False

        loadmdl = 0
    else:
        print("Loading model: ", model_name)
        model=load_model(model_name+'.hdf5')
        for layer in model.layers:
           layer.trainable = True

        loadmdl = 1


    return model,model_name,loadmdl


def train_CNN(params,model,nepochs,sgd,gen_train,gen_val,model_name,nfactor,class_weights,ntrain, nval):

    # Train the CNN

    losses = {
            "motion": "categorical_crossentropy",
            "view": "categorical_crossentropy",
    }
    lossWeights = {"motion": 1.0,
                   "view": 0.8
                   }

    if params.class_weight_working:
        classWeight = {"motion":class_weights,
                       "view": {0:1.,1:1.,2:1.},
                       }

    print("[INFO] compiling model...",model_name)
    model.compile(optimizer=sgd, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
    print("[INFO] compiled...")
        
    if nepochs == 15:

        reduce_lr = ReduceLROnPlateau(monitor='val_motion_loss', factor=0.2,
              patience=2, min_lr=1e-8)

        earlyStopping = EarlyStopping(monitor='val_motion_loss',
                       patience=3, 
                       verbose=1, mode='auto')
    else:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
              patience=7, min_lr=1e-8)

        earlyStopping = EarlyStopping(monitor='val_loss',
                       patience=10, 
                       verbose=1, mode='auto')

    checkpoint = ModelCheckpoint(model_name+'.hdf5', monitor='val_loss',
                     verbose=0, save_best_only=True)

    csvlogger = CSVLogger(model_name+'.csv')

    nsamplesperfold = nfactor*nval*(params.nslices-params.nch)*(params.npt*params.ntrans)/(params.batch_size)
    print('nval',nval)
    print('samples',nsamplesperfold)

    if params.class_weight_working:
        model.fit(gen_train, steps_per_epoch = ((params.nfolds-1)*nsamplesperfold),
              epochs=nepochs, verbose=1,
              class_weight=class_weights, 
              callbacks=[checkpoint,earlyStopping,reduce_lr,csvlogger],
              validation_data=gen_val, validation_steps = (nsamplesperfold),
              )
    else:

       model.fit(gen_train, steps_per_epoch = ((params.nfolds-1)*nsamplesperfold),
              epochs=nepochs, verbose=1,
              callbacks=[checkpoint,earlyStopping,reduce_lr,csvlogger],
              validation_data=gen_val, validation_steps = (nsamplesperfold),
              )


    return model

sys.argv = ['foo']

parser = argparse.ArgumentParser()

parser.add_argument('-input',
                    type=str,
                    default='path\to\dataset',
                    help='txt file containing input data filenames')

parser.add_argument('-file',
                    type=str,
                    default='path\to\csv\file',
                    help='txt file containing input data filenames')

parser.add_argument('-models', type=str, 
                    default='path\to\save\model',
                    help='models prefix directory')

parser.add_argument('-architecture', type=str, default='Inception',
                    help='architecture to be fine-tuned')

parser.add_argument('-psize', type=int, default=[128,128],
                    help='patch size extracted from dataset')

parser.add_argument('-nfolds', type=int, default=3,
                    help='number of folds for training')

parser.add_argument('-nch', type=int, default=3,
                    help='number of input channels')

parser.add_argument('-ntrans', type=int, default=36,
                    help='number of transformations ')

parser.add_argument('-batch_size', type=int, default= 12,
                    help='size of batch')

parser.add_argument('-nslices', type=int, default=40,
                    help='number of slices extracted per acquisition')

parser.add_argument('-class_weight_working', type=int, default=0,
                    help='Set the class_weight usage')


params = parser.parse_args()

run_CNN(params)
