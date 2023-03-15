from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import gc

import glob
import nibabel as nib
import pydicom

import time
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from skimage.transform import resize

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

import logging


tf.get_logger().setLevel(logging.ERROR)


def normalize_data_keras(imgslc):
    """
    Normalize the input data per channel. Input values range from [-1, +1]
    @param imgslc: input image data
    @return: normalized image data

    """
    # normalize the data
    for nc in np.arange(imgslc.shape[-1]):
        imgslc[:, :, :, nc] -= np.min(imgslc[:, :, :, nc], axis=(1, 2)).reshape(imgslc.shape[0], 1, 1)
        imgslc[:, :, :, nc] /= np.max(imgslc[:, :, :, nc], axis=(1, 2)).reshape(imgslc.shape[0], 1, 1)

    imgslc -= 0.5
    imgslc *= 2.

    return imgslc


def run_cnn(params):
    """
    Main function to predict the motion-artifact presence on T1w images.
    @param params: function parameters
    @return: motion-presence probability
    """

    nfolds = 3
    f_pred = []

    # Check if data directory is valid
    if os.path.isdir(params.data_path):
        data_dir = glob.glob(os.path.join(params.data_path, '*'))
        data_dir.sort()

        if params.data_type != 'multi-dicom' and params.data_type != 'dicom-2D':
            if os.path.isdir(data_dir[0]):
                print('Check the data path')
        elif not os.path.isdir(data_dir[0]):
            data_dir = [params.data_path]

        # predict the motion-artifact presence on data
        for nsub in data_dir:
            # Start timer
            timestart = time.time()

            if params.data_type == 'dicom-2D':

                try:
                    path = os.path.join(nsub, '*.dcm')
                    origs = glob.glob(path)
                    origs.sort()

                    cw = len(origs) // 3
                    nsl = cw - cw % 3
                    iimg = []

                    for sl in np.arange(cw, cw + nsl):
                        dcm_data = pydicom.dcmread(origs[sl]).pixel_array.astype(np.float32)
                        iimg += [np.expand_dims(dcm_data, axis=0)]

                    img = np.vstack(iimg)

                    img -= np.min(img)
                    img /= np.max(img)
                    img *= 1000.

                    h, l, w = img.shape
                    image = img.reshape(h // 3, 3, l, w).transpose(0, 2, 3, 1)

                    del (img, iimg)
                    gc.collect()

                    data = normalize_data_keras(image)

                    len_data = 1
                    del image
                    gc.collect()

                except:
                    print(' File not found: ', nsub)
                    continue

            else:

                try:
                    image = read_dataset(params, nsub)
                    len_data = 3
                except:
                    print(' File not found: ', nsub)
                    continue

            name = nsub.split('/')[-1]
            pred = []

            if params.save_slice:
                pred_min = 1.0
                pred_max = 0.0

            for f1 in np.arange(nfolds):

                try:
                    model = my_model(params, f1)

                    for nv in np.arange(len_data):

                        if len_data == 3:
                            data = slices_for_prediction(image.copy(), params, nv)

                        pred1 = model.predict(data,
                                              batch_size=10,
                                              verbose=0)

                        pred += [pred1]

                        if params.save_slice:

                            arg_min = np.argmin(pred1[:, 1])
                            arg_max = np.argmax(pred1[:, 1])

                            if pred1[arg_min, 1] < pred_min:
                                pred_min = pred1[arg_min, 1]
                                plt.imshow(data[arg_min, :, :, 1], cmap='gray')
                                plt.title(pred_min)
                                plt.savefig('min_' + name + '.png')
                            if pred1[arg_max, 1] > pred_max:
                                pred_max = pred1[arg_max, 1]
                                plt.imshow(data[arg_max, :, :, 1], cmap='gray')
                                plt.title(pred_max)
                                plt.savefig('max_' + name + '.png')

                except:
                    print('Model not found')

                del model
                gc.collect()
                K.clear_session()

            # consensus by voting
            y1 = np.argmax(np.vstack(pred), axis=1)

            bn = np.bincount(y1)
            if len(bn) == 1:
                y_pred = 0
            else:
                y_pred = (bn / len(y1))[1]

            f_pred += [[nsub, y_pred]]
            if params.display:
                print(nsub, y_pred)
                print('Time: ', time.time() - timestart)

    # save results
    if params.display:
        print(np.vstack(f_pred))
    np.savetxt(params.save_file, np.vstack(f_pred), delimiter=',', fmt='%s')

    return


def read_dataset(params, nsub):
    """
    Read T1w files. They can be dicom or nifti.
    @param params: function parameters
    @param nsub: acquisition path
    @return: acquisition slices
    """

    if params.data_type == 'multi-dicom':
        path = os.path.join(nsub, '*.dcm')
        origs = glob.glob(path)
        origs.sort()
        nsub = origs[0]

    if os.path.isfile(nsub):

        if params.data_type == 'nifti':
            # read data when nifti files
            data = nib.load(nsub)
            data_img = data.get_fdata()

            pixdim = data.header['pixdim']
            h, l, w = data_img.shape
            image = resize(data_img, (int(h * pixdim[1]), int(l * pixdim[2]), int(w * pixdim[3])))

            img3 = (np.rot90(np.rot90(image, k=2, axes=(1, 0)), k=1, axes=(0, 2))).astype('float64')

        elif params.data_type == 'multi-dicom':
            orig = pydicom.dcmread(origs[0])

            pat_position = orig[(0x0018, 0x5100)].value
            ras_position = orig[(0x0027, 0x1040)].value

            img1 = orig.pixel_array.astype(np.float32)
            img3 = np.zeros((np.array(origs).shape[0], img1.shape[0], img1.shape[1]))

            if pat_position == 'HFS':

                if ras_position == 'A':
                    img3[0, :, :] = img1

                    for j in np.arange(1, np.array(origs).shape[0]):
                        orig = pydicom.dcmread(origs[j])
                        img1 = orig.pixel_array.astype(np.float32)
                        img3[j, :, :] = img1
                    img3 = np.rot90(np.rot90(img3, k=3), k=2, axes=(1, 2)).astype('float64')

                elif ras_position == 'P':
                    img3[-1, :, :] = img1

                    for j in np.arange(1, np.array(origs).shape[0]):
                        orig = pydicom.dcmread(origs[j])
                        img1 = orig.pixel_array.astype(np.float32)
                        img3[-1 - j, :, :] = img1
                    img3 = np.rot90(np.rot90(img3, k=3), k=2, axes=(1, 2)).astype('float64')

                elif ras_position == 'L':
                    img3[0, :, :] = img1

                    for j in np.arange(1, np.array(origs).shape[0]):
                        orig = pydicom.dcmread(origs[j])
                        img1 = orig.pixel_array.astype(np.float32)
                        img3[j, :, :] = img1

                    img3 = np.rot90(np.rot90(img3, k=3), k=3, axes=(1, 2)).astype('float64')

                elif ras_position == 'R':
                    img3[-1, :, :] = img1

                    for j in np.arange(1, np.array(origs).shape[0]):
                        orig = pydicom.dcmread(origs[j])
                        img1 = orig.pixel_array.astype(np.float32)
                        img3[-1 - j, :, :] = img1

                    img3 = np.rot90(np.rot90(img3, k=3), k=3, axes=(1, 2)).astype('float64')
                del (orig, origs, img1)
                gc.collect()

        elif params.data_type == 'dicom':
            # read dcm file
            orig = pydicom.dcmread(nsub)
            img3 = orig.pixel_array.astype(np.float32)
            img3 = np.rot90(np.rot90(img3, k=3), k=3, axes=(1, 2)).astype('float64')

    return img3


def slices_for_prediction(img, params, nv):
    """
    Read each T1w volume slice to predict its motion-artifact presence
    @param img: volumetric image
    @param params: function parameters
    @param nv: image view
    @return: normalized slices from volumetric image
    """

    img -= np.min(img)
    img /= np.max(img)
    img *= 1000.

    nch = 3
    nch2 = 43

    h, l, w = img.shape
    image = []

    if nv == 0:
        cw = img.shape[nv] // 3
    else:
        cw = img.shape[nv] // 2 - 20

    for ni in np.arange(cw, cw + nch2, nch):

        if nv == 0:
            iimg3 = img[ni:(ni + nch), :, :].transpose(1, 2, 0)
        elif nv == 1:
            iimg3 = img[:, ni:(ni + nch), :].transpose(0, 2, 1)
        elif nv == 2:
            iimg3 = img[:, :, ni:(ni + nch)]

        image += [np.expand_dims(iimg3, axis=0)]

    imgslc = np.vstack(image)
    del (image, iimg3, img)
    gc.collect()

    # normalize the data

    imgslc = normalize_data_keras(imgslc)

    return imgslc


def my_model(params, f1):
    """
    Load each model per fold
    @param params: function parameters
    @param f1: fold number
    @return: cnn model
    """

    model_name = params.model_path + "MoCIDet_f" + str(f1) + ".h5"

    model = load_model(model_name)

    return model


parser = argparse.ArgumentParser()

parser.add_argument('-data_path',
                    type=str,
                    default='./test_anon/dicom/',
                    help='path to directory containing input data')

parser.add_argument('-save_file',
                    type=str,
                    default='test_dicom.txt',
                    help='file to save the predicted data')

parser.add_argument('-model_path',
                    type=str,
                    default='./models/',
                    help='path to directory containing models')

parser.add_argument('-data_type', type=str, default='dicom',
                    help='type of data: nifti, dicom, multi-dicom,dicom-2D')

parser.add_argument('-save_slice',
                    action="store_true",
                    help='flag to save minimum and maximum prediction image')

parser.add_argument('-display',
                    action="store_true",
                    help='flag to print prediction image and time spend')

params = parser.parse_args()

run_cnn(params)
