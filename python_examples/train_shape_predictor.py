#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to use dlib's implementation of the paper:
#   One Millisecond Face Alignment with an Ensemble of Regression Trees by
#   Vahid Kazemi and Josephine Sullivan, CVPR 2014
#
#   In particular, we will train a face landmarking model based on a small
#   dataset and then evaluate it.  If you want to visualize the output of the
#   trained model on some images then you can run the
#   face_landmark_detection.py example program with predictor.dat as the input
#   model.
#
#   It should also be noted that this kind of model, while often used for face
#   landmarking, is quite general and can be used for a variety of shape
#   prediction tasks.  But here we demonstrate it only on a simple face
#   landmarking task.
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

import os
import sys
import glob

import dlib
from skimage import io

import argparse


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-p", "--path", required=True,
                    help="path to dataset")
    ap.add_argument("-v", "--verbose", default=True,
                    help="be_verbose flag")
    ap.add_argument("-cd", "--cascade_depth", type=int, default=10,
                    help="cascade_depth value")
    ap.add_argument("-td", "--tree_depth", type=int,
                    default=4, help="tree_depth value")
    ap.add_argument("-ntpcl", "--num_tress_per_cascade_level", type=int,
                    default=500, help="num_tress_per_cascade_level value")
    ap.add_argument("-nu", "--nu", default=0.1, type=float, help="nu value")
    ap.add_argument("-oa", "--oversampling_amount", type=int, default=20,
                    help="oversampling_amount value")
    ap.add_argument("-fps", "--feature_pool_size", type=int, default=400,
                    help="feature_pool_size value")
    ap.add_argument("-l", "--lambda_param", default=0.1,
                    type=float, help="lambda_param value")
    ap.add_argument("-nts", "--num_test_splits", type=int, default=20,
                    help="num_test_splits value")
    ap.add_argument("-fprp", "--feature_pool_region_padding",
                    default=0, type=int, help="feature_pool_region_padding value")
    ap.add_argument("-rs", "--random_seed", default="",
                    help="random_seed value")
    ap.add_argument("-nt", "--num_threads", default=0,
                    type=int, help="num_threads value")
    ap.add_argument("-m", "--model_name", required=True, help="model_name")
    args = ap.parse_args()

    # In this example we are going to train a face detector based on the small
    # faces dataset in the examples/faces directory.  This means you need to supply
    # the path to this faces folder as a command line argument so we will know
    # where it is.
    faces_folder = args.path

    options = dlib.shape_predictor_training_options()
    # Now make the object responsible for training the model.
    # This algorithm has a bunch of parameters you can mess with.  The
    # documentation for the shape_predictor_trainer explains all of them.
    # You should also read Kazemi's paper which explains all the parameters
    # in great detail.  However, here I'm just setting three of them
    # differently than their default values.  I'm doing this because we
    # have a very small dataset.  In particular, setting the oversampling
    # to a high amount (300) effectively boosts the training set size, so
    # that helps this example.
    # options.oversampling_amount = args.oversampling_amount
    # I'm also reducing the capacity of the model by explicitly increasing
    # the regularization (making nu smaller) and by using trees with
    # smaller depths.
    # options.nu = 0.05
    # options.lambda_param = 0.1
    # options.tree_depth = 5
    # options.be_verbose = True
    # options.num_threads = 0
    # model_name = "predictor_1.dat"
    options.be_verbose = args.verbose
    options.cascade_depth = args.cascade_depth
    options.tree_depth = args.tree_depth
    options.num_tress_per_cascade_level = args.num_tress_per_cascade_level
    options.nu = args.nu
    options.oversampling_amount = args.oversampling_amount
    options.feature_pool_size = args.feature_pool_size
    options.lambda_param = args.lambda_param
    options.num_test_splits = args.num_test_splits
    options.feature_pool_region_padding = args.feature_pool_region_padding
    options.random_seed = args.random_seed
    options.num_threads = args.num_threads
    model_name = args.model_name

    # dlib.train_shape_predictor() does the actual training.  It will save the
    # final predictor to predictor.dat.  The input is an XML file that lists the
    # images in the training dataset and also contains the positions of the face
    # parts.
    training_xml_path = os.path.join(
        faces_folder, "labels_ibug_300W_train.xml")
    dlib.train_shape_predictor(training_xml_path, model_name, options)

    # Now that we have a model we can test it.  dlib.test_shape_predictor()
    # measures the average distance between a face landmark output by the
    # shape_predictor and where it should be according to the truth data.
    print("\nTraining accuracy: {}".format(
        dlib.test_shape_predictor(training_xml_path, model_name)))
    # The real test is to see how well it does on data it wasn't trained on.  We
    # trained it on a very small dataset so the accuracy is not extremely high, but
    # it's still doing quite good.  Moreover, if you train it on one of the large
    # face landmarking datasets you will obtain state-of-the-art results, as shown
    # in the Kazemi paper.
    testing_xml_path = os.path.join(faces_folder, "labels_ibug_300W.xml")
    print("Testing accuracy: {}".format(
        dlib.test_shape_predictor(testing_xml_path, model_name)))

    # Now let's use it as you would in a normal application.  First we will load it
    # from disk. We also need to load a face detector to provide the initial
    # estimate of the facial location.
    predictor = dlib.shape_predictor(model_name)
    detector = dlib.get_frontal_face_detector()

    # Now let's run the detector and shape_predictor over the images in the faces
    # folder and display the results.
    print("Showing detections and predictions on the images in the faces folder...")
    win = dlib.image_window()
    for f in glob.glob(os.path.join(faces_folder, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)

        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))
            # Draw the face landmarks on the screen.
            win.add_overlay(shape)

        win.add_overlay(dets)
        dlib.hit_enter_to_continue()


if __name__ == "__main__":
    main()
