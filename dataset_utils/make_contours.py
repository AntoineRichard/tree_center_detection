from multiprocessing import Pool
import numpy as np
import argparse
import pickle
import cv2
import os

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xray_path',type=str)
    parser.add_argument('--save_path',type=str)
    return parser.parse_args()

def make_save_folders(output, specie):
    os.makedirs(os.path.join(output,'imgs',specie), exist_ok=True)
    os.makedirs(os.path.join(output,'contours',specie), exist_ok=True)

def make_morphing_kernel():
    morphing_kernel = np.zeros((5,5),np.uint8)
    morphing_kernel[2,:] = 1
    morphing_kernel[:,2] = 1
    morphing_kernel[1,1:4] = 1
    morphing_kernel[3,1:4] = 1
    return morphing_kernel

def process_tree(tree_path, save_path):
    contour_dict = {}
    # Get images, check and sort
    images = os.listdir(tree_path)
    images = [i for i in images if i.split('.')[-1]=='png']
    images.sort()
    # Make nice kernel for morphological operations
    morphing_kernel = make_morphing_kernel()
    # Process tree
    for i in images:
        image = cv2.imread(tree_path+'/'+i,-1)
        # Get outside Xray background value
        void = np.mean(image[0:20,0:20])
        # Get Xray background noise value
        # Not super stable, ideally should take the mean on the whole tree
        M = np.zeros_like(image)
        M1 = cv2.circle(M, (256,256), 245, 1, 10)
        # Fill the outside of the Xray with the mean value of the Xray noise
        fill = np.mean(image[M1==1])
        image[image<=void] = fill
        # Normalize and Blur
        norm = (255*(image*1.0-np.min(image))/(np.max(image)-np.min(image))).astype(np.uint8)
        blur = cv2.GaussianBlur(norm,(5,5),0)
        blur = cv2.GaussianBlur(blur,(5,5),0)
        # Compute threshold/segmentation
        ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Clean using morphological operations
        th = cv2.dilate(cv2.erode(th,np.ones((3,3)),iterations=1),np.ones((3,3)),iterations=1)
        th = cv2.dilate(cv2.erode(th,morphing_kernel,iterations=2),morphing_kernel,iterations=2)
        # Eliminate remaining blobs by taking the largest
        numblobs, blobs = cv2.connectedComponents(th)
        max_blob = 0
        max_blob_area = 0
        for j in range(1,numblobs):
            blob_area = np.count_nonzero(blobs==j)
            if blob_area > max_blob_area:
                max_blob_area = blob_area
                max_blob =  j
        inside = ((blobs==max_blob)*255).astype(np.uint8)
        # Get contour
        contours, hierarchy = cv2.findContours(inside, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Select largest contour if there is more than one
        cmax = None
        amax = 0
        for contour in contours:
            if cv2.contourArea(contour) > amax:
                cmax = contour
                amax = cv2.contourArea(contour)
        # Transform in RGB
        img = np.expand_dims(norm,-1)
        img = np.repeat(img,3,-1)
        # Add contour for visualization
        img = cv2.drawContours(cv2.resize(img,(512,512),interpolation=cv2.INTER_NEAREST),cmax,-1, (255,0,0), 1, lineType=cv2.LINE_4)
        # Save
        cv2.imwrite(os.path.join(save_path,i), img)
        contour_dict[i] = cmax
    return contour_dict

def dump_pickle_file(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_specie(data_path, img_save_path, cnt_save_path):
    tree_ids = os.listdir(data_path)
    for tree_id in tree_ids:
        tree_path = os.path.join(data_path, tree_id)
        cnt_path = os.path.join(cnt_save_path)
        img_path = os.path.join(img_save_path,tree_id)
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(cnt_path, exist_ok=True)
        contours = process_tree(tree_path, img_path)
        dump_pickle_file(contours, os.path.join(cnt_path,tree_id+'.pkl'))

args = parse()
species = os.listdir(args.xray_path)
species_path = [os.path.join(args.xray_path,specie) for specie in species]
cnts_save_path = [os.path.join(args.save_path,'contours',specie) for specie in species]
imgs_save_path = [os.path.join(args.save_path,'contours_image',specie) for specie in species]

pool = Pool(12)
pool.starmap(process_specie, (zip(species_path, imgs_save_path, cnts_save_path)))