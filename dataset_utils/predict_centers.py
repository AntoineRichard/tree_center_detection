import tensorflow as tf
import numpy as np
import argparse
import pickle
import scipy
import sys
import cv2
import os 

sys.path.append("/home/gpu_user/antoine/WoodSeer/Xray_center_detection")

from utils import equalize

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xray_path',type=str)
    parser.add_argument('--save_path',type=str)
    return parser.parse_args()

#BATCH_SIZE = 128
ROTATIONS  = [2*np.pi*(i/32) for i in range(32)]

encoder = tf.keras.models.load_model('/home/gpu_user/antoine/WoodSeer/Xray_center_detection/models/20210802-080512/encoder_25')
decoder = tf.keras.models.load_model('/home/gpu_user/antoine/WoodSeer/Xray_center_detection/models/20210802-080512/direct_25')
decoder(encoder(np.ones((1,256,256,1))))

encoder.summary()
decoder.summary()


def process_tree(tree_path, save_image=False, save_root='.', tree='.'):
    center_dict = {}
    center_dict['mean'] = []
    center_dict['std'] = []
    center_dict['points'] = []
    # Get images, check and sort
    images = os.listdir(tree_path)
    images = [i for i in images if i.split('.')[-1]=='png']
    images.sort()
    if save_image:
        os.makedirs(os.path.join(save_root, tree), exist_ok=True)
    # Process tree
    for i in images:
        image = cv2.imread(tree_path+'/'+i, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (256,256))/1.0
        image = equalize(image)
        image = np.expand_dims(image,-1)
        images = []
        for rot in ROTATIONS:
            images.append(scipy.ndimage.rotate(image, -180*rot/np.pi, reshape=False, cval=-0.5))
        images = np.array(images)
        pred = np.array(decoder(encoder(images, False),False))
        #out = pred 
        out = []
        for rot, y in zip(ROTATIONS, pred):
            c, s = np.cos(-rot), np.sin(-rot)
            R = np.array(((c, -s), (s, c)))
            out.append(np.matmul(R,y.T).T)
        out = (np.array(out)+0.5)*512
        center_dict['points'].append(out)
        std = np.std(out,axis=0)
        center_dict['std'].append(std)
        mean = np.mean(out,axis=0)
        center_dict['mean'].append(mean)
        if save_image:
            c_img = cv2.drawMarker(np.repeat((np.clip(image+0.5,0,1.0)*255).astype(np.uint8),3,-1), (int(mean[0]/2),int(mean[1]/2)),markerType=cv2.MARKER_CROSS,color=(255,0,0))
            cv2.imwrite(os.path.join(save_root, tree, str(i)+'.jpg'), c_img)

    return center_dict

def dump_pickle(data, path):
    with open(path+'.pkl','wb') as handle:
        pickle.dump(data, handle)

def process_specie(data_path, center_save_path):
    tree_ids = os.listdir(data_path)
    for tree_id in tree_ids:
        tree_path = os.path.join(data_path, tree_id)
        ctr_path = os.path.join(center_save_path)
        os.makedirs(ctr_path, exist_ok=True)
        centers = process_tree(tree_path, save_image=True, save_root='/home/gpu_user/antoine/WoodSeer/test',tree=tree_id)
        dump_pickle(centers, os.path.join(ctr_path,tree_id))

args = parse()
species = os.listdir(args.xray_path)
species_path = [os.path.join(args.xray_path,specie) for specie in species]
ctrs_save_path = [os.path.join(args.save_path,'nn_centers_predictions',specie) for specie in species]

for specie_path, ctr_save_path in zip(species_path, ctrs_save_path):
    process_specie(specie_path, ctr_save_path)