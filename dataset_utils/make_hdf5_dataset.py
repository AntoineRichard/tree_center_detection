from multiprocessing import Pool
from datetime import datetime
from collections import deque
import skimage as ski
import numpy as np
import pickle
import h5py
import cv2
import os 

from skimage import exposure
from tensorflow._api.v2 import data

def removeBackground(img, bg_clr_select=20):
    """!Removes the darker background around the Xray and fills it by the value of the noise within the xray.
    @type img: np.array
    @param img: an xray image
    @type bg_clr_select: int
    @param bg_clr_select: the amount of pixels to select the background from

    @rtype: np.array
    @return: the image without its black background
    """
    background_color = np.mean(img[0:bg_clr_select,0:bg_clr_select])
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (256,256), 245, 1, 10)
    xray_noise_value = np.mean(img[mask==1])
    img[img<=background_color] = xray_noise_value
    return img

def histogramCut(img, shift = 0., cap = 1.):
    """! Restricts the histogram to a range between 'shift' and 'cap'.

    @type img: np.array
    @param img: an ND array [?, .., ?]
    @type shift: float
    @param shift: a float between 0 and 1. With shift < cap.
    @type cap: float
    @param cap: a float between 0 and 1. With cap > shift.
    
    @rtype: np.array
    @return: an ND array [?, .., ?]
    """
    img = img
    img[img>cap] = cap

    img -= shift
    img[img<0] = 0.

    img /= (cap-shift)

    return img

def normalize(img):
    """! Normalizes an image

    @type img: np.array
    @param img: an ND array [?, .., ?]

    @rtype: np.array
    @return: an ND array [?, .., ?]
    """
    img = img - np.amin(img)
    img = img / np.amax(img)
    return img

def equalize(img):
    """! Peforms adaptative histogram normalization on a flatten sequence of images

    @type img: np.array
    @param img: a 2D array [?, ?]
    
    @type ada: np.array
    @param ada: a 2D array [?, ?]
    """
    ada = normalize(img)
    ada = ski.exposure.equalize_adapthist(ada, clip_limit=0.02)
    ada = normalize(ada)
    ada = ski.exposure.adjust_log(ada, gain=1, inv= False)
    ada = histogramCut(ada)
    ada = normalize(ada) - 0.5
    return ada

class H5Converter:
    def __init__(self, data_path, trees_to_load, output_path):
        """!
        @type data_path: str
        @param data_path: the path to the Xrays
        @type trees_to_load: str
        @param trees_to_load: the path to the txt file containing the trees' id
        @type output_path: str
        @param output_path: the path to the hdf5 file that will be created (must include the name and extension)
        """
        self.data_path = data_path
        self.output_path = output_path
        self.load_list(trees_to_load)
        self.max_threads = 20

    def load_list(self, tree_path):
        """! Reads which trees should be processed

        @type tree_path: str
        @param tree_path: the path to the file
        """
        with open(tree_path,'r') as file:
            f = file.readlines()
        f = [line.strip().split('/') for line in f]
        self.tree_dict = {}
        for i in f:
            if i[0] not in self.tree_dict.keys():
                self.tree_dict[i[0]] = []
            self.tree_dict[i[0]].append(i[1])

    def build(self):
        """! Processes the whole of the INRAE datas in a threaded fashion
        """
        h5 = h5py.File(self.output_path,'w')
        paths = []
        keys = []
        for tree_key in self.tree_dict.keys():
            for id_key in self.tree_dict[tree_key]:
                paths.append(os.path.join(self.data_path,tree_key,id_key))
                keys.append(id_key)
        pool = Pool(self.max_threads)
        key_list = [keys[i:i+self.max_threads] for i in range(0,len(keys), self.max_threads)]
        path_list = [paths[i:i+self.max_threads] for i in range(0,len(paths), self.max_threads)]
        for keys, paths in zip(key_list,path_list):
            processed = pool.map(self.build_hdf5_from_folder, paths)
            for res, key_name in zip(processed, keys):
                grp=h5.create_group(key_name)
                for i, image in enumerate(res):
                    grp.create_dataset(str(i),data=image)
        h5.close()

    def build_hdf5_from_folder(self, folder):
        """! Takes a folder, loads the images, equalizes the images and compresses them in memory. This is a slow process, threading is recommended. We use 20 threads on a Ryzen 9 3900X

        @type folder: str
        @param folder: the path to the folder

        @rtype: list
        @return: an ordered list of processed images
        """
        start_stamp = datetime.now()
        images = [img for img in os.listdir(folder) if img.split('.')[-1] == 'png']
        print(folder+ " :: Starting Processing :: "+str(len(images))+" images found")
        images.sort()
        processed = []
        buff = deque()
        for image_name in images:
            img = cv2.imread(os.path.join(folder, image_name), cv2.IMREAD_UNCHANGED)
            img = removeBackground(img)
            buff.append(img)
            if len(buff) >= 30:
                img_seq = np.array(buff)
                C,H,W = img_seq.shape
                img_seq = equalize(img_seq.reshape(C*H,W))
                img_seq = img_seq.reshape(C,H,W)
                img_seq = ((img_seq + 0.5)*(0xFFFF)).astype(np.uint16)
                processed.append(img_seq[:15])
                for _ in range(15):
                    buff.popleft()
        img_seq = np.array(buff)
        C,H,W = img_seq.shape
        img_seq = equalize(img_seq.reshape(C*H,W))
        img_seq = img_seq.reshape(C,H,W)
        img_seq = ((img_seq + 0.5)*(0xFFFF)).astype(np.uint16)
        processed.append(img_seq)
        processed = np.concatenate(processed, axis = 0)
        encoded = []
        eq_stamp = datetime.now()
        print(folder+" :: Equalization Done :: "+str((eq_stamp - start_stamp).total_seconds())+"s")

        for image in processed:
            encoded.append(cv2.imencode('.png',image,[cv2.IMWRITE_PNG_COMPRESSION, 9])[1])
        enc_stamp = datetime.now()
        print(folder+" :: Encoding Done :: "+str((enc_stamp - eq_stamp).total_seconds())+"s")
        return encoded

if __name__ == "__main__":
    H5C = H5Converter('/home/gpu_user/antoine/WoodSeer/XRays','/home/gpu_user/antoine/WoodSeer/XRays/trees_to_use.txt', '/home/gpu_user/antoine/WoodSeer/XRays_v2.hdf5')
    H5C.build()