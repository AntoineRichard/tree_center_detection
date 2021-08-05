from multiprocessing import Pool
import numpy as np
import argparse
import pickle
import cv2
import os

def saveContourImage(contour, img, path):
    img = np.expand_dims(img,-1)
    img = np.repeat(img,3,-1)
    img = cv2.drawContours(img, contour, -1, (255,0,0), 1, lineType=cv2.LINE_4)
    cv2.imwrite(path, img)

def saveContours(contours, path):
    """
    Saves a dictionnary as a pickle file in binary format

    INPUT
    dataset: a dictionnary
    path: a str, the path where to save the dictionnary
    OUTPUT
    True
    """
    with open(path, 'wb') as handle:
        pickle.dump(contours, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def removeBackground(img, bg_clr_select=20):
    """
    Removes the darker background around the Xray and fills it by the value of the noise within the xray.
    """
    background_color = np.mean(img[0:bg_clr_select,0:bg_clr_select])
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (256,256), 245, 1, 10)
    xray_noise_value = np.mean(img[mask==1])
    img[img<=background_color] = xray_noise_value
    return img

def normalize(img):
    img = (255*(img*1.0-np.min(img))/(np.max(img)-np.min(img))).astype(np.uint8)
    return img

def applyOTSU(img): 
    blur = cv2.GaussianBlur(img.copy(),(5,5),0)
    blur = cv2.GaussianBlur(blur,(5,5),0)
    _, bin = cv2.threshold(blur,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bin = cv2.dilate(cv2.erode(bin,np.ones((3,3)),iterations=1),np.ones((3,3)),iterations=1)
    bin = cv2.dilate(cv2.erode(bin,np.ones((5,5)),iterations=2),np.ones((5,5)),iterations=2)
    return bin

def getLargestBlob(img):
    numblobs, blobs = cv2.connectedComponents(img)
    max_blob = 0
    max_blob_area = 0
    for j in range(1, numblobs):
        blob_area = np.count_nonzero(blobs==j)
        if blob_area > max_blob_area:
            max_blob_area = blob_area
            max_blob =  j
    largest_blob = ((blobs==max_blob)*255).astype(np.uint8)
    return largest_blob

def getLargestContour(blob):
    contours, _ = cv2.findContours(blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cmax = None
    amax = 0
    for contour in contours:
        if cv2.contourArea(contour) > amax:
            cmax = contour
            amax = cv2.contourArea(contour)
    return cmax

def processTree(args):
    specie = args[1]
    tree_id = args[2]
    save_path = args[3]
    print(specie, tree_id)
    path = os.path.join(args[0],specie,tree_id)
    imgs_save_dir = os.path.join(save_path,'images',specie,tree_id)
    contours_save_dir = os.path.join(save_path,'contours',specie)
    os.makedirs(imgs_save_dir, exist_ok=True)
    os.makedirs(contours_save_dir, exist_ok=True)
    files = os.listdir(path)
    pngs = [i for i in files if i.split('.')[-1]=='png']
    contours = {}
    for i in pngs:
        img = cv2.imread(os.path.join(path,i),-1)
        img = removeBackground(img)
        img = normalize(img)
        bin = applyOTSU(img)
        blob = getLargestBlob(bin)
        contour = getLargestContour(blob)
        saveContourImage(contour, img, os.path.join(imgs_save_dir,i))
        contours[i] = contour
    saveContours(contours, os.path.join(contours_save_dir,tree_id+'.pkl'))

def runPreprocessor(path, save_path):
    tree_species = os.listdir(path)
    for specie in tree_species:
        tree_ids = os.listdir(os.path.join(path,specie))
        arg_lists = [[path, specie, tree_id, save_path] for tree_id in tree_ids]
        Pool(8).map(processTree,arg_lists)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=str)
    parser.add_argument('--save_path',type=str,default=".")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    runPreprocessor(args.source, args.save_path)
