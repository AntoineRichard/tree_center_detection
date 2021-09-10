import tensorflow.compat.v1 as tf1
from subprocess import Popen, PIPE
from skimage import exposure
import tensorflow as tf
import skimage as ski
import numpy as np
import scipy
import cv2

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

def normalizeTF(img):
    """! Normalizes an image between 0 and 1

    @type img: tf.tensor
    @param img: an ND tensor [?, .., ?]

    @rtype: tf.tensor
    @return: an ND tensor [?, .., ?]
    """
    img = img - tf.reduce_min(img)
    img = img / tf.reduce_max(img)
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

def remove_image_block(x,y):
    """! Randomly removes the lower part of an image or a sequence of images

    @type x: tf.tensor
    @param x: a tensor of shape [Height, Width, Sequence_size]
    @type y: tf.tensor
    @param y: a tensor of shape [Sequence_size, 2]

    @rtype: tuple
    @return: a tuple of tensors of shape([Height, Width, Sequence_size], [Sequence_size, 2])
    """
    # 0 no mask, 1 mask from lower
    mask_height = int(np.random.rand()*2)
    #mask_width = int(np.random.rand()*3)
    min_pixels = 10
    max_pixels = 30
    x = np.array(x)
    if mask_height == 1:
        delta = int(min_pixels + np.random.rand()*(max_pixels - min_pixels))
        x[:,:int((y[0,0] + 0.5)*256 - delta)] = -0.5
    #elif mask_height == 2:
    #    delta = int(min_pixels + np.random.rand()*(max_pixels - min_pixels))
    #    x[:,int((y[0,0] + 0.5)*256 + delta):] = -0.5
    #if mask_width == 1:
    #    delta = int(min_pixels + np.random.rand()*(max_pixels - min_pixels))
    #    x[:int((y[0,1] + 0.5)*256 - delta),:] = -0.5
    #elif mask_width == 2:
    #    delta = int(min_pixels + np.random.rand()*(max_pixels - min_pixels))
    #    x[int((y[0,1] + 0.5)*256 + delta):,:] = -0.5
    return x, y

def rotate(x, y):
    """! Rotates a whole sequence by a random amount

    @type x: tf.tensor
    @param x: a tensor of shape [Height, Width, Sequence_size]
    @type y: tf.tensor
    @param y: a tensor of shape [Sequence_size, 2]
    
    @rtype: tuple
    @return: a tuple of tensors of shape([Height, Width, Sequence_size], [Sequence_size, 2])
    """
    y = np.array(y)
    theta = np.random.rand()*2*np.pi
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    y = np.matmul(R,y.T).T
    x = scipy.ndimage.rotate(x, -180*theta/np.pi, reshape=False, cval=-0.5)
    return (x,y)

def shift(x, y, max_shift=125):
    """! Shifts a whole sequence by a random amount

    @type x: tf.tensor
    @param x: a tensor of shape [Height, Width, Sequence_size]
    @type y: tf.tensor
    @param y: a tensor of shape [Sequence_size, 2]
    
    @rtype: tuple
    @return: a tuple of tensors of shape([Height, Width, Sequence_size], [Sequence_size, 2])
    """
    y = np.array(y)
    shifts = (np.random.rand(2) - 0.5)*max_shift/256
    x = scipy.ndimage.shift(x, [int(shifts[0]*256), int(shifts[1]*256), 0],cval=-0.5)
    y[:,0] = y[:,0] + shifts[1]
    y[:,1] = y[:,1] + shifts[0]
    return x, y

def augmentSequence(x,y):
    """
    Rotates a whole sequence by a random amount. To be used on a tf.dataset using map.

    INPUTS
    x: a tensor of shape [Sequence_size, Height, Width, 1]
    y: a tensor of shape [Sequence_size, 2]
    OUTPUTS
    x: a tensor of shape [Sequence_size, Height, Width, 1]
    y: a tensor of shape [Sequence_size, 2]
    """
    x = tf.squeeze(x) # [S,H,W,1] -> [S,H,W]
    x = tf.transpose(x,perm=[1,2,0]) # [S,H,W] -> [H,W,S]
    x,y = tf.py_function(rotate, [x,y], (tf.float32, tf.float32))
    x,y = tf.py_function(shift, [x,y], (tf.float32, tf.float32))
    x = tf.transpose(x,perm=[2,0,1])
    x = tf.expand_dims(x,-1)
    return x,y

def augmentImage(x,y):
    """
    Rotates a whole sequence by a random amount. To be used on a tf.dataset using map.

    INPUTS
    x: a tensor of shape [Height, Width, 1]
    y: a tensor of shape [2]
    OUTPUTS
    x: a tensor of shape [Height, Width, 1]
    y: a tensor of shape [2]
    """
    y = tf.expand_dims(y,0)
    x,y = tf.py_function(remove_image_block, [x,y], (tf.float32, tf.float32))
    x,y = tf.py_function(rotate, [x,y], (tf.float32, tf.float32))
    x,y = tf.py_function(remove_image_block, [x,y], (tf.float32, tf.float32))
    x,y = tf.py_function(rotate, [x,y], (tf.float32, tf.float32))
    x,y = tf.py_function(shift, [x,y], (tf.float32, tf.float32)) # replace by tf.roll ? More like a smart concat
    x = randomBrightnessContrast(x)
    x = gaussianNoise(x)
    x = normalizeTF(x)
    x = tf.py_function(remove_image_block, [x], (tf.float32, tf.float32))
    #x = tf.expand_dims(x,-1)
    y = tf.squeeze(y)
    return x,y

def applyCorrection(x):
    """
    Applies adaptative histogram normalization on a whole sequence. To be used on a tf.dataset using map.

    INPUT
    x: a tensor of shape [Sequence_size, Height, Width, 1]
    OUTPUT
    x: a tensor of shape [Sequence_size, Height, Width, 1]
    """
    s = x.shape
    x = tf.reshape(x, [s[0]*s[1],s[2]])
    x = tf.py_function(equalize, [x], tf.float32)
    x = tf.reshape(x, [s[0],s[1],s[2],1])
    return x

def applyCorrectionImage(x):
    """
    Applies adaptative histogram normalization on an image sequence. To be used on a tf.dataset using map.

    INPUT
    x: a tensor of shape [Sequence_size, Height, Width, 1]
    OUTPUT
    x: a tensor of shape [Sequence_size, Height, Width, 1]
    """
    x = tf.squeeze(x)
    x = tf.py_function(equalize, [x], tf.float32)
    x = tf.expand_dims(x,-1)
    return x

def randomBrightnessContrast(x, max_brightness_delta=0.2, max_contrast_factor=0.3):
    """! Randomly changes images brightness and contrast.
    @type x: tf.tensor
    @param x: a tensor of shape [Height, Width, Sequence_size]
    
    @rtype: tf.tensor
    @return: a tensor of shape [Height, Width, Sequence_size]
    """
    x = tf.image.random_brightness(x, max_brightness_delta)
    x = tf.image.random_contrast(x, 1 - max_contrast_factor, 1 + max_contrast_factor)
    return x

def gaussianNoise(x, mean=0.0, max_sigma=0.005**0.5):
    """! Adds a random amount of gausian noise to the images.
    @type x: tf.tensor
    @param x: a tensor of shape [Height, Width, Sequence_size]
    
    @rtype: tf.tensor
    @return: a tensor of shape [Height, Width, Sequence_size]
    """
    sigma = tf.random.uniform([1])*max_sigma
    gauss = tf.random.normal((256,256,1), mean, sigma)
    x = x + gauss
    return x

def MRINoise(x):
    """! Creates a noise map that looks like the one generated by
     MRI machines and applies it to images.

    @type x: tf.tensor
    @param x: a tensor of shape [Height, Width, Sequence_size]
    
    @rtype: tf.tensor
    @return: a tensor of shape [Height, Width, Sequence_size]
    """
    LR_noise = np.random.normal(256,10)
    LR_noise = cv2.resize(LR_noise,(256,256))
    MR_noise = np.random.normal(256,64)
    MR_noise = cv2.resize(MR_noise,(256,256))
    HR_noise = np.random.normal(256,256)
    full_noise = LR_noise + MR_noise + HR_noise
    polar_noise = cv2.warpPolar(full_noise, (256,256), (128,128), 128, cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
    normed_polar_noise = normalize(polar_noise)
    noise_mean = np.mean(normed_polar_noise)
    mask = np.zeros_like(x)
    mask = cv2.circle(mask, (128,128), 122, 1, 5)
    x = x + normed_polar_noise * np.random.rand()*0.5
    x[mask] = noise_mean
    x[x<=noise_mean] = noise_mean
    return x

def randomKill(x, max_kills=4, kill_ratio=0.2):
    """! Selects one or more image of the sequence and randomly sets it to 0.
    @type x: tf.tensor
    @param x: a tensor of shape [Height, Width, Sequence_size]

    @rtype: tf.tensor
    @return: a tensor of shape [Height, Width, Sequence_size]
    """
    return x

def randomBlock(x, max_block_size=32, max_boxes=3):
    """! Creates blocks of random positions, values, and sizes in images.
    @type x: tf.tensor
    @param x: a tensor of shape [Height, Width, Sequence_size]
    @type max_block_size: int
    @param max_block_size: The maximum size of the blocks
    @type max_boxes: int
    @param max_boxes: The maximum number of boxes that can be added
    
    @rtype: tf.tensor
    @return: a tensor of shape [Height, Width, Sequence_size]
    """
    h, w, s = x.shape
    for i in range(s):
        boxes = int(np.random.rand()*max_boxes)
        for _ in range(boxes):
            xyhw = np.random.rand(4)
            xy = int(xyhw[0:2]*256)
            hw = int(xyhw[2:]*max_block_size)
            x[xy[0]+hw[0]:xy[0]+hw[0],xy[1]+hw[1]:xy[1]+hw[1],i] = np.random.rand() - 0.5
    return x

    

def encodeGif(frames, fps):
    """
    Encodes a GIF from a numpy array

    INPUTS
    frames: a numpy array of shape [Sequence_size, Height, Batch_size*Width, Channel]
    fps: an int, the number of frames per second to be used when encoding the GIF 
    OUTPUT
    out: the encoded GIF
    """
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join(['ffmpeg -y -f rawvideo -vcodec rawvideo','-r '+str("%.02f"%fps)+' -s '+str(w)+'x'+str(h)+' -pix_fmt '+pxfmt+' -i - -filter_complex','[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse','-r '+str("%.02f"%fps)+' -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out

def videoSummary(name, video, step=None, fps=10):
    """
    Allows to embed a video inside tensorboard

    INPUTS
    name: a string, the name of the video summary in tensorboard.
    video: a tensor of shape [Batch_size, Sequence_size, Height, Width, Channels]
    step: an int, the step at which for which this summary will be recorded
    fps: an int, the frame rate at which the GIF will be encoded
    """
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encodeGif(frames, fps)#.numpy()
        summary.value.add(tag=name + '/gif', image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + '/grid', frames.astype(np.uint8), step)

def imageSummaryCross(name, img, coords, step=None):
    """
    Allows to embed a video inside tensorboard with a red cross at the coordinates given in 'coords'.

    INPUTS
    name: a string, the name of the video summary in tensorboard.
    video: a tensor of shape [Batch_size, Sequence_size, Height, Width, Channels]
    coord: a tensor of shape [Batch_size, Sequence_size, 2]
    step: an int, the step at which for which this summary will be recorded
    fps: an int, the frame rate at which the GIF will be encoded
    name = name if isinstance(name, str) else name.decode('utf-8')
    """
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(255 * img, 0, 255).astype(np.uint8)
    img2 = np.zeros((img.shape[0],img.shape[1],img.shape[2],3),dtype=np.uint8)
    coords = np.array(coords).astype(np.int32)
    for i in range(img.shape[0]):
        img2[i] = cv2.drawMarker(np.repeat(img[i],3,-1), (coords[i,0],coords[i,1]),markerType=cv2.MARKER_CROSS,color=(255,0,0))
    tf.summary.image(name, img2.astype(np.uint8), step)

def videoSummaryCross(name, video, coords, step=None, fps=10):
    """
    Allows to embed a video inside tensorboard with a red cross at the coordinates given in 'coords'.
    INPUTS
    name: a string, the name of the video summary in tensorboard.
    video: a tensor of shape [Batch_size, Sequence_size, Height, Width, Channels]
    coord: a tensor of shape [Batch_size, Sequence_size, 2]
    step: an int, the step at which for which this summary will be recorded
    fps: an int, the frame rate at which the GIF will be encoded
    name = name if isinstance(name, str) else name.decode('utf-8')
    """
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    video2 = np.zeros((video.shape[0],video.shape[1],video.shape[2],video.shape[3],3),dtype=np.uint8)
    B, T, H, W, C = video2.shape
    coords = np.array(coords).astype(np.int32)
    try:
        for i in range(video.shape[0]):
              for j in range(video.shape[1]):
                  video2[i,j] = cv2.drawMarker(np.repeat(video[i,j],3,-1), (coords[i,j,0],coords[i,j,1]),markerType=cv2.MARKER_CROSS,color=(255,0,0))
        frames = video2.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encodeGif(frames, fps)#.numpy()
        summary.value.add(tag=name + '/gif', image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video2.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + '/grid', frames.astype(np.uint8), step)
