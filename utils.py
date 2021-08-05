import tensorflow.compat.v1 as tf1
from subprocess import Popen, PIPE
from skimage import exposure
import tensorflow as tf
import skimage as ski
import numpy as np
import scipy
import cv2

def histogramCut(img, shift = 0., cap = 1.):
    """
    Restricts the histogram to a rante between 'shift' and 'cap'.

    INPUT
    img: an ND array [?, .., ?]
    shift: a float between 0 and 1. With shift < cap.
    cap: a float between 0 and 1. With cap > shift.
    OUTPUT
    img: an ND array [?, .., ?]
    """
    img = img
    img[img>cap] = cap

    img -= shift
    img[img<0] = 0.

    img /= (cap-shift)

    return img

def normalize(img):
    """
    Normalizes an image

    INPUT
    img: an ND array [?, .., ?]
    OUTPUT
    img: an ND array [?, .., ?]
    """
    img = img - np.amin(img)
    img = img / np.amax(img)
    return img

def equalize(img):
    """
    Peforms adaptative histogram normalization on a flatten sequence of images

    INPUT
    img: a 2D array [?, ?]
    OUTPUT
    ada: a 2D array [?, ?]
    """
    ada = normalize(img)
    ada = ski.exposure.equalize_adapthist(ada, clip_limit=0.02)
    ada = normalize(ada)
    ada = ski.exposure.adjust_log(ada, gain=1, inv= False)
    ada = histogramCut(ada)
    ada = normalize(ada) - 0.5
    return ada

def remove_image_block(x,y):
    # 0 no mask, 1 mask from lower, 2 mask from higher
    mask_height = int(np.random.rand()*3)
    mask_width = int(np.random.rand()*3)
    min_pixels = 10
    max_pixels = 30
    x = np.array(x)
    if mask_height == 1:
        delta = int(min_pixels + np.random.rand()*(max_pixels - min_pixels))
        x[:,:int((y[0,0] + 0.5)*256 - delta)] = -0.5
    elif mask_height == 2:
        delta = int(min_pixels + np.random.rand()*(max_pixels - min_pixels))
        x[:,int((y[0,0] + 0.5)*256 + delta):] = -0.5
    if mask_width == 1:
        delta = int(min_pixels + np.random.rand()*(max_pixels - min_pixels))
        x[:int((y[0,1] + 0.5)*256 - delta),:] = -0.5
    elif mask_width == 2:
        delta = int(min_pixels + np.random.rand()*(max_pixels - min_pixels))
        x[int((y[0,1] + 0.5)*256 + delta):,:] = -0.5
    return x, y

def rotate(x, y):
    """
    Rotates a whole sequence by a random amount

    INPUTS
    x: a tensor of shape [Height, Width, Sequence_size]
    y: a tensor of shape [Sequence_size, 2]
    OUTPUTS:
    x: a tensor of shape [Height, Width, Sequence_size]
    y: a tensor of shape [Sequence_size, 2]
    """
    y = np.array(y)
    theta = np.random.rand()*2*np.pi
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    y = np.matmul(R,y.T).T
    # Rotate x
    x = scipy.ndimage.rotate(x, -180*theta/np.pi, reshape=False, cval=-0.5)
    return (x,y)

def shift(x, y, max_shift=125):
    """
    Shifts a whole sequence by a random amount

    INPUTS
    x: a tensor of shape [Height, Width, Sequence_size]
    y: a tensor of shape [Sequence_size, 2]
    OUTPUTS:
    x: a tensor of shape [Height, Width, Sequence_size]
    y: a tensor of shape [Sequence_size, 2]
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
    x = tf.squeeze(x)
    x = tf.transpose(x,perm=[1,2,0])
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
    x,y = tf.py_function(rotate, [x,y], (tf.float32, tf.float32))
    x,y = tf.py_function(shift, [x,y], (tf.float32, tf.float32))
    x,y = tf.py_function(remove_image_block, [x,y], (tf.float32, tf.float32))
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

def randomAtenuation(x, max_attenuation=3):
    """
    Selects one or more images of the sequence and randomly changes its brightness.
    """
    return x

def randomKill(x, max_kills=1, kill_ratio=0.5):
    """
    Selects one or more image of the sequence and randomly sets it to 0.
    """
    return x

def randomBlock(x, max_kills=1, kill_ratio=0.5):
    """
    Adds blocks of random size and values at random positions in some images of the sequence
    """
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
