def prepocess_train_array(data):
    # a function that reshapes and renormalises the training numpy array
    im_arr = np.load(data).reshape(23194, 150, 150,1).astype('float32')
    im_arr = im_arr[:,11:139, 11:139,:]
    im_arr = (im_arr - 127.5) / 127.5 # Normalize the images to [-1, 1]
    return im_arr

