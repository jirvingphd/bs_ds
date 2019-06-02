"""A collection of image processing tools"""

#Your code here preview an image

def show_random_img(image_array, n=1):
    """Display n rendomly-selected images from image_array"""
    from keras.preprocessing.image import array_to_img, img_to_array, load_img
    import numpy as np
    from IPython.display import display
    i=1
    while i <= n:
        choice = np.random.choice(range(0,len(image_array)))
        print(f'Image #:{choice}')
        display(array_to_img(image_array[choice]))
        i+=1
    return
