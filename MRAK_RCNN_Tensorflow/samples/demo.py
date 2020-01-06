import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from skimage import transform,data,io
import time
def detect(image_dir):

    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from MRAK_RCNN_Tensorflow.mrcnn import utils
    import MRAK_RCNN_Tensorflow.mrcnn.model as modellib
    from MRAK_RCNN_Tensorflow.mrcnn import visualize
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "MRAK_RCNN_Tensorflow/samples/coco/"))  # To find local version
    import coco

    #get_ipython().run_line_magic('matplotlib', 'inline')

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "MRAK_RCNN_Tensorflow/mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    #IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    IMAGE_DIR = image_dir
   # print("end")


    # ## Configurations
    #
    # We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
    #
    # For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

    # In[2]:


    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
   # config.display()
   # print("end")


    # ## Create Model and Load Trained Weights

    # In[3]:


    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
  #  print("end")


    # ## Class Names
    #
    # The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
    #
    # To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
    #
    # To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
    # ```
    # # Load COCO dataset
    # dataset = coco.CocoDataset()
    # dataset.load_coco(COCO_DIR, "train")
    # dataset.prepare()
    #
    # # Print class names
    # print(dataset.class_names)
    # ```
    #
    # We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

    # In[4]:


    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['bird']
 #   print("end")


    # ## Run Object Detection

    # In[5]:


    # Load a random image from the images folder
#    file_names = next(os.walk(IMAGE_DIR))[2]
   # file_names=image_dir

    image = skimage.io.imread(image_dir)

    start_time = time.time()

    # Run detection
    results = model.detect([image], verbose=1)

    end_time = time.time()
    detect_time = end_time - start_time

    # Visualize results
    r = results[0]

 #   print("end")

    new_img_dir = os.path.join(ROOT_DIR, r"UI\config\detectipn-1.jpg")
    #最原始的图片大小
    primitive_img = io.imread(image_dir)
    x_1=primitive_img.shape[0]
    y_1=primitive_img.shape[1]

    #存储的图片大小
    img2 = io.imread(new_img_dir)
    x_2 = img2.shape[0]
    y_2 = img2.shape[1]
    #new_img = transform.rescale(img2, [float(x_1/x_2), float(y_1/y_2)])
    new_img = transform.resize(img2, (x_1, y_1))
    #new_img = transform.rescale(img2, [0.5, 0.5])
    io.imsave(new_img_dir,new_img)


    return os.path.join(ROOT_DIR, r"UI\config\detectipn-1.jpg"),detect_time,r['scores']

    # In[ ]:




