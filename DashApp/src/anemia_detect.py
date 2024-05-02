"""
To import this module, keep it in the same directory as the main.py file.
Then use <from anemia_detect import seg_detect>
"""



# Importing all the required libraries
import numpy as np
import pandas as pd
import roboflow
import cv2
from inference.models.utils import get_roboflow_model


# The function to be called

def seg_detect(im):
    """
    segmented_conjunctiva, diagnosis, conf = seg_detect( <YOUR_CV2_IMAGE_HERE> )
    """
    # Reading image
    # im=cv2.imread(img_path)
    im=cv2.resize(im, (640,640), interpolation = cv2.INTER_LINEAR)


    # Model 1: will DETECT Anemia
    detect_name = "anemia_detection"
    detect_version = "1"
    detect_key="KoMdsQBlrTd9ic7gQWl7"

    model1 = get_roboflow_model(
    model_id="{}/{}".format(detect_name, detect_version),
    api_key=detect_key)

    # Model 2: will SEGMENT conjucntiva
    seg_name = "conjunctiva-segmentation-2"
    seg_version = "2"
    seg_key="DBs1tkJ51Pb1J6YEDsqk"

    model2 = get_roboflow_model(
    model_id="{}/{}".format(seg_name, seg_version),
    api_key=seg_key)



    # Segmentation inference
    seg_result = model2.infer(image=im, confidence=0.5, iou_threshold=0.5)

    # Creating masked image
    mask = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.uint8)
    points=[]
    
    if seg_result:
        for i in seg_result[0].predictions[0].points:
            l=[int(i.x), int(i.y)]
            points.append(l) # Polygon points

        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1, cv2.LINE_AA)
        final_im = cv2.bitwise_and(im, im, mask=mask) # final masked image
        
        rim = final_im[:,:]

        # Sending segmented image for Anemia detection
        results = model1.infer(image=rim, confidence=0.5, iou_threshold=0.5)
        diagnosis=results[0].predictions[0].class_name
        conf=results[0].predictions[0].confidence

        # Final Output
        print(f'Model is -- {float(conf)*100 : .2f}% -- confident that the conjunctiva-palor is --{diagnosis}--')

        return rim, diagnosis, float(conf)*100
    else:
        return [], 'Not found', 0