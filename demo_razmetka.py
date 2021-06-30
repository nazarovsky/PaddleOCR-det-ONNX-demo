# demo for ONNX text areas detector
import os, sys, glob, shutil
import numpy as np
import cv2
import onnxruntime

from db_postprocess import DBPostProcess

CKPT_FOLDER = 'output/det_db/'
CKPT_TEST_WEIGHTS_FILE = 'det_db.onnx'
DET_DB_THRESH  = 0.3
DET_BOX_THRESH = 0.6
DET_MAX_CANDIDATES = 1000
DET_UNCLIP_RATIO   = 1.5

def imagenet_preprocess(img_in):
    img  = np.array(img_in, np.float32)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img /= 255.0
    img = (img - mean) / std
    return img

def detect(session, image_src):

    print('ONNX input size is {}'.format(session.get_inputs()[0].shape))
    print('ONNX output size is {}'.format(session.get_outputs()[0].shape))
    if len(image_src.shape) == 2:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)

    IN_IMAGE_H, IN_IMAGE_W, IN_IMAGE_D = image_src.shape

    IN_ONNX_H  = session.get_inputs()[0].shape[2]
    IN_ONNX_W  = session.get_inputs()[0].shape[3]
    IN_PROP_H  = float(IN_ONNX_H/IN_IMAGE_H)
    IN_PROP_W  = float(IN_ONNX_W/IN_IMAGE_W)
    # resize input
    resized = cv2.resize(image_src, (IN_ONNX_W, IN_ONNX_H), interpolation=cv2.INTER_LINEAR)
    img_in = imagenet_preprocess(resized)
    print('img_in size is {} - {}'.format(img_in.shape, img_in.dtype))
    xb = np.array(np.expand_dims(np.transpose(img_in, (2, 0, 1)), axis = 0),np.float32)
    print('xb size is {}'.format(xb.shape))
    # debug part - save input tensor to to txt
#    xb1 = np.squeeze(xb)
#    for k in range(3):
#        np.savetxt('image_{}.txt'.format(k), xb1[k,:,:] ,fmt='%+3.3f',delimiter=' ') 

    # run onnx inference session
    x = xb if isinstance(xb, list) else [xb]
    feed = dict([(input.name, x[n]) for n, input in enumerate(session.get_inputs())])
    pred_onnx   = session.run(None, feed)
    print('pred_onnx output shape is {}'.format(pred_onnx[0].shape))
    # construct dict and shape list for postprocessing 
    outs_dict = {'maps': pred_onnx[0]}
    shape_list = [[IN_IMAGE_H, IN_IMAGE_W, IN_PROP_H, IN_PROP_W]]
    print('shape_list is {}'.format(shape_list))
    db = DBPostProcess(thresh=DET_DB_THRESH, box_thresh=DET_BOX_THRESH, max_candidates=DET_MAX_CANDIDATES,unclip_ratio=DET_UNCLIP_RATIO, use_dilation=False)
    boxes_obj = db(outs_dict, shape_list)
    boxes = boxes_obj[0]['points']
    # draw boxes on added_image
    added_image = image_src
    for i, box in enumerate(boxes):
        print('Box[{}] = {}'.format(i,box))
        box_np = np.array(box, dtype = np.int32)
        added_image = cv2.polylines(added_image, [box_np] ,True, (255,255,255), 2)

    cv2.imshow('added_image',added_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return added_image

def main():
    # define model
    onnx_model_filename = os.path.join(CKPT_FOLDER,CKPT_TEST_WEIGHTS_FILE)
    sess = onnxruntime.InferenceSession(onnx_model_filename)
    for i, file_path in enumerate(glob.glob(os.path.join('output','*.jpg'))):
        fname =  os.path.splitext(file_path)[0]
        frame =  cv2.imread(file_path)
        image_detected = detect(sess, frame)
#        cv2.imwrite('{}_out.jpg'.format(fname), image_detected)

if __name__ == '__main__':
    main()
