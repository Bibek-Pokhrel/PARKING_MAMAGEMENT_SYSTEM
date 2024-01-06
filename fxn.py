def project():
    import cv2
    import matplotlib.pyplot as plt

    config_file="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozen_model="frozen_inference_graph.pb"

    model = cv2.dnn_DetectionModel(frozen_model,config_file)

    classlabel=[]
    file_name="coco.names"
    with open(file_name,"rt") as fpt:
        classlabel=fpt.read().rstrip('\n').split("\n")


    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean(127.5)
    model.setInputSwapRB(True)

    cap = cv2.VideoCapture('easy.mp4')

    if not cap.isOpened():
        cap= cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError('cannot open video')


    font_scale=2
    font=cv2.QT_FONT_NORMAL

    while True:
        ret,frame=cap.read()
        frame=cv2.resize(frame,(950,700))

        ClassIndex,confidence,bbox = model.detect(frame,confThreshold=0.45)

        if (len(ClassIndex)!=0):
            for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
                if (ClassInd<=80):
                    cv2.rectangle(frame,boxes,(255,0,0),2)
                    cv2.putText(frame,classlabel[ClassInd-1],(boxes[0]+5,boxes[1]+20),font,fontScale=font_scale,color=(0,255,0),thickness=1)
    
        cv2.imshow('Object Detection Tutorial' , frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



