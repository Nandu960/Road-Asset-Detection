# Road-Asset-Detection
The project aims to develop a Road Asset Detection system using convolution neural network. We had modified existing tiny YOLO model for the detections.

Our project is an extended work on the blooming Computer Vision on  localization of relevant objects. We implement a real-time system capable of accounting for different road assets like traffic signs,potholes,lanes,zebra lines and vehicles . We have used the YOLO model for this purpose.We used this model as they have superior localization and classification in addition to easier modification of layers,via transfer learning.We further seek to implement a fully functional real time system on Indian Roads.

Epochs	             120

Dataset size	       
                     
                     Pascal VOC: 2029
                     Potholes : 630                     
                     LISA: 6618

Accuracy	           

                      Pothole Detection :84
                      Vehicles :88
                      Traffic Signs:87
                      
This accuracy can be increased further by adding more images to the dataset and training them further.

Experiment Results

After building the models and training them, we had to test on various images and videos. The following are the snap shots of a few of them detecting various assets which pays off our objective.

![Image description](https://github.com/Nandu960/Road-Asset-Detection/blob/master/images/lane_det.PNG|width=500)

![Image description](https://github.com/Nandu960/Road-Asset-Detection/blob/master/images/pothole.PNG)

![Image description](https://github.com/Nandu960/Road-Asset-Detection/blob/master/images/traffic%20sign.PNG)

