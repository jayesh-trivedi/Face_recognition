# Face Recognition
This face recognizer implements the MTCNN and FaceNet model to recognise people in the given image. MTCNN algorithm is used to extract the face of the person, whereas the FaceNet system create the embedding vector for the face. A linear support vector classifier then classfies the given face vector. The current model has been trained on the 5 Celebratory Dataset. The model can recognise any person by adding their images to the dataset.
*Input*                      |  *Output* 
:-------------------------:|:-------------------------:
<img src="https://github.com/jayesh-trivedi/face_recognition/blob/master/images/test.jpg" height="480" width="380" ></a>   |  ![](https://github.com/jayesh-trivedi/face_recognition/blob/master/images/screenshot.png)
