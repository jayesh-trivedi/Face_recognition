# Face Recognition
This face recognizer implements the MTCNN and FaceNet model to recognise people in the given image. MTCNN algorithm is used to extract the face of the person, whereas the FaceNet system create the embedding vector for the face. A linear support vector classifier then classfies the given face vector. The current model has been trained on the 5 Celebratory Dataset. The model can recognise any person by adding their images to the dataset.
*Input*                      |  *Output* 
:-------------------------:|:-------------------------:
<img src="https://github.com/jayesh-trivedi/face_recognition/blob/master/images/test.jpg" height="480" width="380" ></a>   |  ![](https://github.com/jayesh-trivedi/face_recognition/blob/master/images/screenshot.png)

## Dataset
The [5 Celebratory dataset](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset) by DanB is used here.

This is a small dataset for experimenting with computer vision techniques. It has a training directory containing 14-20 photos each of the celebrities

  * Ben Afflek
  * Elton John
  * Jerry Seinfeld
  * Madonna
  * Mindy Kaling
  
The validation directory has 5 photos of each celebrity.

## Usage

#### 1. Installing Requirements and dependencies
After cloning this repository, run the following commands in your terminal window

```
cd face_recognition

pip install --user -r requirements.txt
```

#### 2. Loading dataset
The following script uses the extract_face method from the extract_face script to extract faces from the training and validation dataset and then saves the dataset file as npz object for future use.

`python load_dataset.py`

#### 3. Creating embeddings
Run the create_embedding script to generate the face embedding vectors for all the extracted faces form the training and validation dataset. The embedding are created using the FaceNet model. The embeddings are then saved as npz object for future use.

`python create_embedding.py`

#### 4. Recognizing Faces
The face_recognizer script takes the input image, creates it's embeddings and then classifies it using a linear support vector classifer.

`python face_recognizer.py`

To test a particular image save them as test.jpg in the image directory of the project directory.

The Webcam can also be used to capture and recognize a person just by changing images/test.jpg to images/cap.jpg in the extract_face function

```python
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.write('images/cap.jpg', frame)
cap.release()
face = extract_face('images/test.jpg')
```

## Using custom dataset
Custom dataset can be used with this model by arranging the images of the people in the following structure and then then running the scripts in the above mentioned fashion. 


   |- face-recognizer/
  
      |- dataset/
         |- data/
            |- train/
               |- person1/
                  |- p11.jpg
                  |- p12.jpg
                     .
                     .
                  |- p1N.jpg
               |- person2/
                  |- p21.jpg
                     .
                     .
                  .
                  .
               |- personN/
                  . 
                  .
            |- val/
               |- person1/
                  |- p11.jpg
                  |- p12.jpg
                     .
                     .
                  |- p1N.jpg
               |- person2/
                  |- p21.jpg
                     .
                     .
                  .
                  .
               |- personN/
                     .
                     .
              
               
                
