from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.applications.mobilenet import preprocess_input
import numpy as np
import cv2
from flask import Flask , request , jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)



@app.route('/image' , methods=['GET' , 'POST'])
def img():
    model = load_model('image_classification_model.h5')
    image = request.files.get('image')
    if image is None:
        return jsonify({'error': 'Invalid request'}), 400

    # Save the image to a file
    filename = secure_filename(image.filename)
    image.save(filename)

    def predict_image(image_path):
        img = load_img(image_path, target_size=(224, 224) )
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)

        return jsonify({'data' : predicted_class})


    return predict_image(filename)


@app.route('/video' , methods=['GET' , 'POST'])
def video():
    model = load_model('video_classification_model.h5')

    video = request.files.get('video')
    if video is None:
        return jsonify({'error': 'Invalid request'}), 400

    # Save the video to a file
    filename = secure_filename(video.filename)
    video.save(filename)
    def video_frame_generator(video_path, num_frames):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
            break
        
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
          frame = cv2.Canny(frame, 100, 200)
          frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
          frames.append(frame)
          if len(frames) == num_frames:
              break
          cap.release()
        return frames

    def preprocess_data(video_path, num_frames):
        frames = video_frame_generator(video_path, num_frames)
        X = []
        for frame in frames:
          img = cv2.resize(frame, (224, 224 ))
          img = img_to_array(img)
          img = preprocess_input(img)
          X.append(img)
        X = np.array(X)
        return X
    
    
    def predict_video(video_path):


        X_test = preprocess_data(video_path, 20)


        predictions = model.predict(X_test)

        predicted_class_index = np.argmax(predictions[0])

        return predicted_class_index


    return predict_video(filename)

if __name__=='__main__':
    app.run(host='0.0.0.0' , port=5001  , debug=True)