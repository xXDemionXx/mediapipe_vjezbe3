# Libraries
import cv2
import mediapipe as mp
from deepface import DeepFace
#

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2


IMAGE_FILES = ['one_face.jpg', 'brother.jpg', 'test_leaves.jpg']
path_of_modified_images = 'C:/Users/demio/Python projects/mediapipe_vjezbe3/image_results/Output_image'

def face_detection_from_image(IMAGE_FILES, path_of_modified_images):

    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #
            if results.detections:
                for detection in results.detections:
                    bounding_box = detection.location_data.relative_bounding_box
                    image_h, image_w, _ = image.shape
                    box_h = int(bounding_box.height * image_h)
                    box_w = int(bounding_box.width * image_w)
                    box_x_start_point = int(bounding_box.xmin * image_w)
                    box_y_start_point = int(bounding_box.ymin * image_h)
                    cv2.rectangle(image, (box_x_start_point, box_y_start_point), (box_x_start_point + box_w, box_y_start_point + box_h), color, thickness)
                    face_detections = DeepFace.analyze(image, actions=['age', 'gender', 'race', 'emotion'])
                    start_position = (box_x_start_point + box_w, box_y_start_point + box_h)
                    picture_text = [str(face_detections[0]['age']), str(face_detections[0]['dominant_race']), str(face_detections[0]['dominant_gender']), str(face_detections[0]['dominant_emotion'])]
                    for i, line_of_text in enumerate(picture_text):
                        position = (start_position[0], start_position[1] + (-i) * 30)  # Adjust y-coordinate for each line
                        cv2.putText(image, line_of_text, position, font, 1, color, thickness, cv2.LINE_AA)
                    #cv2.imshow("test", image)
                    #cv2.waitKey(0)
            cv2.imwrite(path_of_modified_images + str(idx) + '.jpg', image)


def camera_face_detection():
    video = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        while video.isOpened():
            success, image = video.read()
            if not success:
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #
            if results.detections:
                for detection in results.detections:
                    bounding_box = detection.location_data.relative_bounding_box
                    image_h, image_w, _ = image.shape
                    box_h = int(bounding_box.height * image_h)
                    box_w = int(bounding_box.width * image_w)
                    box_x_start_point = int(bounding_box.xmin * image_w)
                    box_y_start_point = int(bounding_box.ymin * image_h)
                    cv2.rectangle(image, (box_x_start_point, box_y_start_point), (box_x_start_point + box_w, box_y_start_point + box_h), color, thickness)
                    #
                    face_detections = DeepFace.analyze(image, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
                    start_position = (box_x_start_point + box_w, box_y_start_point + box_h)
                    picture_text = [str(face_detections[0]['age']), str(face_detections[0]['dominant_race']), str(face_detections[0]['dominant_gender']), str(face_detections[0]['dominant_emotion'])]
                    for i, line_of_text in enumerate(picture_text):
                        position = (start_position[0], start_position[1] + (-i) * 30)
                        cv2.putText(image, line_of_text, position, font, 1, color, thickness, cv2.LINE_AA)
                        #
            cv2.imshow('Face', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video.release()

#face_detection_from_image(IMAGE_FILES, path_of_modified_images)
camera_face_detection()


