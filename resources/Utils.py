from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import cv2

mtcnn = MTCNN(keep_all=False)


# Constants for canonical landmarks and image size
CANONICAL_LANDMARKS = np.array([
    [35.3436, 60.0562],  # Left eye
    [76.2041, 60.0016],  # Right eye
    [56.6933, 83.7126],  # Nose tip
    [39.3617, 107.2982], # Left mouth corner
    [73.0954, 107.0652]  # Right mouth corner
], dtype=np.float32)

IMAGE_SIZE = (112, 112)

def preprocess_image(image):
    """
    Aligns and preprocesses a face image using MTCNN for detection and landmarks.

    Parameters:
    - image (PIL.Image): The input image containing a face.

    Returns:
    - aligned_face (PIL.Image): The aligned and preprocessed face image.
    """
    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Detect face boxes, probabilities, and landmarks
    boxes, probs, landmarks = mtcnn.detect(image_np, landmarks=True)

    # If no landmarks are detected, return the original image
    if landmarks is None or len(landmarks) == 0:
        return image

    # Get the detected landmarks for the first face
    detected_landmarks = landmarks[0].astype(np.float32)

    # Compute the similarity transformation
    transformation_matrix = cv2.estimateAffinePartial2D(detected_landmarks, CANONICAL_LANDMARKS)[0]

    # Apply the transformation to the image
    aligned_face = cv2.warpAffine(image_np, transformation_matrix, IMAGE_SIZE)

    # Convert the aligned face back to a PIL Image and return
    aligned_face = Image.fromarray(aligned_face.astype(np.uint8))
    return aligned_face