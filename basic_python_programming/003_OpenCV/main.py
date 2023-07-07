import cv2
import dlib
from imutils import face_utils

# Task 1: rotate image
person_image = cv2.imread("person.jpg", cv2.IMREAD_COLOR)

person_height, person_width, person_channel = person_image.shape
person_matrix = cv2.getRotationMatrix2D((person_width / 2, person_height / 2), 270, 1)
person_image = cv2.warpAffine(
    person_image, person_matrix, (person_width, person_height)
)

# Task 2: resize image

person_ratio = 0.5
person_dim = (int(person_width * person_ratio), int(person_height * person_ratio))
person_resized = cv2.resize(person_image, person_dim, interpolation=cv2.INTER_AREA)

# Task 3: put sunglasses on person image
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_COLOR)
sunglasses_png = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
gray_sunglasses = cv2.cvtColor(sunglasses, cv2.COLOR_BGR2GRAY)

gray_person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

person_rects = detector(gray_person_image, 0)

# loop over the face detections
for i, rect in enumerate(person_rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    person_shape = predictor(gray_person_image, rect)
    person_shape = face_utils.shape_to_np(person_shape)

    # resize sunglasses to match each faces
    face_width = person_shape[15][0] - person_shape[1][0]
    x_of_eyes = person_shape[1][0]
    y_of_eyes = int((person_shape[15][1] + person_shape[1][1]) / 2)
    height, width = sunglasses.shape[:2]
    sunglasses_ratio = face_width / width
    sunglasses_dim = (face_width, int(height * sunglasses_ratio))
    sunglasses = cv2.resize(sunglasses, sunglasses_dim, interpolation=cv2.INTER_AREA)
    sunglasses_png = cv2.resize(
        sunglasses_png, sunglasses_dim, interpolation=cv2.INTER_AREA
    )
    gray_sunglasses = cv2.resize(
        gray_sunglasses, sunglasses_dim, interpolation=cv2.INTER_AREA
    )

    # copy sunglasses to image
    h, w = sunglasses.shape[:2]
    y_of_sunglasses = int(y_of_eyes - h * 0.8)
    person_crop = person_image[
        y_of_sunglasses : h + y_of_sunglasses, x_of_eyes : x_of_eyes + face_width
    ]

    for i in range(0, h):
        for j in range(0, w):
            if sunglasses_png[i, j, 3] > 0:
                person_crop[i, j, :] = sunglasses[i, j, :]

# Task 4: put sunglasses on cam image

cap = cv2.VideoCapture(0)

while True:
    sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_COLOR)
    sunglasses_png = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
    gray_sunglasses = cv2.cvtColor(sunglasses, cv2.COLOR_BGR2GRAY)

    # load the input image and convert it to grayscale
    _, image = cap.read()
    image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for i, rect in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # resize sunglasses to match each faces
        face_width = shape[15][0] - shape[1][0]
        x_of_eyes = shape[1][0]
        y_of_eyes = int((shape[15][1] + shape[1][1]) / 2)
        height, width = sunglasses.shape[:2]
        sunglasses_ratio = face_width / width
        sunglasses_dim = (face_width, int(height * sunglasses_ratio))
        sunglasses = cv2.resize(
            sunglasses, sunglasses_dim, interpolation=cv2.INTER_AREA
        )
        sunglasses_png = cv2.resize(
            sunglasses_png, sunglasses_dim, interpolation=cv2.INTER_AREA
        )
        gray_sunglasses = cv2.resize(
            gray_sunglasses, sunglasses_dim, interpolation=cv2.INTER_AREA
        )

        # copy sunglasses to image
        h, w = sunglasses.shape[:2]
        y_of_sunglasses = int(y_of_eyes - h * 0.8)
        crop = image[
            y_of_sunglasses : h + y_of_sunglasses, x_of_eyes : x_of_eyes + face_width
        ]

        for i in range(0, h):
            for j in range(0, w):
                if sunglasses_png[i, j, 3] > 0:
                    crop[i, j, :] = sunglasses[i, j, :]

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
