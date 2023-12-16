
import cv2

# Read an image
image = cv2.imread('brother.jpg')

# Define the text, font, and position
text = 'Hello, OpenCV!'
font = cv2.FONT_HERSHEY_SIMPLEX
position = (50, 50)  # (x, y) coordinates of the bottom-left corner of the text

# Define text color and thickness
color = (255, 0, 0)  # (B, G, R) values
thickness = 2

# Put text on the image
cv2.putText(image, text, position, font, 1, color, thickness, cv2.LINE_AA)

# Display the result
cv2.imshow('Image with Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()