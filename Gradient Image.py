import numpy as np
import cv2

def generate_gradient_image(width, height, start_color, end_color, start_point, end_point):
    # Create an empty image with 3 color channels (BGR)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the vector from start to end point
    vector = np.array([end_point[0] - start_point[0], end_point[1] - start_point[1]])
    vector_length = np.linalg.norm(vector)
    
    # Check if the vector length is zero to avoid division by zero
    if vector_length == 0:
        raise ValueError("Start point and end point cannot be the same.")

    unit_vector = vector / vector_length

    for y in range(height):
        for x in range(width):
            # Calculate the vector from the start point to the current pixel
            current_vector = np.array([x - start_point[0], y - start_point[1]])
            
            # Project the current vector onto the unit vector to get the distance along the gradient line
            projection_length = np.dot(current_vector, unit_vector)
            
            # Normalize the distance to the range [0, 1]
            t = projection_length / vector_length
            t = np.clip(t, 0, 1)
            
            # Interpolate between the start and end colors (note the BGR format)
            for i in range(3):
                image[y, x, 2 - i] = int(start_color[i] * (1 - t) + end_color[i] * t)

    return image

# Define the size of the image
width, height = 800, 600

# Define the start and end colors (BGR)
start_color = (255, 0, 0)  # Red (in BGR format: Blue, Green, Red)
end_color = (0, 0, 255)    # Blue (in BGR format: Blue, Green, Red)

# Define the start and end points
start_point = (150, 200)
end_point = (150, 555)

# Generate the gradient image
gradient_image = generate_gradient_image(width, height, start_color, end_color, start_point, end_point)

# Save the image
cv2.imwrite('gradient_image.png', gradient_image)

# Show the image
cv2.imshow('Gradient Image', gradient_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
