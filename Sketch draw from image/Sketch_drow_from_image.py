import cv2
import numpy as np

def sketch_transform(image):
    """Convert an image to a pencil sketch."""
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7, 7), 0)
    image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    _, mask = cv2.threshold(image_canny, 30, 255, cv2.THRESH_BINARY_INV)
    return mask

# Initialize camera capture
cam_capture = cv2.VideoCapture(0)

if not cam_capture.isOpened():
    print("Error: Could not open video device.")
else:
    # Capture a single frame
    ret, frame = cam_capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
    else:
        # Display the frame to select ROI
        print("Select ROI with the mouse, press Enter/Space to confirm, or press 'c' to cancel.")

        cv2.imshow("Select ROI", frame)

        # Handle key events
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # If 'c' is pressed, cancel the selection
                print("ROI selection canceled by pressing 'c' button!")
                break
            elif key == 13 or key == 32:  # If Enter or Space is pressed, proceed to select ROI
                r = cv2.selectROI("Select ROI", frame, False, False)
                cv2.destroyWindow("Select ROI")  # Close the ROI selection window

                # Ensure the ROI is valid
                if r[2] > 0 and r[3] > 0:
                    # Crop the region of interest from the frame
                    rect_img = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

                    # Apply the sketch transformation to the ROI
                    sketcher_rect = sketch_transform(rect_img)

                    # Convert the sketch to 3 channels to match the original frame
                    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2BGR)

                    # Replace the original ROI in the frame with the sketched ROI
                    frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])] = sketcher_rect_rgb

                    # Display the result
                    cv2.imshow("Sketcher ROI", frame)
                    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
                else:
                    print("No valid ROI selected or ROI is too small.")
                break

# Release the capture and close all windows
cam_capture.release()
cv2.destroyAllWindows()
