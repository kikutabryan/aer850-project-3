import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


class ObjectMasking:
    @staticmethod
    def get_extracted_image(image: np.ndarray) -> np.ndarray:
        """Extracts the object from the given image using a mask.

        Args:
            image (np.ndarray): The input image from which the object will be extracted.

        Returns:
            np.ndarray: The image with the background removed, showing only the extracted object.
        """
        # Get the object mask
        mask = ObjectMasking._generate_mask(image)

        # Remove background using mask
        masked_image = cv.bitwise_and(image, image, mask=mask)

        return masked_image

    @staticmethod
    def _generate_mask(image: np.ndarray) -> np.ndarray:
        """Generates a binary mask for the given image to isolate the object.

        Args:
            image (np.ndarray): The input image from which the mask will be generated.

        Returns:
            np.ndarray: A binary mask where the object is white (255) and the background is
            black (0).
        """
        # Convert image to grayscale, and equalize to help with shadows
        gray_image = cv.equalizeHist(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

        # Blur the image to remove noise
        blurred_image = cv.GaussianBlur(gray_image, (21, 21), 0)

        # Threshold (inv since background is brighter than board)
        _, thresh_image = cv.threshold(blurred_image, 127, 255, cv.THRESH_BINARY_INV)

        # Detect edges
        edges = cv.Canny(thresh_image, 50, 150)

        # Fill gaps in edges
        kernel = np.ones((31, 31), np.uint8)
        closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv.findContours(
            closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # Find the largest contour
        largest_contour = max(contours, key=cv.contourArea)

        # Create a blank mask
        mask = np.zeros_like(gray_image)

        # Draw contours on the mask
        cv.drawContours(mask, [largest_contour], -1, (255), thickness=cv.FILLED)

        return mask

    @staticmethod
    def plot_cv_image(image: np.ndarray, title: str = None):
        """Displays an image using Matplotlib.

        Args:
            image (numpy.ndarray): The image to be displayed, in BGR format.
            title (str, optional): The title of the plot. Defaults to None.
        """
        plt.axis("off")
        image_RGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert to RGB
        plt.imshow(image_RGB)
        if title:
            plt.title(title)


if __name__ == "__main__":
    # Define the image path
    image_name = "motherboard_image.JPEG"
    image_dir = os.path.join("images")
    image_path = os.path.join(image_dir, image_name)

    # Load the image
    image = cv.imread(image_path)

    # Extract object
    extracted_image = ObjectMasking.get_extracted_image(image=image)

    # Show the image
    cv.imshow("Extracted Image", extracted_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
