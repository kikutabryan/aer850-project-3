import cv2 as cv
import matplotlib.pyplot as plt


class ObjectMasking:
    @staticmethod
    def plot_cv_image(image, title=None):
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
    # Load the image
    image = cv.imread("motherboard_image.JPEG")
    pass
