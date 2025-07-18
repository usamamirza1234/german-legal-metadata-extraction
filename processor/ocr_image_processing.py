import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image
import os


class ImagePreprocessor:
    """
    ImagePreprocessor - Simple Guide

    This class helps prepare images for better text recognition (OCR). 
    It applies various filters and adjustments to make text clearer and easier to read by computer programs.

    Core Functions:
    1. invert_image() - Flip Colors: Makes black text white, and white background black
    2. rescale_image(scale_factor) - Resize Image: Makes image bigger or smaller
    3. grayscale() - Remove Colors: Converts colorful image to black, white, and gray only
    4. binarize_image(threshold) - Make Pure Black & White: Converts gray pixels to pure black or white
    5. noise_removal() - Clean Up Spots: Removes small dots, specks, and unwanted marks
    6. thin_font()/thick_font() - Adjust Text Thickness: Make letters thinner or thicker
    7. deskew() - Straighten Tilted Text: Rotates image to make text lines horizontal
    8. remove_borders() - Crop to Content: Cuts away empty white space around text
    9. add_borders() - Add White Space: Adds padding around the image
    10. remove_white_areas() - NEW: Remove large white/blank areas from document
    """

    def __init__(self, image_path):
        """Initialize the preprocessor with an image path."""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"The file '{image_path}' does not exist or cannot be read.")

    def display(self, im_path):
        """Display image with actual size using matplotlib."""
        dpi = 80
        im_data = plt.imread(im_path)

        height, width = im_data.shape[:2]

        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Hide spines, ticks, etc.
        ax.axis('off')

        # Display the image.
        ax.imshow(im_data, cmap='gray')

        plt.show()

    def remove_white_areas(self, image=None, white_threshold=240, min_content_area=1000):
        """
        Remove large white/blank areas from document

        What it does: Detects and removes large white areas while preserving text content
        Variables:
        - white_threshold: Pixel intensity above which is considered "white" (0-255, default 240)
        - min_content_area: Minimum area in pixels to consider as content (default 1000)

        Effect: Focuses on actual document content, removes margins and blank spaces
        Returns: Cropped image with white areas removed
        """
        if image is None:
            image = self.original_image

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Create binary mask where non-white areas are marked
        # Values below white_threshold are considered content (non-white)
        _, binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY_INV)

        # Find contours of non-white areas
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # If no contours found, return original image
            return image

        # Filter contours by area to remove noise
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_content_area]

        if not valid_contours:
            # If no valid contours, return original image
            return image

        # Find bounding box that encompasses all content areas
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Add small padding to ensure we don't cut off text
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        # Crop the image to the content area
        if len(image.shape) == 3:
            cropped = image[y:y + h, x:x + w]
        else:
            cropped = image[y:y + h, x:x + w]

        return cropped

    def smart_crop_content(self, image=None, margin_threshold=0.95):
        """
        Alternative method: Smart cropping based on content density

        What it does: Analyzes rows and columns to find content boundaries
        Variables:
        - margin_threshold: Percentage of white pixels to consider as margin (default 0.95)

        Effect: More precise content detection, good for documents with varying layouts
        """
        if image is None:
            image = self.original_image

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate content density for each row and column
        height, width = gray.shape

        # For each row, calculate percentage of white pixels
        row_whiteness = []
        for i in range(height):
            white_pixels = np.sum(gray[i, :] > 240)
            whiteness_ratio = white_pixels / width
            row_whiteness.append(whiteness_ratio)

        # For each column, calculate percentage of white pixels
        col_whiteness = []
        for j in range(width):
            white_pixels = np.sum(gray[:, j] > 240)
            whiteness_ratio = white_pixels / height
            col_whiteness.append(whiteness_ratio)

        # Find content boundaries
        # Top boundary
        top = 0
        for i in range(height):
            if row_whiteness[i] < margin_threshold:
                top = i
                break

        # Bottom boundary
        bottom = height - 1
        for i in range(height - 1, -1, -1):
            if row_whiteness[i] < margin_threshold:
                bottom = i
                break

        # Left boundary
        left = 0
        for j in range(width):
            if col_whiteness[j] < margin_threshold:
                left = j
                break

        # Right boundary
        right = width - 1
        for j in range(width - 1, -1, -1):
            if col_whiteness[j] < margin_threshold:
                right = j
                break

        # Add small padding
        padding = 10
        top = max(0, top - padding)
        bottom = min(height - 1, bottom + padding)
        left = max(0, left - padding)
        right = min(width - 1, right + padding)

        # Crop the image
        if len(image.shape) == 3:
            cropped = image[top:bottom, left:right]
        else:
            cropped = image[top:bottom, left:right]

        return cropped

    def extract_fraktur_text(self, image_path):
        """Extract text from an image using Tesseract with the Fraktur language model."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file '{image_path}' does not exist.")

        # Load image
        image = cv2.imread(image_path)

        # Preprocess image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Optional: add more preprocessing like thresholding, denoising, etc.

        # Extract text using Fraktur model
        text = pytesseract.image_to_string(gray, lang='deu_frak')
        return text

    def invert_image(self, image=None):
        """
        1. Inverted Images - Flip Colors
        What it does: Makes black text white, and white background black
        When to use: Sometimes inverted images work better for text recognition
        Variables: None to change
        Effect: Complete color reversal
        """
        if image is None:
            image = self.original_image
        inverted_image = cv2.bitwise_not(image)
        return inverted_image

    def rescale_image(self, image=None, scale_factor=1.0):
        """
        2. Rescaling - Resize Image
        What it does: Makes image bigger or smaller
        Variables:
        - scale_factor = 1.0 → Same size
        - scale_factor = 2.0 → Double size
        - scale_factor = 0.5 → Half size
        Effect on text: Bigger images often give better OCR results
        """
        if image is None:
            image = self.original_image

        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dimensions = (width, height)

        rescaled = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
        return rescaled

    def grayscale(self, image=None):
        """
        Convert image to grayscale - Remove Colors
        What it does: Converts colorful image to black, white, and gray only
        Variables: None to change
        Effect: Often improves text recognition, removes color distractions
        """
        if image is None:
            image = self.original_image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def binarize_image(self, image=None, threshold=105):
        """3. Binarization - Convert to binary (black and white) image."""
        if image is None:
            image = self.original_image

        if len(image.shape) == 3:
            gray_image = self.grayscale(image)
        else:
            gray_image = image

        thresh, im_bw = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return im_bw

    def noise_removal(self, image):
        """
        4. Noise Removal - Clean Up Spots
        What it does: Removes small dots, specks, and unwanted marks
        Variables: Kernel sizes are hardcoded (1x1 pixels)
        Effect: Cleaner image but may remove small text details
        """
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image

    def thin_font(self, image):
        """
        5. Dilation and Erosion - Make font thinner
        What it does: Makes letters thinner
        Variables: Kernel size (2x2 pixels) controls how much change
        Effect: Better for thick, bold text
        """
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image

    def thick_font(self, image):
        """
        5. Dilation and Erosion - Make font thicker
        What it does: Makes letters thicker
        Variables: Kernel size (2x2 pixels) controls how much change
        Effect: Better for very thin, light text
        """
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image

    def get_skew_angle(self, image):
        """6. Rotation / Deskewing - Get skew angle of image."""
        # Prep image, copy, convert to gray scale, blur, and threshold
        new_image = image.copy()
        gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=2)

        # Find all contours
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find largest contour and surround in min area box
        largest_contour = contours[0]
        print(len(contours))
        min_area_rect = cv2.minAreaRect(largest_contour)
        cv2.imwrite("../temp/boxes.jpg", new_image)
        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        angle = min_area_rect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    def rotate_image(self, image, angle):
        """Rotate the image around its center."""
        new_image = image.copy()
        (h, w) = new_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_image = cv2.warpAffine(new_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return new_image

    def deskew(self, image):
        """
        6. Rotation / Deskewing - Straighten Tilted Text
        What it does: Rotates image to make text lines horizontal
        Variables: Automatically detects angle
        Effect: Improves OCR accuracy for crooked scanned documents
        """
        angle = self.get_skew_angle(image)
        return self.rotate_image(image, -1.0 * angle)

    def remove_borders(self, image):
        """
        7. Removing Borders - Crop to Content
        What it does: Cuts away empty white space around the text
        Variables: None to change
        Effect: Focuses OCR on actual content, may improve accuracy
        """
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        cnt = cnts_sorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y + h, x:x + w]
        return crop

    def add_borders(self, image, border_size=150, color=[255, 255, 255]):
        """
        8. Missing Borders - Add White Space
        What it does: Adds padding around the image
        Variables:
        - border_size = 150 → 150 pixels of padding (default)
        - color = [255,255,255] → White padding (default)
        Effect: Sometimes OCR works better with white space around text
        """
        top, bottom, left, right = [border_size] * 4
        image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return image_with_border

    def add_transparency(self, image, alpha_value=255):
        """
        9. Transparency / Alpha Channel - Add transparency
        What it does: Add alpha channel to image for transparency effects
        Variables: alpha_value (0=fully transparent, 255=fully opaque)
        Effect: Useful for overlaying images or creating transparent backgrounds
        """
        if len(image.shape) == 3:
            # Convert BGR to BGRA
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            # Convert grayscale to BGRA
            rgba_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)

        # Set alpha channel
        rgba_image[:, :, 3] = alpha_value
        return rgba_image

    def remove_transparency(self, image, background_color=[255, 255, 255]):
        """Remove alpha channel and replace with background color."""
        if len(image.shape) == 4:
            # Create a background image
            background = np.full(image.shape[:3], background_color, dtype=image.dtype)

            # Normalize alpha channel to 0-1 range
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0

            # Blend the image with background
            result = image[:, :, :3].astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)
            return result.astype(np.uint8)
        else:
            return image

    def preprocess_pipeline(self, output_dir="temp/", save_intermediate=True, remove_white_spaces=True):
        """
        Complete Processing Pipeline - Do Everything
        Runs these steps in order:
        1. Remove white areas (if enabled)
        2. Invert colors
        3. Convert to grayscale
        4. Make black & white (binarize)
        5. Remove noise
        6. Remove borders
        7. Add new borders

        Variables:
        - remove_white_spaces: Boolean to enable/disable white space removal

        Files saved: Each step saves an image file so you can see the changes
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Start with original image
        current_image = self.original_image

        # 0. Remove white areas (NEW STEP)
        if remove_white_spaces:
            current_image = self.remove_white_areas(current_image)
            if save_intermediate:
                cv2.imwrite(f"{output_dir}white_removed.jpg", current_image)

        # 1. Inverted Images
        inverted_image = self.invert_image(current_image)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}inverted.jpg", inverted_image)

        # Convert to grayscale
        gray_image = self.grayscale(inverted_image)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}gray.jpg", gray_image)

        # 3. Binarization
        bw_image = self.binarize_image(gray_image)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}bw_image.jpg", bw_image)

        # 4. Noise Removal
        no_noise = self.noise_removal(bw_image)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}no_noise.jpg", no_noise)

        # 7. Remove Borders
        no_borders = self.remove_borders(no_noise)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}no_borders.jpg", no_borders)

        # 8. Add Borders
        with_borders = self.add_borders(no_borders)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}image_with_border.jpg", with_borders)

        return with_borders