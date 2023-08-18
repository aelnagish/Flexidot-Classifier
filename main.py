import cv2
import os
import numpy as np
from pdf2image import convert_from_path


def contour_sort_key(contour):
    x, y, _, _ = cv2.boundingRect(contour)
    return y, x


def is_valid_input_path(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return False

    return True


def test_output_path(filepath):
    if not os.path.exists(filepath):
        print(f'Creating directory {filepath}.')
        os.makedirs(filepath)
    else:
        # Check if the output directory is empty
        if len(os.listdir(filepath)) > 0:
            # Delete all files in the output directory
            print(f'Deleting all files in {filepath}.')
            for file in os.listdir(filepath):
                file_path = os.path.join(filepath, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)


def line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def split_pdf_pages(input_dir, output_dir):

    is_valid_input_path(input_dir)
    test_output_path(output_dir)

    file_name = os.path.basename(input_dir)
    file_name = file_name.split(".")[0]

    images = convert_from_path(input_dir)

    for i, image in enumerate(images):

        page_number = str(i).zfill(4)
        unique_pdf_filename = f'page{page_number}_{file_name}.pdf'
        images[i].save(os.path.join(output_dir, unique_pdf_filename), 'PDF')


def convert_pdf_to_png(input_dir, output_dir):

    is_valid_input_path(input_dir)
    test_output_path(output_dir)

    file_name = os.path.basename(input_dir)
    file_name = file_name.split(".")[0]

    images = convert_from_path(input_dir)

    for i, image in enumerate(images):

        images[i].save(os.path.join(output_dir, f'page{i}_{file_name}.png'), 'PNG')

    return images


def detect_single_plots(input_dir, output_dir):

    is_valid_input_path(input_dir)
    test_output_path(output_dir)

    # Get a list of PNG files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    min_width = 100
    min_height = 100

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)

        # Load the image using OpenCV
        img = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to obtain binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        plot_masks = []  # Store individual plot masks
        contours_of_plots = []

        # Iterate over contours
        for contour in contours:
            # Approximate the contour as a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon has four sides (a square)
            if len(approx) == 4:
                # Extract the bounding rectangle coordinates
                x, y, w, h = cv2.boundingRect(approx)

                # Check the size treshold
                if w >= min_width and h >= min_height:
                    # Shrink the mask to the diagonal line coordinates
                    mask_shrunk = img[y:y + h, x:x + w].copy()

                    # Convert the mask to grayscale
                    gray_mask = cv2.cvtColor(mask_shrunk, cv2.COLOR_BGR2GRAY)

                    # Apply binary thresholding to obtain a binary image
                    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

                    # Find contours in the binary mask
                    contours_mask, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Store the largest contour in plot_masks
                    contours_of_plots.append((x, y, w, h))

                    # Add the shrunken mask to the list
                    plot_masks.append(mask_shrunk)

        # Sort the contours based on their top-left corner coordinates with several pixels tolerance
        tolerance = 10
        contours_of_plots.sort(key=lambda rect: (rect[1] // tolerance, rect[0] // tolerance))

        for i, (x, y, w, h) in enumerate(contours_of_plots):
            plot_mask = img[y:y + h, x:x + w]  # Create the mask from the image

            filename = f'{i}.png'
            plot_mask_path = os.path.join(output_dir, filename)
            cv2.imwrite(plot_mask_path, plot_mask)

        return contours_of_plots


def detect_colored_plots(input_dir, output_dir):
    is_valid_input_path(input_dir)
    test_output_path(output_dir)

    # Get a list of PNG files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    plot_numbers = []

    # Read the plot mask image
    for image in image_files:
        img_path = os.path.join(input_dir, image)  # Construct the complete file path
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        file_number = image.split('.')[0]

        # Convert the image to the HSV color space for better color thresholding
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply color thresholding to detect green regions
        green_mask = cv2.inRange(hsv_img, (30, 75, 20), (90, 255, 255))

        # Check if any green color is found within the plot mask
        if np.any(green_mask > 0):
            print(f'Green color found within the plot mask {file_number}')

            plot_numbers.append(file_number)
            # Save the green mask image for visualization
            output_path = os.path.join(output_dir, f"green_mask_{image}")
            cv2.imwrite(output_path, green_mask)

    return plot_numbers


def remove_noise(input_dir, output_dir):

    is_valid_input_path(input_dir)
    test_output_path(output_dir)

    for filename in os.listdir(input_dir):
        # Read the plot mask image
        img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Apply opening operation to reduce point-wise noise
        # kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]], dtype=np.uint8)
        img_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # dilation
        img_close = cv2.morphologyEx(img_open, cv2.MORPH_DILATE, kernel, iterations=2)

        # Save the new mask
        new_filename = os.path.splitext(filename)[0] + '_clean.png'
        cv2.imwrite(os.path.join(output_dir, new_filename), img_close)


def crop_images(input_dir, output_dir):

    is_valid_input_path(input_dir)
    test_output_path(output_dir)

    for filename in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)

        bottom_left_crop = 37
        top_right_crop = 15

        # Get the dimensions of the image
        height, width = img.shape[:2]

        img_cropped = img[top_right_crop:height-bottom_left_crop, bottom_left_crop:width-top_right_crop]

        # Save the new mask
        new_filename = os.path.splitext(filename)[0] + '_cropped.png'
        cv2.imwrite(os.path.join(output_dir, new_filename), img_cropped)


def ltr_retrotransposon_check(img, contours):

    if len(contours) > 3:

        ltr_left = False
        ltr_right = False
        ltr = False
        # Get the two largest contours
        largest_contours = contours[1:3]

        # Draw the largest contours on the image
        # cv2.drawContours(img_color, largest_contours, -1, (0, 255, 0), 2)
        # contours_filename = os.path.splitext(filename)[0] + '_contours.png'
        # cv2.imwrite(os.path.join(output_dir, contours_filename), img_color)

        for contour in largest_contours:
            x, y, w, h = cv2.boundingRect(contour)

            contour_area = cv2.contourArea(contour)
            contour_width = contour_area / h

            # Check if squares or blobs were detected --> no LTR
            if contour_width > 15:
                return False

            else:
                box_area = w * h

                # Check conditions for each contour
                if x <= 0.05 * img.shape[1] and y + h >= img.shape[0] * 0.95:
                    ltr_left = True
                elif y <= 0.05 * img.shape[0] and x + w >= img.shape[1] * 0.95:
                    ltr_right = True
                    if box_area > 250:
                        ltr = True

                if ltr_left is True and ltr_right is True and ltr is True:
                    return True
                    break
        else:
            return False


def tandem_retro_check(img, contours):

    if len(contours) > 5:
        # Get the four largest contours
        largest_contours = contours[1:5]

        for contour in largest_contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check if all contours run from top left to bottom right
            if not ((x <= 0.05 * img.shape[1] and y + h >= img.shape[0] * 0.95) or
                    (y <= 0.05 * img.shape[0] and x + w >= img.shape[1] * 0.95)):
                return False
                break
        # If all contours fulfill the condition, return True
        return True

    else:
        return False


def satellite_check(img, contours):

    if len(contours) > 7:
        # Get the six largest contours
        largest_contours = contours[1:7]

        for contour in largest_contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check if all contours run from top left to bottom right
            if not ((x <= 0.05 * img.shape[1] and y + h >= img.shape[0] * 0.95) or
                    (y <= 0.05 * img.shape[0] and x + w >= img.shape[1] * 0.95)):
                return False
                break
        # If all contours fulfill the condition, return True
        return True

    else:
        return False


def detect_transposable_elements(input_dir, output_dir):

    is_valid_input_path(input_dir)
    test_output_path(output_dir)

    plot_numbers = {'ltr': [], 'tandem': [], 'satellite': []}

    for filename in os.listdir(input_dir):

        file_number = filename.split('_')[0]

        # Read the processed mask image
        img_color = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_COLOR)
        img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)

        # Apply a threshold to obtain binary image (already done before)
        #_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        # Find contours in the image
        contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Detect LTRs using several conditions and check for tandems and satellites
        if ltr_retrotransposon_check(img, contours):

            if tandem_retro_check(img, contours):

                if satellite_check(img, contours):
                    # print(f'{filename}is a satellite')
                    plot_numbers['satellite'].append(file_number)

                else:
                    # print(f'{filename} is a tandem')
                    plot_numbers['tandem'].append(file_number)
            else:
                # print(f'{filename} is a LTR')
                plot_numbers['ltr'].append(file_number)
        # else:


        # TODO check for color here and also save it in plot numbers

    return plot_numbers


if __name__ == '__main__':
    # Input and output directories
    input_dir_single_pdf = 'input_single_pdf'
    input_dir_pdf = 'input'
    output_dir_images = 'output/png'
    output_dir_masks = 'output/masks'
    output_dir_cleaned = 'output/cleaned-masks'
    output_dir_cropped = 'output/cropped-masks'
    output_dir_contours = 'output/contours'
    output_dir_with_masks = 'output/final'

    output_dir_colored = 'output/colored_plots'

    # Ensure the output directory exists
    test_output_path(output_dir_with_masks)
    ltr_list = []
    tandem_list = []
    satellite_list = []
    colored_sequence_list = []

    # Retrieve the PDF files in the input directory
    single_pdf_file = [f for f in os.listdir(input_dir_single_pdf) if f.endswith('.pdf')]

    # Workaround for splitting a large pdf into single pdfs
    for num_pdf, pdf_filename in enumerate(single_pdf_file):
        pdf_path = os.path.join(input_dir_single_pdf, pdf_filename)

        split_pdf_pages(pdf_path, input_dir_pdf)

    # Retrieve the PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_dir_pdf) if f.endswith('.pdf')]

    # Iterate over the PDF files in the input directory and track the number of files
    for num_pdf, pdf_filename in enumerate(pdf_files):
        # Print the filename as a header
        output_str = f'{pdf_filename} '
        print('\n')
        print(output_str.center(50, '#'))

        # Construct the full path to the PDF file
        pdf_path = os.path.join(input_dir_pdf, pdf_filename)

        # Convert PDF to PNG images
        png_images = convert_pdf_to_png(pdf_path, output_dir_images)

        # Convert the resulting PIL image to a NumPy array
        image_array = np.array(png_images[0])

        # Detect single plots and obtain their bounding boxes
        bounding_boxes_of_plots = detect_single_plots(output_dir_images, output_dir_masks)

        # TODO Integrate if statement for switching between detection of transposable elements and color in plots
        # Detect color within bounding boxes
        # colored_plot_numbers = detect_colored_plots(output_dir_masks, output_dir_colored)
        #
        # Remove noise from the masks of the single plots
        remove_noise(output_dir_masks, output_dir_cleaned)

        crop_images(output_dir_cleaned, output_dir_cropped)

        # Detect transposable elements in the single plots
        detected_elements = detect_transposable_elements(output_dir_cropped, output_dir_contours)

        # Draw bounding boxes of transposable elements on the image
        # and get the number of the sequence in the FASTA file
        for ltr_element in detected_elements['ltr']:
            x_final, y_final, w_final, h_final = bounding_boxes_of_plots[int(ltr_element)]
            cv2.rectangle(image_array, (x_final, y_final), (x_final + w_final, y_final + h_final), (255, 0, 0), 8)

            ltr_number = num_pdf * 20 + int(ltr_element)
            ltr_list.append(ltr_number)

        for tandem_element in detected_elements['tandem']:
            x_final, y_final, w_final, h_final = bounding_boxes_of_plots[int(tandem_element)]
            cv2.rectangle(image_array, (x_final, y_final), (x_final + w_final, y_final + h_final), (0, 255, 0), 8)

            tandem_number = num_pdf * 20 + int(tandem_element)
            tandem_list.append(tandem_number)

        for satellite_element in detected_elements['satellite']:
            x_final, y_final, w_final, h_final = bounding_boxes_of_plots[int(satellite_element)]
            cv2.rectangle(image_array, (x_final, y_final), (x_final + w_final, y_final + h_final), (0, 0, 255), 8)

            satellite_number = num_pdf * 20 + int(satellite_element)
            satellite_list.append(satellite_number)

        # Create the output filename
        output_filename = f'{pdf_filename}_with_all_masks.png'

        # Save the image with all plot masks
        cv2.imwrite(os.path.join(output_dir_with_masks, output_filename), image_array)

        # Write the sequence_numbers to a .txt file
        output_file_path_ltr = os.path.join(output_dir_with_masks, 'ltr.txt')
        output_file_path_tandem = os.path.join(output_dir_with_masks, 'tandem.txt')
        output_file_path_satellite = os.path.join(output_dir_with_masks, 'satellite.txt')

        ltr_list = sorted(ltr_list)
        tandem_list = sorted(tandem_list)
        satellite_list = sorted(satellite_list)

        with open(output_file_path_ltr, 'w') as ltr_file:
            for number in ltr_list:
                ltr_file.write(str(number) + '\n')

        with open(output_file_path_tandem, 'w') as tandem_file:
            for number in tandem_list:
                tandem_file.write(str(number) + '\n')

        with open(output_file_path_satellite, 'w') as satellite_file:
            for number in satellite_list:
                satellite_file.write(str(number) + '\n')

