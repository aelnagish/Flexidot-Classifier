"""
Flexidot Classifier - Detects transposable elements in Flexidot plot images.

This module processes PDF or PNG images of Flexidot plots to identify
LTR retrotransposons, tandem repeats, and satellite sequences.
"""

import cv2
import os
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional
from PIL import Image

# PDF support via PyMuPDF (no Poppler needed!)
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# =============================================================================
# INPUT TYPE DETECTION
# =============================================================================

class InputType(Enum):
    """Types of input configurations."""
    EMPTY = auto()
    SINGLE_MULTIPAGE_PDF = auto()
    MULTIPLE_PDFS = auto()
    MULTIPLE_PNGS = auto()
    MIXED = auto()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Central configuration for all processing parameters."""

    # Paths
    input_dir: str = "input"
    output_dir_images: str = "output/png"
    output_dir_masks: str = "output/masks"
    output_dir_cleaned: str = "output/cleaned-masks"
    output_dir_cropped: str = "output/cropped-masks"
    output_dir_colored: str = "output/colored_plots"
    output_dir_final: str = "output/final"

    # PDF rendering settings
    pdf_dpi: int = 200

    # Plot size output (pixels)
    standard_mask_size: tuple = (750, 750)

    # Detection thresholds
    min_plot_width: int = 100
    min_plot_height: int = 100
    contour_tolerance: int = 10

    # Crop margins (pixels)
    crop_bottom_left: int = 37
    crop_top_right: int = 15

    # Element detection thresholds
    edge_threshold: float = 0.05
    ltr_min_box_area: int = 250
    ltr_max_contour_width: int = 15

    # Contour count requirements
    min_contours_ltr: int = 3
    min_contours_tandem: int = 5
    min_contours_satellite: int = 7

    # HSV color range for green detection
    green_hsv_lower: tuple = (30, 75, 20)
    green_hsv_upper: tuple = (90, 255, 255)

    # Annotation colors (BGR format) and thickness
    colors: dict = field(default_factory=lambda: {
        'ltr': (128, 128, 0),
        'tandem': (60, 237, 246),
        'satellite': (44, 86, 202)
    })
    border_thickness_percent: float = 0.005 # 0.5% of image height

    # Plots per page
    plots_per_page: int = 20


# =============================================================================
# FILE UTILITIES
# =============================================================================

class FileManager:
    """Handles file and directory operations."""

    SUPPORTED_EXTENSIONS = ['.pdf', '.png']

    @staticmethod
    def validate_input(filepath: str) -> bool:
        """Check if input path exists."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input path does not exist: {filepath}")
        return True

    @staticmethod
    def ensure_input_dir_exists(filepath: str) -> bool:
        """
        Ensure input directory exists. Create it if missing.

        Returns:
            True if directory exists and contains supported files, False otherwise.
        """
        path = Path(filepath)

        if not path.exists():
            print(f"[DIR] Creating input directory: {filepath}")
            path.mkdir(parents=True, exist_ok=True)

            # Create a helpful README file
            readme_path = path / "README.txt"
            readme_content = """
================================================================================
                        FLEXIDOT CLASSIFIER - INPUT FOLDER
================================================================================

Place your Flexidot plot files in this folder!

SUPPORTED INPUT MODES:
  
  1. SINGLE MULTI-PAGE PDF
     -> Place ONE PDF file containing multiple Flexidot plot pages
     -> The program will automatically split and process all pages
  
  2. MULTIPLE PDF FILES  
     -> Place multiple single-page PDF files
     -> Each PDF will be processed separately
  
  3. MULTIPLE PNG FILES
     -> Place multiple PNG images (one plot page per image)
     -> Images will be processed directly without conversion

SUPPORTED FORMATS:
  * PDF files (.pdf)
  * PNG files (.png)

NOTE: Do not mix PDFs and PNGs in the same folder.

USAGE:
  1. Copy your Flexidot plot files into this folder
  2. Run the classifier again
  3. Check the 'output/final' folder for results

================================================================================
"""
            readme_path.write_text(readme_content.strip())
            return False

        # Check for supported files
        files = FileManager.get_files_by_extension(filepath, FileManager.SUPPORTED_EXTENSIONS)
        return len(files) > 0

    @staticmethod
    def prepare_output_dir(filepath: str, clean: bool = True) -> None:
        """Create output directory, optionally clearing existing files."""
        path = Path(filepath)

        if not path.exists():
            print(f"Creating directory: {filepath}")
            path.mkdir(parents=True, exist_ok=True)
        elif clean and any(path.iterdir()):
            print(f"Cleaning directory: {filepath}")
            for file in path.iterdir():
                if file.is_file():
                    file.unlink()

    @staticmethod
    def get_files_by_extension(directory: str, extensions: list[str]) -> list[str]:
        """Get all files matching given extensions."""
        extensions = [ext.lower() for ext in extensions]
        return sorted([
            f for f in os.listdir(directory)
            if Path(f).suffix.lower() in extensions
        ])

    @staticmethod
    def detect_input_type(directory: str) -> tuple[InputType, dict]:
        """
        Analyze input directory and determine what type of input is present.

        Returns:
            Tuple of (InputType, info_dict with file counts and lists)
        """
        pdf_files = FileManager.get_files_by_extension(directory, ['.pdf'])
        png_files = FileManager.get_files_by_extension(directory, ['.png'])

        info = {
            'pdf_files': pdf_files,
            'png_files': png_files,
            'pdf_count': len(pdf_files),
            'png_count': len(png_files)
        }

        if info['pdf_count'] == 0 and info['png_count'] == 0:
            return InputType.EMPTY, info

        elif info['pdf_count'] > 0 and info['png_count'] > 0:
            return InputType.MIXED, info

        elif info['pdf_count'] == 1:
            return InputType.SINGLE_MULTIPAGE_PDF, info

        elif info['pdf_count'] > 1:
            return InputType.MULTIPLE_PDFS, info

        else:
            return InputType.MULTIPLE_PNGS, info


# =============================================================================
# IMAGE LOADING (PDF & PNG SUPPORT)
# =============================================================================

class ImageLoader:
    """Handles loading images from various formats using PyMuPDF."""

    def __init__(self, config: Config):
        self.config = config

    def load_from_pdf(self, pdf_path: str) -> list[Image.Image]:
        """Convert PDF pages to PIL Images using PyMuPDF."""
        if not PDF_SUPPORT:
            raise ImportError(
                "PDF support requires PyMuPDF. Install with: pip install pymupdf"
            )

        FileManager.validate_input(pdf_path)

        images = []
        pdf_document = fitz.open(pdf_path)

        zoom = self.config.pdf_dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            images.append(img)

        pdf_document.close()
        return images

    def load_from_png(self, image_path: str) -> list[Image.Image]:
        """Load PNG file as PIL Image."""
        FileManager.validate_input(image_path)
        return [Image.open(image_path).convert("RGB")]

    def load(self, filepath: str) -> list[Image.Image]:
        """Universal loader - detects format and loads appropriately."""
        ext = Path(filepath).suffix.lower()

        if ext == '.pdf':
            return self.load_from_pdf(filepath)
        elif ext == '.png':
            return self.load_from_png(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def save_images(self, images: list[Image.Image], output_dir: str,
                    base_name: str, format: str = 'PNG') -> list[str]:
        """Save list of PIL Images to output directory."""
        FileManager.prepare_output_dir(output_dir)

        saved_paths = []
        for i, image in enumerate(images):
            filename = f"page{i}_{base_name}.png"
            output_path = os.path.join(output_dir, filename)
            image.save(output_path, format)
            saved_paths.append(output_path)

        return saved_paths

    def get_pdf_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF file."""
        if not PDF_SUPPORT:
            return 0

        pdf_document = fitz.open(pdf_path)
        count = len(pdf_document)
        pdf_document.close()
        return count


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

class ImageProcessor:
    """Core image processing operations."""

    def __init__(self, config: Config):
        self.config = config

    def detect_plots(self, input_dir: str, output_dir: str) -> list[tuple[int, int, int, int]]:
        """Detect individual plots in an image."""
        FileManager.validate_input(input_dir)
        FileManager.prepare_output_dir(output_dir)

        image_files = FileManager.get_files_by_extension(input_dir, ['.png'])

        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bounding_boxes = self._extract_plot_boxes(contours)
            bounding_boxes = self._sort_boxes_spatially(bounding_boxes)

            for i, (x, y, w, h) in enumerate(bounding_boxes):
                # Extract at original resolution
                plot_mask = img[y:y + h, x:x + w]

                # Normalize to 750x750 for consistent downstream processing
                plot_mask = cv2.resize(
                    plot_mask,
                    self.config.standard_mask_size,
                    interpolation=cv2.INTER_AREA if plot_mask.shape[0] > 750 else cv2.INTER_CUBIC
                )

                cv2.imwrite(os.path.join(output_dir, f'{i}.png'), plot_mask)

            return bounding_boxes

        return []

    def _extract_plot_boxes(self, contours) -> list[tuple[int, int, int, int]]:
        """Extract bounding boxes from contours that meet size requirements."""
        boxes = []

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                if w >= self.config.min_plot_width and h >= self.config.min_plot_height:
                    boxes.append((x, y, w, h))

        return boxes

    def _sort_boxes_spatially(self, boxes: list[tuple]) -> list[tuple]:
        """Sort boxes by position (top-to-bottom, left-to-right)."""
        tolerance = self.config.contour_tolerance
        return sorted(boxes, key=lambda rect: (rect[1] // tolerance, rect[0] // tolerance))

    def remove_noise(self, input_dir: str, output_dir: str) -> None:
        """Apply morphological operations to clean up mask images."""
        FileManager.validate_input(input_dir)
        FileManager.prepare_output_dir(output_dir)

        kernel = np.eye(5, dtype=np.uint8)

        for filename in os.listdir(input_dir):
            img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            dilated = cv2.morphologyEx(opened, cv2.MORPH_DILATE, kernel, iterations=2)

            new_filename = Path(filename).stem + '_clean.png'
            cv2.imwrite(os.path.join(output_dir, new_filename), dilated)

    def crop_images(self, input_dir: str, output_dir: str) -> None:
        """Crop margins from images to remove axes/labels."""
        FileManager.validate_input(input_dir)
        FileManager.prepare_output_dir(output_dir)

        bl = self.config.crop_bottom_left
        tr = self.config.crop_top_right

        for filename in os.listdir(input_dir):
            img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)
            height, width = img.shape[:2]

            cropped = img[tr:height-bl, bl:width-tr]

            new_filename = Path(filename).stem + '_cropped.png'
            cv2.imwrite(os.path.join(output_dir, new_filename), cropped)

    def detect_colored_plots(self, input_dir: str, output_dir: str) -> list[str]:
        """Detect plots containing green color annotations."""
        FileManager.validate_input(input_dir)
        FileManager.prepare_output_dir(output_dir)

        image_files = FileManager.get_files_by_extension(input_dir, ['.png'])
        colored_plots = []

        for image_file in image_files:
            img = cv2.imread(os.path.join(input_dir, image_file), cv2.IMREAD_COLOR)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            green_mask = cv2.inRange(
                hsv,
                self.config.green_hsv_lower,
                self.config.green_hsv_upper
            )

            if np.any(green_mask > 0):
                file_number = Path(image_file).stem.split('.')[0]
                print(f"Green color found in plot {file_number}")
                colored_plots.append(file_number)

                output_path = os.path.join(output_dir, f"green_mask_{image_file}")
                cv2.imwrite(output_path, green_mask)

        return colored_plots


# =============================================================================
# TRANSPOSABLE ELEMENT DETECTION
# =============================================================================

class ElementDetector:
    """Detects transposable elements based on contour patterns."""

    def __init__(self, config: Config):
        self.config = config

    def _is_diagonal_contour(self, img: np.ndarray, contour) -> bool:
        """Check if contour runs diagonally from corner to corner."""
        x, y, w, h = cv2.boundingRect(contour)
        img_h, img_w = img.shape[:2]
        edge = self.config.edge_threshold

        at_bottom_left = (x <= edge * img_w) and (y + h >= img_h * (1 - edge))
        at_top_right = (y <= edge * img_h) and (x + w >= img_w * (1 - edge))

        return at_bottom_left or at_top_right

    def _check_ltr(self, img: np.ndarray, contours: list) -> bool:
        """Check for LTR retrotransposon pattern."""
        if len(contours) <= self.config.min_contours_ltr:
            return False

        largest_contours = contours[1:3]

        ltr_left = False
        ltr_right = False
        has_area = False

        img_h, img_w = img.shape[:2]
        edge = self.config.edge_threshold

        for contour in largest_contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_area = cv2.contourArea(contour)
            contour_width = contour_area / h if h > 0 else 0

            if contour_width > self.config.ltr_max_contour_width:
                return False

            box_area = w * h

            if x <= edge * img_w and y + h >= img_h * (1 - edge):
                ltr_left = True
            if y <= edge * img_h and x + w >= img_w * (1 - edge):
                ltr_right = True
                if box_area > self.config.ltr_min_box_area:
                    has_area = True

        return ltr_left and ltr_right and has_area

    def _check_tandem(self, img: np.ndarray, contours: list) -> bool:
        """Check for tandem repeat pattern."""
        if len(contours) <= self.config.min_contours_tandem:
            return False

        largest_contours = contours[1:5]
        return all(self._is_diagonal_contour(img, c) for c in largest_contours)

    def _check_satellite(self, img: np.ndarray, contours: list) -> bool:
        """Check for satellite pattern."""
        if len(contours) <= self.config.min_contours_satellite:
            return False

        largest_contours = contours[1:7]
        return all(self._is_diagonal_contour(img, c) for c in largest_contours)

    def detect(self, input_dir: str) -> dict[str, list[str]]:
        """Detect transposable elements in processed plot images."""
        FileManager.validate_input(input_dir)

        results = {'ltr': [], 'tandem': [], 'satellite': []}

        for filename in os.listdir(input_dir):
            file_number = filename.split('_')[0]
            img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)

            contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            if self._check_ltr(img, contours):
                if self._check_tandem(img, contours):
                    if self._check_satellite(img, contours):
                        results['satellite'].append(file_number)
                    else:
                        results['tandem'].append(file_number)
                else:
                    results['ltr'].append(file_number)

        return results


# =============================================================================
# RESULT VISUALIZATION & OUTPUT
# =============================================================================

class ResultWriter:
    """Handles output generation and visualization."""

    def __init__(self, config: Config):
        self.config = config

    def draw_annotations(self, image: np.ndarray,
                         detected_elements: dict[str, list[str]],
                         bounding_boxes: list[tuple]) -> np.ndarray:
        """Draw colored rectangles around detected elements."""
        annotated = image.copy()

        # Calculate thickness relative to image size
        height = image.shape[0]
        thickness = max(2, int(height * self.config.border_thickness_percent))

        for element_type, plot_numbers in detected_elements.items():
            color = self.config.colors.get(element_type, (255, 255, 255))

            for plot_num in plot_numbers:
                idx = int(plot_num)
                if idx < len(bounding_boxes):
                    x, y, w, h = bounding_boxes[idx]
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        return annotated

    def save_results(self, output_dir: str, results: dict[str, list[int]]) -> None:
        """Save detection results to text files."""
        FileManager.prepare_output_dir(output_dir, clean=False)

        for element_type, numbers in results.items():
            filepath = os.path.join(output_dir, f'{element_type}.txt')
            with open(filepath, 'w') as f:
                for num in sorted(numbers):
                    f.write(f'{num}\n')

        print(f"Results saved to {output_dir}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class FlexidotClassifier:
    """Main pipeline for Flexidot plot classification."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.loader = ImageLoader(self.config)
        self.processor = ImageProcessor(self.config)
        self.detector = ElementDetector(self.config)
        self.writer = ResultWriter(self.config)

    def _print_input_summary(self, input_type: InputType, info: dict) -> None:
        """Print a summary of detected input."""
        print(f"\n{'=' * 60}")
        print("[INFO] INPUT ANALYSIS")
        print('=' * 60)

        if input_type == InputType.SINGLE_MULTIPAGE_PDF:
            pdf_file = info['pdf_files'][0]
            page_count = self.loader.get_pdf_page_count(
                os.path.join(self.config.input_dir, pdf_file)
            )
            print(f"  Mode: Single multi-page PDF")
            print(f"  File: {pdf_file}")
            print(f"  Pages: {page_count}")

        elif input_type == InputType.MULTIPLE_PDFS:
            print(f"  Mode: Multiple PDF files")
            print(f"  Files: {info['pdf_count']} PDFs")
            for f in info['pdf_files'][:5]:
                print(f"    * {f}")
            if info['pdf_count'] > 5:
                print(f"    ... and {info['pdf_count'] - 5} more")

        elif input_type == InputType.MULTIPLE_PNGS:
            print(f"  Mode: Multiple PNG files")
            print(f"  Files: {info['png_count']} PNGs")
            for f in info['png_files'][:5]:
                print(f"    * {f}")
            if info['png_count'] > 5:
                print(f"    ... and {info['png_count'] - 5} more")

        print('=' * 60)

    def _handle_empty_input(self) -> None:
        """Display message when no input files are found."""
        print("\n" + "=" * 60)
        print("[WARNING] NO INPUT FILES FOUND")
        print("=" * 60)
        print(f"""
The input directory '{self.config.input_dir}' is empty or was just created.

NEXT STEPS:
  1. Add your Flexidot plot files to:
     -> {os.path.abspath(self.config.input_dir)}
  
  2. Run this program again

SUPPORTED INPUT MODES:
  * Single multi-page PDF (one PDF with multiple pages)
  * Multiple PDF files (one page per PDF)
  * Multiple PNG files (one page per PNG)

NOTE: Do not mix PDFs and PNGs in the same folder.
""")
        print("=" * 60)

    def _handle_mixed_input(self, info: dict) -> None:
        """Display warning when both PDFs and PNGs are found."""
        print("\n" + "=" * 60)
        print("[WARNING] MIXED INPUT DETECTED")
        print("=" * 60)
        print(f"""
Found both PDF and PNG files in '{self.config.input_dir}':
  * {info['pdf_count']} PDF file(s)
  * {info['png_count']} PNG file(s)

Please use only ONE of the following input modes:
  1. Single multi-page PDF
  2. Multiple PDF files  
  3. Multiple PNG files

Remove either the PDFs or PNGs and run again.
""")
        print("=" * 60)

    def process_file(self, filepath: str, file_index: int = 0) -> dict:
        """Process a single input file (PDF or PNG)."""
        print(f"\n{'-' * 40}")
        print(f"Processing: {Path(filepath).name}")
        print('-' * 40)

        images = self.loader.load(filepath)
        base_name = Path(filepath).stem
        self.loader.save_images(images, self.config.output_dir_images, base_name)

        image_array = np.array(images[0])

        bounding_boxes = self.processor.detect_plots(
            self.config.output_dir_images,
            self.config.output_dir_masks
        )

        self.processor.remove_noise(
            self.config.output_dir_masks,
            self.config.output_dir_cleaned
        )

        self.processor.crop_images(
            self.config.output_dir_cleaned,
            self.config.output_dir_cropped
        )

        detected = self.detector.detect(self.config.output_dir_cropped)

        sequence_numbers = {
            element_type: [
                file_index * self.config.plots_per_page + int(num)
                for num in plot_nums
            ]
            for element_type, plot_nums in detected.items()
        }

        annotated = self.writer.draw_annotations(image_array, detected, bounding_boxes)
        output_path = os.path.join(
            self.config.output_dir_final,
            f'{base_name}_annotated.png'
        )
        cv2.imwrite(output_path, annotated)

        return sequence_numbers

    def process_multipage_pdf(self, pdf_path: str) -> dict[str, list[int]]:
        """Process a single multi-page PDF by handling each page."""
        print(f"\n[PDF] Processing multi-page PDF: {Path(pdf_path).name}")

        images = self.loader.load_from_pdf(pdf_path)
        all_results = {'ltr': [], 'tandem': [], 'satellite': []}

        for page_idx, image in enumerate(images):
            print(f"\n{'-' * 40}")
            print(f"Processing page {page_idx + 1} of {len(images)}")
            print('-' * 40)

            base_name = f"{Path(pdf_path).stem}_page{page_idx}"
            self.loader.save_images([image], self.config.output_dir_images, base_name)

            image_array = np.array(image)

            bounding_boxes = self.processor.detect_plots(
                self.config.output_dir_images,
                self.config.output_dir_masks
            )

            self.processor.remove_noise(
                self.config.output_dir_masks,
                self.config.output_dir_cleaned
            )

            self.processor.crop_images(
                self.config.output_dir_cleaned,
                self.config.output_dir_cropped
            )

            detected = self.detector.detect(self.config.output_dir_cropped)

            for element_type, plot_nums in detected.items():
                for num in plot_nums:
                    global_num = page_idx * self.config.plots_per_page + int(num)
                    all_results[element_type].append(global_num)

            annotated = self.writer.draw_annotations(image_array, detected, bounding_boxes)
            output_path = os.path.join(
                self.config.output_dir_final,
                f'{base_name}_annotated.png'
            )
            cv2.imwrite(output_path, annotated)

        return all_results

    def run(self) -> None:
        """Run the full classification pipeline."""

        has_files = FileManager.ensure_input_dir_exists(self.config.input_dir)

        if not has_files:
            self._handle_empty_input()
            return

        input_type, info = FileManager.detect_input_type(self.config.input_dir)

        if input_type == InputType.EMPTY:
            self._handle_empty_input()
            return

        if input_type == InputType.MIXED:
            self._handle_mixed_input(info)
            return

        self._print_input_summary(input_type, info)

        FileManager.prepare_output_dir(self.config.output_dir_final)

        all_results = {'ltr': [], 'tandem': [], 'satellite': []}

        try:
            if input_type == InputType.SINGLE_MULTIPAGE_PDF:
                pdf_path = os.path.join(self.config.input_dir, info['pdf_files'][0])
                all_results = self.process_multipage_pdf(pdf_path)

            elif input_type == InputType.MULTIPLE_PDFS:
                for i, filename in enumerate(info['pdf_files']):
                    filepath = os.path.join(self.config.input_dir, filename)
                    results = self.process_file(filepath, file_index=i)

                    for element_type in all_results:
                        all_results[element_type].extend(results.get(element_type, []))

            elif input_type == InputType.MULTIPLE_PNGS:
                for i, filename in enumerate(info['png_files']):
                    filepath = os.path.join(self.config.input_dir, filename)
                    results = self.process_file(filepath, file_index=i)

                    for element_type in all_results:
                        all_results[element_type].extend(results.get(element_type, []))

        except Exception as e:
            print(f"\n[ERROR] Error during processing: {e}")
            raise

        self.writer.save_results(self.config.output_dir_final, all_results)

        print(f"\n{'=' * 60}")
        print("[SUCCESS] CLASSIFICATION COMPLETE")
        print('=' * 60)
        for element_type, numbers in all_results.items():
            print(f"  {element_type.upper()}: {len(numbers)} detected")
        print(f"\n[OUTPUT] Results saved to: {os.path.abspath(self.config.output_dir_final)}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    config = Config(
        input_dir="input",
        pdf_dpi=200,
    )

    if not PDF_SUPPORT:
        print("[WARNING] PyMuPDF not installed. PDF support disabled.")
        print("          Install with: pip install pymupdf")
        print("          PNG files will still work.\n")

    classifier = FlexidotClassifier(config)
    classifier.run()
