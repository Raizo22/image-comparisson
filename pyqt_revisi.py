import cv2
import numpy as np
import os
import zipfile
import shutil
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QProgressBar, QFileDialog, QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Global variable
MAX_THREADS = min(4, psutil.cpu_count(logical=False))

# Enhanced alignment function with improved accuracy
def align_images(image, template, maxFeatures=7000, keepPercent=0.2, reprojThresh=5.0, scale_percent=100):
    if scale_percent < 100:
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        template = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imageGray = clahe.apply(imageGray)
    templateGray = clahe.apply(templateGray)

    sift = cv2.SIFT_create(maxFeatures)
    kpsA, descsA = sift.detectAndCompute(imageGray, None)
    kpsB, descsB = sift.detectAndCompute(templateGray, None)
    if kpsA is None or kpsB is None:
        print("Feature detection failed.")
        return image

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=100)  # Disamakan dengan Tkinter
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(descsA, descsB, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]

    if len(good_matches) > 10:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good_matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good_matches])
        
        H, _ = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=reprojThresh)
        if H is not None:
            (h, w) = template.shape[:2]
            aligned = cv2.warpPerspective(image, H, (w, h))
            templateGray = templateGray.astype(np.float32)
            aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            warp_matrix = H.copy()
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50000, 1e-10)  # Disamakan dengan Tkinter
            try:
                _, refined_warp = cv2.findTransformECC(templateGray, aligned_gray, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria)
                refined_aligned = cv2.warpPerspective(image, refined_warp, (w, h))
                return refined_aligned
            except cv2.error as e:
                print(f"ECC refinement failed: {e}")
                return aligned
        else:
            print("Homography computation failed.")
            return image
    else:
        print("Not enough good matches.")
        return image

def image_difference(image1, image2):
    aligned = align_images(image1, image2)
    diff_image = cv2.absdiff(image2, aligned)
    gray_diff = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
    _, binary_diff = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilated_diff = cv2.dilate(binary_diff, kernel, iterations=1)
    return dilated_diff, aligned

def redblue_image(input_image1, input_image2):
    img_1 = cv2.cvtColor(input_image1, cv2.COLOR_BGR2GRAY)
    img_2 = cv2.cvtColor(input_image2, cv2.COLOR_BGR2GRAY)
    img_merge_1 = cv2.merge((img_1, img_1, img_1))
    img_merge_2 = cv2.merge((img_2, img_2, img_2))
    red = np.zeros((1, 1, 3), np.uint8)
    red[:] = (0, 0, 255)
    blue = np.zeros((1, 1, 3), np.uint8)
    blue[:] = (255, 0, 0)
    white = np.zeros((1, 1, 3), np.uint8)
    white[:] = (255, 255, 255)
    lut_1 = np.concatenate((blue, white), axis=0)
    lut_2 = np.concatenate((red, white), axis=0)
    lut_1 = cv2.resize(lut_1, (1, 256), interpolation=cv2.INTER_CUBIC)
    lut_2 = cv2.resize(lut_2, (1, 256), interpolation=cv2.INTER_CUBIC)
    result_template = cv2.LUT(img_merge_1, lut_1)
    result_compared = cv2.LUT(img_merge_2, lut_2)
    return result_template, result_compared

def product_compare(image_path_1, image_path_2):
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)
    diff_image, aligned = image_difference(image1, image2)
    red_blue_template, red_blue_aligned = redblue_image(image2, aligned)
    overlay = cv2.addWeighted(red_blue_template, 0.5, red_blue_aligned, 0.5, 0, dtype=cv2.CV_32F)
    return diff_image, overlay

class ImageComparisonThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)

    def __init__(self, zip_path, output_dir, folder_name):
        super().__init__()
        self.zip_path = zip_path
        self.output_dir = output_dir
        self.folder_name = folder_name

    def run(self):
        start_time = time.time()
        temp_dir = "temp_images"
        output_comparison_folder = os.path.join(self.output_dir, self.folder_name)
        os.makedirs(output_comparison_folder, exist_ok=True)

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        top_level_folder = next(os.scandir(temp_dir)).path
        subfolders = [f.path for f in os.scandir(top_level_folder) if f.is_dir()]
        if len(subfolders) != 2:
            self.finished_signal.emit("ZIP file must contain exactly two folders with images.")
            shutil.rmtree(temp_dir)
            return

        folder_a, folder_b = subfolders
        comp_images = self.find_images_in_directory(folder_a)
        ref_images = self.find_images_in_directory(folder_b)
        if not comp_images or not ref_images:
            self.finished_signal.emit("No images found in selected folders.")
            shutil.rmtree(temp_dir)
            return

        ref_images_dict = {os.path.basename(f).lower(): f for f in ref_images}
        total_images = len(comp_images)
        processed_images_count = 0

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(self.process_image, img, ref_images_dict, output_comparison_folder) for img in comp_images]
            for future in as_completed(futures):
                future.result()
                processed_images_count += 1
                progress = int((processed_images_count / total_images) * 100)
                self.progress_signal.emit(progress)
        
        shutil.rmtree(temp_dir)
        end_time = time.time()
        total_time = end_time - start_time
        self.finished_signal.emit(f"Comparison completed in {total_time:.2f} seconds for {total_images} images. Results saved in {output_comparison_folder}")

    def find_images_in_directory(self, directory):
        image_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        return image_files

    def process_image(self, comp_img_path, ref_images_dict, output_comparison_folder):
        comp_img_name = os.path.basename(comp_img_path).lower()
        matched_ref_path = self.match_files_regex(comp_img_name, ref_images_dict)
        if matched_ref_path:
            _, overlay = product_compare(comp_img_path, matched_ref_path)
            if overlay is not None:
                base_name = os.path.basename(comp_img_path)
                cv2.imwrite(os.path.join(output_comparison_folder, f"overlay_{base_name}"), overlay)

    def match_files_regex(self, comp_image_name, ref_images):
        pattern = re.sub(r'[^a-zA-Z0-9]', '', os.path.splitext(comp_image_name)[0].lower())
        for ref_name, ref_path in ref_images.items():
            ref_pattern = re.sub(r'[^a-zA-Z0-9]', '', os.path.splitext(ref_name)[0].lower())
            if pattern == ref_pattern:
                return ref_path
        return None

class ImageComparisonApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Comparison Tool")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.zip_label = QLabel("No ZIP file selected")
        self.output_label = QLabel("No output folder selected")
        self.folder_name = QLineEdit()
        self.folder_name.setPlaceholderText("Result Folder Name")

        self.btn_select_zip = QPushButton("Select ZIP")
        self.btn_select_zip.clicked.connect(self.select_zip_file)
        
        self.btn_select_output = QPushButton("Select Output Folder")
        self.btn_select_output.clicked.connect(self.select_output_directory)

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")

        self.compare_button = QPushButton("Compare Images")
        self.compare_button.clicked.connect(self.start_comparison)
        
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_form)

        self.layout.addWidget(self.zip_label)
        self.layout.addWidget(self.btn_select_zip)
        self.layout.addWidget(self.output_label)
        self.layout.addWidget(self.btn_select_output)
        self.layout.addWidget(QLabel("Result Folder Name:"))
        self.layout.addWidget(self.folder_name)
        self.layout.addWidget(self.compare_button)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.btn_reset)

        self.zip_file_path = ""
        self.output_dir_path = ""

    def reset_form(self):
        self.zip_file_path = ""
        self.output_dir_path = ""
        self.folder_name.clear()
        self.zip_label.setText("No ZIP file selected")
        self.output_label.setText("No output folder selected")
        self.progress_bar.setValue(0)
        self.status_label.setText("")

    def select_zip_file(self):
        self.zip_file_path, _ = QFileDialog.getOpenFileName(self, "Select ZIP Containing Images", "", "ZIP files (*.zip)")
        if self.zip_file_path:
            self.zip_label.setText(f"ZIP File: {os.path.basename(self.zip_file_path)}")

    def select_output_directory(self):
        self.output_dir_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if self.output_dir_path:
            self.output_label.setText(f"Output Folder: {self.output_dir_path}")

    def start_comparison(self):
        if not self.zip_file_path or not self.output_dir_path:
            QMessageBox.warning(self, "Error", "Please select a ZIP file and output directory.")
            return

        folder_name = self.folder_name.text()
        self.thread = ImageComparisonThread(self.zip_file_path, self.output_dir_path, folder_name)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished_signal.connect(self.show_finished_message)
        self.thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.status_label.setText(f"{value}% completed")

    def show_finished_message(self, message):
        self.status_label.setText(f"Process completed: {message}")
        self.progress_bar.setValue(0)
        QMessageBox.information(self, "Finished", message)

if __name__ == "__main__":
    app = QApplication([])
    window = ImageComparisonApp()
    window.show()
    app.exec_()
    
