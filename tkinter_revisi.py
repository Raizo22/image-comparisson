import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import zipfile
import shutil
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil  # To dynamically set thread count based on system capacity

# Global variables
output_dir = None
zip_file_path = None
MAX_THREADS = min(4, psutil.cpu_count(logical=False))  # Set threads dynamically based on CPU cores

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
    search_params = dict(checks=100)
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
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50000, 1e-10)
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

def select_zip_file():
    global zip_file_path
    zip_file_path = filedialog.askopenfilename(title="Select ZIP Containing Images", filetypes=[("ZIP files", "*.zip")])
    if zip_file_path:
        label_zip_path.config(text=f"ZIP File: {os.path.basename(zip_file_path)}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall("temp_images_for_count")
        temp_top_folder = next(os.scandir("temp_images_for_count")).path
        subfolders = [f.path for f in os.scandir(temp_top_folder) if f.is_dir()]
        if len(subfolders) == 2:
            folder_a, _ = subfolders
            total_images = len(find_images_in_directory(folder_a))
            label_total_images.config(text=f"Total images to process: {total_images}")
        shutil.rmtree("temp_images_for_count")

def select_output_directory():
    global output_dir
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if output_dir:
        label_output_dir.config(text=f"Output Folder: {output_dir}")

def find_images_in_directory(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def match_files_regex(comp_image_name, ref_images):
    pattern = re.sub(r'[^a-zA-Z0-9]', '', os.path.splitext(comp_image_name)[0].lower())
    for ref_name, ref_path in ref_images.items():
        ref_pattern = re.sub(r'[^a-zA-Z0-9]', '', os.path.splitext(ref_name)[0].lower())
        if pattern == ref_pattern:
            return ref_path
    return None

def compare_images_thread():
    temp_dir = "temp_images"
    folder_name = folder_name_entry.get()
    output_comparison_folder = os.path.join(output_dir, folder_name)
    if not os.path.exists(output_comparison_folder):
        os.makedirs(output_comparison_folder)

    start_time = time.time()
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    top_level_folder = next(os.scandir(temp_dir)).path
    subfolders = [f.path for f in os.scandir(top_level_folder) if f.is_dir()]
    if len(subfolders) != 2:
        messagebox.showerror("Error", "ZIP file must contain exactly two folders with images.")
        shutil.rmtree(temp_dir)
        return

    folder_a, folder_b = subfolders
    comp_images = find_images_in_directory(folder_a)
    ref_images = find_images_in_directory(folder_b)
    
    if not comp_images or not ref_images:
        messagebox.showerror("Error", "No images found in selected folders.")
        shutil.rmtree(temp_dir)
        return

    ref_images_dict = {os.path.basename(f).lower(): f for f in ref_images}
    total_images = len(comp_images)
    label_total_images.config(text=f"Total images to process: {total_images}")

    def process_image(comp_img_path):
        comp_img_name = os.path.basename(comp_img_path).lower()
        matched_ref_path = match_files_regex(comp_img_name, ref_images_dict)
        if matched_ref_path:
            _, overlay = product_compare(comp_img_path, matched_ref_path)
            if overlay is not None:
                base_name = os.path.basename(comp_img_path)
                cv2.imwrite(os.path.join(output_comparison_folder, f"overlay_{base_name}"), overlay)
            return True
        return False

    successful_comparisons = 0
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_image, img) for img in comp_images]
        for idx, future in enumerate(as_completed(futures), 1):
            success = future.result()
            if success:
                successful_comparisons += 1
            progress = int((successful_comparisons / total_images) * 100)
            progress_bar['value'] = progress
            progress_label.config(text=f"{progress}% completed")
            root.update_idletasks()

    end_time = time.time()
    label_time.config(text=f"Processing Time: {end_time - start_time:.2f} seconds")
    progress_bar['value'] = 0
    progress_label.config(text="")
    shutil.rmtree(temp_dir)
    messagebox.showinfo("Finished", f"Comparison completed and saved in {output_comparison_folder}")

def compare_images():
    if not zip_file_path or not output_dir:
        messagebox.showerror("Error", "Please select a ZIP file and output directory.")
        return
    btn_compare.config(state=tk.DISABLED)
    threading.Thread(target=compare_images_thread).start()
    btn_compare.config(state=tk.NORMAL)

def reset_form():
    global output_dir, zip_file_path
    output_dir = None
    zip_file_path = None
    folder_name_entry.delete(0, tk.END)
    label_zip_path.config(text="No ZIP file selected")
    label_output_dir.config(text="No output folder selected")
    progress_bar['value'] = 0
    progress_label.config(text="")
    label_time.config(text="")
    label_total_images.config(text="Total images to process: 0")

# GUI Setup
root = tk.Tk()
root.title("Image Comparison Tool")
frame = tk.Frame(root)
frame.pack(pady=10)
btn_select_zip = tk.Button(frame, text="Select ZIP", command=select_zip_file)
btn_select_zip.pack(pady=5)
label_zip_path = tk.Label(frame, text="No ZIP file selected")
label_zip_path.pack(pady=5)
btn_select_output = tk.Button(frame, text="Select Output Folder", command=select_output_directory)
btn_select_output.pack(pady=5)
label_output_dir = tk.Label(frame, text="No output folder selected")
label_output_dir.pack(pady=5)
label_total_images = tk.Label(frame, text="Total images to process: 0")
label_total_images.pack(pady=5)
folder_name_label = tk.Label(frame, text="Result Folder Name:")
folder_name_label.pack(pady=5)
folder_name_entry = tk.Entry(frame)
folder_name_entry.pack(pady=5)
btn_compare = tk.Button(frame, text="Compare Images", command=compare_images)
btn_compare.pack(pady=20)
progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
progress_bar.pack(pady=5)
progress_label = tk.Label(frame, text="0% completed")
progress_label.pack(pady=5)
label_time = tk.Label(frame, text="")
label_time.pack(pady=5)
btn_reset = tk.Button(frame, text="Reset", command=reset_form)
btn_reset.pack(pady=10)
root.mainloop()
