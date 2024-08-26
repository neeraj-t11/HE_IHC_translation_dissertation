import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

# Function to read and keep image in its original color
def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to load the Excel file with statuses
def load_excel(excel_path):
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:
        df = pd.DataFrame(columns=["Filename", "Status"])
    return df

# Function to save the updated DataFrame to Excel
def save_to_excel(df, excel_path):
    df.to_excel(excel_path, index=False)

# Class for Image Pair Reviewer
class ImagePairReviewer:
    def __init__(self, root, he_files, ihc_files, he_dir, ihc_dir, output_excel):
        self.root = root
        self.he_files = he_files
        self.ihc_files = ihc_files
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.output_excel = output_excel
        self.current_index = 0
        
        # Load existing statuses from the Excel file
        self.status_df = load_excel(self.output_excel)
        
        # Initialize filtered lists
        self.filtered_he_files = self.he_files
        self.filtered_ihc_files = self.ihc_files

        # Set the window to full-screen mode
        self.root.attributes("-fullscreen", True)
        
        # Bind the Escape key to exit full-screen mode
        self.root.bind("<Escape>", self.exit_fullscreen)

        # Main frame to hold images and buttons
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Image frames
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # Filename display frame
        self.filename_frame = ttk.Frame(self.main_frame)
        self.filename_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Label to display filenames and status
        self.filename_label = ttk.Label(self.filename_frame, text="", anchor="center")
        self.filename_label.pack(side=tk.TOP, fill=tk.X)

        # Buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        self.yes_button = ttk.Button(self.button_frame, text="Yes", command=self.yes_action)
        self.yes_button.pack(side=tk.LEFT, expand=True)
        
        self.no_button = ttk.Button(self.button_frame, text="No", command=self.no_action)
        self.no_button.pack(side=tk.LEFT, expand=True)

        self.maybe_button = ttk.Button(self.button_frame, text="Maybe", command=self.maybe_action)
        self.maybe_button.pack(side=tk.LEFT, expand=True)
        
        self.none_button = ttk.Button(self.button_frame, text="None", command=self.none_action)
        self.none_button.pack(side=tk.LEFT, expand=True)

        self.backward_button = ttk.Button(self.button_frame, text="Backward", command=self.backward_action)
        self.backward_button.pack(side=tk.LEFT, expand=True)
        
        self.forward_button = ttk.Button(self.button_frame, text="Forward", command=self.forward_action)
        self.forward_button.pack(side=tk.LEFT, expand=True)

        # Filter buttons
        self.filter_frame = ttk.Frame(self.main_frame)
        self.filter_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        self.show_yes_button = ttk.Button(self.filter_frame, text="Show Yes", command=lambda: self.apply_filter("Yes"))
        self.show_yes_button.pack(side=tk.LEFT, expand=True)

        self.show_no_button = ttk.Button(self.filter_frame, text="Show No", command=lambda: self.apply_filter("No"))
        self.show_no_button.pack(side=tk.LEFT, expand=True)

        self.show_maybe_button = ttk.Button(self.filter_frame, text="Show Maybe", command=lambda: self.apply_filter("Maybe"))
        self.show_maybe_button.pack(side=tk.LEFT, expand=True)

        self.show_none_button = ttk.Button(self.filter_frame, text="Show None", command=lambda: self.apply_filter("None"))
        self.show_none_button.pack(side=tk.LEFT, expand=True)

        # Initialize image labels
        self.image_label_1 = ttk.Label(self.image_frame)
        self.image_label_1.pack(side=tk.LEFT, expand=True, padx=10, pady=10)

        self.image_label_2 = ttk.Label(self.image_frame)
        self.image_label_2.pack(side=tk.RIGHT, expand=True, padx=10, pady=10)

        # Display the first image pair
        self.display_images()

    def display_images(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate available height after accounting for button and filename area
        button_area_height = self.button_frame.winfo_reqheight() + self.filename_frame.winfo_reqheight() + 20  # Padding for both areas
        max_image_height = screen_height - button_area_height
        max_image_width = screen_width // 2

        he_image = read_image(os.path.join(self.he_dir, self.filtered_he_files[self.current_index]))
        ihc_image = read_image(os.path.join(self.ihc_dir, self.filtered_ihc_files[self.current_index]))

        he_image_pil = Image.fromarray(he_image)
        ihc_image_pil = Image.fromarray(ihc_image)

        he_image_pil.thumbnail((max_image_width, max_image_height), Image.LANCZOS)
        ihc_image_pil.thumbnail((max_image_width, max_image_height), Image.LANCZOS)

        he_image_tk = ImageTk.PhotoImage(he_image_pil)
        ihc_image_tk = ImageTk.PhotoImage(ihc_image_pil)

        self.image_label_1.config(image=he_image_tk)
        self.image_label_1.image = he_image_tk

        self.image_label_2.config(image=ihc_image_tk)
        self.image_label_2.image = ihc_image_tk

        # Get the current status from the DataFrame
        current_filename = self.filtered_he_files[self.current_index]
        status_row = self.status_df[self.status_df['Filename'] == current_filename]
        current_status = status_row['Status'].values[0] if not status_row.empty else "None"

        # Update the filename label with the current pair's filenames and status
        self.filename_label.config(text=f"{current_filename} (Status: {current_status})")

        self.root.title(f"Image Pair {self.current_index + 1}/{len(self.filtered_he_files)}: {current_filename}")

    def update_status(self, status):
        current_filename = self.filtered_he_files[self.current_index]
        if current_filename in self.status_df['Filename'].values:
            self.status_df.loc[self.status_df['Filename'] == current_filename, 'Status'] = status
        else:
            new_row = pd.DataFrame({"Filename": [current_filename], "Status": [status]})
            self.status_df = pd.concat([self.status_df, new_row], ignore_index=True)
        save_to_excel(self.status_df, self.output_excel)

    def yes_action(self):
        self.update_status("Yes")
        self.forward_action()

    def no_action(self):
        self.update_status("No")
        self.forward_action()

    def maybe_action(self):
        self.update_status("Maybe")
        self.forward_action()

    def none_action(self):
        # Remove entry from DataFrame if exists
        current_filename = self.filtered_he_files[self.current_index]
        self.status_df = self.status_df[self.status_df['Filename'] != current_filename]
        save_to_excel(self.status_df, self.output_excel)
        self.forward_action()

    def forward_action(self):
        if self.current_index < len(self.filtered_he_files) - 1:
            self.current_index += 1
            self.display_images()
        else:
            messagebox.showinfo("End", "This is the last image pair.")

    def backward_action(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_images()
        else:
            messagebox.showinfo("Start", "This is the first image pair.")

    def apply_filter(self, filter_status):
        if filter_status == "None":
            # Filter for files not in the Excel sheet (i.e., "None" status)
            self.filtered_he_files = [file for file in self.he_files if file not in self.status_df['Filename'].values]
            self.filtered_ihc_files = [file for file in self.ihc_files if file not in self.status_df['Filename'].values]
        else:
            # Filter for files with the specified status
            filtered_filenames = self.status_df[self.status_df['Status'] == filter_status]['Filename'].values
            self.filtered_he_files = [file for file in self.he_files if file in filtered_filenames]
            self.filtered_ihc_files = [file for file in self.ihc_files if file in filtered_filenames]

        # Reset the index and display the first image in the filtered list
        self.current_index = 0
        if self.filtered_he_files:
            self.display_images()
        else:
            messagebox.showinfo("No Images", f"No image pairs with status '{filter_status}' found.")

    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)

# Main function to initiate the GUI and process image pairs
def review_image_pairs(he_dir, ihc_dir, output_excel):
    if not os.path.exists(he_dir):
        raise FileNotFoundError(f"The H&E images directory does not exist: {he_dir}")
    if not os.path.exists(ihc_dir):
        raise FileNotFoundError(f"The IHC images directory does not exist: {ihc_dir}")

    he_files = sorted(os.listdir(he_dir))
    ihc_files = sorted(os.listdir(ihc_dir))

    if len(he_files) != len(ihc_files):
        raise ValueError("The number of H&E images and IHC images must be the same.")

    root = tk.Tk()
    app = ImagePairReviewer(root, he_files, ihc_files, he_dir, ihc_dir, output_excel)
    root.mainloop()

# Directories and output file
# he_dir = 'datasets/BCI/A/train'
# ihc_dir = 'datasets/BCI/B/train'
# output_excel = 'data_preprocessing/train_image_pair_status.xlsx'

# # Start reviewing
# review_image_pairs(he_dir, ihc_dir, output_excel)


he_dir = 'datasets/BCI/A/test'
ihc_dir = 'datasets/BCI/B/test'
output_excel = 'data_preprocessing/test_image_pair_status.xlsx'

# Start reviewing
review_image_pairs(he_dir, ihc_dir, output_excel)
