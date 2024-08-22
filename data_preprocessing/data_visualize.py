import os
import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImagePairViewer:
    def __init__(self, master, he_dir, ihc_dir, excel_path):
        self.master = master
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.excel_path = excel_path
        self.image_pairs = sorted(os.listdir(he_dir))
        self.data = pd.read_excel(excel_path)
        self.index = 0

        self.setup_ui()
        self.calculate_statistics()
        self.update_image()

    def setup_ui(self):
        self.master.title("Image Pair Viewer")
        self.master.attributes('-fullscreen', True)

        self.canvas = tk.Canvas(self.master, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.table_frame = tk.Frame(self.master, bg='white')
        self.table_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor='nw')

        self.score_table = ttk.Treeview(self.table_frame, columns=("sr", "name", "value", "view"), show="headings")
        self.score_table.heading("sr", text="Sr No.")
        self.score_table.heading("name", text="Score Name")
        self.score_table.heading("value", text="Score Value")
        self.score_table.heading("view", text="View Distribution")
        self.score_table.pack()

        self.button_frame = tk.Frame(self.master, bg='white')
        self.button_frame.pack(side=tk.RIGHT, padx=10, pady=10, anchor='ne')

        self.blur_button = ttk.Button(self.button_frame, text="Blur", command=self.on_blur)
        self.blur_button.pack(pady=5)

        self.not_aligned_button = ttk.Button(self.button_frame, text="Not Aligned", command=self.on_not_aligned)
        self.not_aligned_button.pack(pady=5)

        self.other_issues_button = ttk.Button(self.button_frame, text="Other Issues", command=self.on_other_issues)
        self.other_issues_button.pack(pady=5)

        self.passed_button = ttk.Button(self.button_frame, text="Passed", command=self.on_passed)
        self.passed_button.pack(pady=5)

        self.navigation_frame = tk.Frame(self.master, bg='white')
        self.navigation_frame.pack(side=tk.BOTTOM, pady=10, anchor='s')

        self.prev_button = ttk.Button(self.navigation_frame, text="<< Previous", command=self.show_prev_image)
        self.prev_button.grid(row=0, column=0, padx=10)

        self.next_button = ttk.Button(self.navigation_frame, text="Next >>", command=self.show_next_image)
        self.next_button.grid(row=0, column=3, padx=10)

        self.close_button = ttk.Button(self.master, text="X", command=self.master.destroy)
        self.close_button.place(relx=1.0, rely=0.0, anchor='ne')

        self.master.bind("<Left>", lambda e: self.show_prev_image())
        self.master.bind("<Right>", lambda e: self.show_next_image())
        self.master.bind("<Escape>", lambda e: self.show_image_view())

    def calculate_statistics(self):
        self.statistics = {}
        for col in self.data.columns[1:]:
            self.statistics[col] = {
                'mean': self.data[col].mean(),
                '25_percentile': self.data[col].quantile(0.25),
                '75_percentile': self.data[col].quantile(0.75)
            }

    def update_image(self):
        he_image_path = os.path.join(self.he_dir, self.image_pairs[self.index])
        ihc_image_path = os.path.join(self.ihc_dir, self.image_pairs[self.index])
        
        he_image = Image.open(he_image_path)
        ihc_image = Image.open(ihc_image_path)

        # Get screen width and height
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # Determine the size of the images to fit side by side
        max_width = (screen_width - 300) // 2  # Leaving some padding
        max_height = screen_height - 200  # Considering space for the info label and buttons

        # Resize images while maintaining aspect ratio
        he_image.thumbnail((max_width, max_height), Image.LANCZOS)
        ihc_image.thumbnail((max_width, max_height), Image.LANCZOS)

        he_image_tk = ImageTk.PhotoImage(he_image)
        ihc_image_tk = ImageTk.PhotoImage(ihc_image)

        self.canvas.delete("all")

        # Calculate positions to center the images
        he_x = (screen_width // 4)
        ihc_x = (3 * screen_width // 4)
        y = (screen_height // 2) - (he_image_tk.height() // 2)

        self.canvas.create_image(he_x, y, image=he_image_tk)
        self.canvas.create_image(ihc_x, y, image=ihc_image_tk)

        # Store references to the images to prevent garbage collection
        self.canvas.he_image_tk = he_image_tk
        self.canvas.ihc_image_tk = ihc_image_tk

        self.update_info()

    def update_info(self):
        file_name = self.image_pairs[self.index]
        scores = self.data[self.data['Filename'] == file_name].iloc[0]

        self.score_table.delete(*self.score_table.get_children())
        for i, col in enumerate(scores.index[1:], start=1):
            self.score_table.insert("", "end", values=(i, col, f"{scores[col]:.4f}", "View"))

        self.score_table.bind("<Button-1>", self.on_score_click)

    def on_score_click(self, event):
        region = self.score_table.identify("region", event.x, event.y)
        if region == "cell":
            column = self.score_table.identify_column(event.x)
            row = self.score_table.identify_row(event.y)
            if column == "#4":  # "View" column
                item = self.score_table.item(row)
                score_name = item["values"][1]
                if score_name in self.statistics:
                    self.show_score_distribution(score_name)

    def show_score_distribution(self, score_name):
        self.canvas.delete("all")

        fig, ax = plt.subplots()
        ax.hist(self.data[score_name], bins=30, alpha=0.7, color='blue')
        ax.axvline(self.statistics[score_name]['mean'], color='r', linestyle='--', label='Mean')
        ax.axvline(self.statistics[score_name]['25_percentile'], color='g', linestyle='--', label='25th Percentile')
        ax.axvline(self.statistics[score_name]['75_percentile'], color='b', linestyle='--', label='75th Percentile')
        ax.legend()
        ax.set_title(f"Distribution of {score_name}")
        ax.set_xlabel(score_name)
        ax.set_ylabel("Frequency")

        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Ensure the plot fills the window
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Close the previous figures to prevent memory leaks
        plt.close(fig)

        self.master.bind("<Escape>", lambda e: self.show_image_view())
        self.current_canvas = canvas

    def show_image_view(self):
        if hasattr(self, 'current_canvas'):
            self.current_canvas.get_tk_widget().pack_forget()
        for widget in self.master.winfo_children():
            widget.destroy()
        self.setup_ui()
        self.update_image()

    def show_next_image(self):
        if self.index < len(self.image_pairs) - 1:
            self.index += 1
            self.update_image()

    def show_prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.update_image()

    def on_blur(self):
        print("Blur button clicked")

    def on_not_aligned(self):
        print("Not Aligned button clicked")

    def on_other_issues(self):
        print("Other Issues button clicked")

    def on_passed(self):
        print("Passed button clicked")


if __name__ == "__main__":
    he_dir = 'datasets/BCI/A/train'  # Update to the correct path
    ihc_dir = 'datasets/BCI/B/train'  # Update to the correct path
    excel_path = 'data_preprocessing/train_image_pair_scores.xlsx'  # Path to the Excel file

    root = tk.Tk()
    viewer = ImagePairViewer(root, he_dir, ihc_dir, excel_path)
    root.mainloop()
