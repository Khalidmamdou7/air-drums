import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import os

class HandMovementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Movement Sound Trigger")

        # Create a style
        style = ttk.Style()
        style.configure("TButton", padding=15, font=('Arial', 16), background='#4CAF50', foreground='white')

        # Create and set the window size (increased to 800x600)
        self.root.geometry("800x600")

        # Start button
        self.start_button = ttk.Button(root, text="Start", command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=20, pady=20)

        # Stop button
        self.stop_button = ttk.Button(root, text="Stop", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=20, pady=20)

        # Quit button
        self.quit_button = ttk.Button(root, text="Quit", command=root.destroy)
        self.quit_button.grid(row=0, column=2, padx=20, pady=20)

        # Process variable to store the subprocess
        self.process = None

        # Load drum image
        drum_image = Image.open("drum.png")  # Replace "drum.png" with the actual path to your drum image
        drum_image = drum_image.resize((400, 400), Image.ANTIALIAS if hasattr(Image, "ANTIALIAS") else print(0))
        self.drum_image = ImageTk.PhotoImage(drum_image)

        # Display drum image
        self.drum_label = tk.Label(root, image=self.drum_image)
        self.drum_label.grid(row=1, column=0, columnspan=3, pady=20)

    def start_detection(self):
        # Check if the process is already running
        if self.process and self.process.poll() is None:
            messagebox.showinfo("Info", "Drums are already running.")
        else:
            # Start drums.py
            self.process = subprocess.Popen(["python", "drums.py"], cwd=os.path.dirname(os.path.abspath(__file__)))
            messagebox.showinfo("Info", "Hand movement detection started.")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

    def stop_detection(self):
        # Check if the process is running
        if self.process and self.process.poll() is None:
            # Terminate the process
            self.process.terminate()
            self.process.wait()
            messagebox.showinfo("Info", "Hand movement detection stopped.")
        else:
            messagebox.showinfo("Info", "No running process to stop.")
            
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = HandMovementApp(root)
    root.mainloop()
