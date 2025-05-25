import tkinter as tk
from tkinter import filedialog, Label, Text, Scrollbar, ttk
from PIL import Image, ImageTk
import os
import numpy as np
from deepface import DeepFace
from emotion_detection import detect_emotion
from anti_spoof_predict import is_real_face

# Directory for stored embeddings
EMBED_DIR = "../output"
EMBED_THRESHOLD = 0.8  # cosine similarity

# Cosine similarity function
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class FaceApp:
    def __init__(self, master):
        self.master = master
        master.title("🎯 A.I. Face Verification System")
        master.configure(bg="#eaeaea")
        master.geometry("850x700")

        # Title
        self.title = Label(master, text="Upload a Face Image", font=("Helvetica", 20, "bold"), bg="#eaeaea")
        self.title.pack(pady=15)

        # Upload button container
        btn_frame = tk.Frame(master, bg="#dfe6f0", bd=2, relief="groove")
        btn_frame.pack(pady=10)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 13, "bold"), padding=10)
        self.upload_btn = ttk.Button(btn_frame, text="📤  Choose Image", command=self.upload_image, style="TButton")
        self.upload_btn.pack()

        # Image display
        self.image_panel = Label(master, bg="#eaeaea")
        self.image_panel.pack(pady=20)

        # Result box
        self.result_box = Text(master, height=12, font=("Courier", 12), wrap="word", bg="#ffffff", fg="#333333")
        self.result_box.pack(padx=20, pady=10, fill="both", expand=True)

        scrollbar = Scrollbar(master, command=self.result_box.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_box.configure(yscrollcommand=scrollbar.set)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
        if not file_path:
            return

        # Show image
        img = Image.open(file_path).resize((250, 250))
        self.photo = ImageTk.PhotoImage(img)
        self.image_panel.config(image=self.photo)

        self.result_box.delete(1.0, tk.END)

        # Anti-spoofing
        real = is_real_face(file_path)
        self.result_box.insert(tk.END, f"[🛡️ Anti-Spoofing]: {'REAL ✅' if real else 'FAKE ❌'}\n", "result")

        # Emotion
        emotion, score = detect_emotion(file_path)
        self.result_box.insert(tk.END, f"[😃 Emotion]: {emotion.upper()} ({score:.2f} confidence)\n", "result")

        # Extract embedding using DeepFace
        try:
            reps = DeepFace.represent(img_path=file_path, model_name="Facenet", enforce_detection=False)
            current_embedding = np.array(reps[0]["embedding"])

            # Load existing embeddings
            best_score = -1
            best_match = None

            for fname in os.listdir(EMBED_DIR):
                if fname.endswith(".csv"):
                    saved_path = os.path.join(EMBED_DIR, fname)
                    saved_embedding = np.loadtxt(saved_path, delimiter=",")
                    sim = cosine_similarity(current_embedding, saved_embedding)

                    if sim > best_score:
                        best_score = sim
                        best_match = fname

            if best_score >= EMBED_THRESHOLD:
                self.result_box.insert(tk.END, f"[🧠 Verification]: MATCH ✅ with {best_match} (score={best_score:.2f})\n", "result")
                self.result_box.insert(tk.END, "👉 Face recognized. Welcome back!\n", "result")
            else:
                # Save new embedding
                new_name = os.path.basename(file_path)
                save_path = os.path.join(EMBED_DIR, f"new_face_{new_name}.csv")
                np.savetxt(save_path, current_embedding, delimiter=",")
                self.result_box.insert(tk.END, "[🧠 Verification]: NO MATCH ❌ – Registered as new identity\n", "result")
                self.result_box.insert(tk.END, "👉 New face registered in the system.\n", "result")

        except Exception as e:
            self.result_box.insert(tk.END, f"[🧠 Verification]: FAILED ❌ – {str(e)}\n", "result")

        self.result_box.tag_config("result", foreground="#007700", font=("Courier", 12, "bold"))

# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
