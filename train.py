from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import os
import numpy as np


class Train:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        title_lbl = Label(self.root, text="Train Data Set", font=("times new roman", 35, "bold"), bg="Black", fg="white")
        title_lbl.place(x=0, y=0, width=1530, height=60)

        img_top = Image.open("college_images/happypepole.jpg")
        img_top = img_top.resize((1530, 325), Image.ANTIALIAS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)

        f_lbl = Label(self.root, image=self.photoimg_top)
        f_lbl.place(x=0, y=55, width=1530, height=325)

        # ============== button==========================
        b1_1 = Button(self.root, text="TRAIN DATA", command=self.train_classifier, cursor="hand2",
                      font=("times new roman", 30, "bold"), bg="black", fg="white")
        b1_1.place(x=0, y=380, width=1530, height=60)

        img_bottom = Image.open("college_images/manypepople.jpg")
        img_bottom = img_bottom.resize((1530, 325), Image.ANTIALIAS)
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)

        f_lbl = Label(self.root, image=self.photoimg_bottom)
        f_lbl.place(x=0, y=440, width=1530, height=325)

    def train_classifier(self):
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        cap = cv2.VideoCapture(0)
        count = 0

        while count < 100:
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_rect = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces_rect:
                    faceROI = gray[y:y + h, x:x + w]
                    cv2.imshow("Capturing Faces", faceROI)

                    # Save the captured face region
                    img_path = os.path.join(data_dir, f"person.{count}.jpg")
                    cv2.imwrite(img_path, faceROI)
                    count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if count == 0:
            messagebox.showinfo("Result", "No faces captured for training!")
        else:
            self.train_model(data_dir)

    def train_model(self, data_dir):
        path = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

        faces = []
        ids = []

        for image in path:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces.append(gray)
            id = int(os.path.split(image)[1].split(".")[1])
            ids.append(id)

        ids = np.array(ids)

        # ===================== Train the classifier And save =============
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")

        messagebox.showinfo("Result", "Training dataset created and saved.")

if __name__ == "__main__":
    root = Tk()
    obj = Train(root)
    root.mainloop()
