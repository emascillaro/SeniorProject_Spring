import PIL
from PIL import Image,ImageTk
import pytesseract
import cv2
from tkinter import *
import tkinter as tk

LARGE_FONT = ("Verdana", 12)

class Page(tk.Tk):
    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side ="top", fill="both", expand= True)

        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)

        self.frames = {}

        frame = StartPage(container, self)

        self.frames[StartPage] = frame

        frame.grid(row = 0, column = 0, sticky = "nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # Start of Our Code #

        width, height = 800, 600
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        lmain = Label(app)  #fix
        lmain.pack()


        def show_image_frame():
            ret, frame = cap.read()
            if ret:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = PIL.Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
                lmain.after(10, show_image_frame)

        def capture_image():
            videoCaptureObject = cv2.VideoCapture(0)
            ret, frame = videoCaptureObject.read()
            img = cv2.imwrite("Capture_Image.jpg", frame)
            #videoCaptureObject.release()

        image_capture = tk.Button(text="Take Picture", command = capture_image)
        image_capture.pack()

        print("Opening Application")

        show_image_frame()
        root.mainloop()

'''
        switch_page = tk.Button(text = "View the solution", command = lambda: controller.show_frame(SolutionPage))
        switch_page.pack()

class SolutionPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text = "answer", font = LARGE_FONT)
        label.pack(pady = 10, padx = 10)

        return_button = tk.Button(self, text = "Take another Photo", command = lambda: controller.show_frame(StartPage))
        return_button.pack()

'''

app = Page()
app.mainloop()