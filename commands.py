from tkinter import filedialog


def open_file(formats):
    filename = filedialog.askopenfilename(title='Select Model', filetypes=formats)
    print(filename)


