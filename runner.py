import ttkbootstrap as tb
from tkinter import *
from commands import open_file

root = tb.Window(themename='solar')
root.title('Traffic')

# Add widgets
options = ['Train', 'Predict']

option = StringVar(root)
option.set('Train')

select_option = tb.OptionMenu(root, option, *options, bootstyle="info")
select_option.pack(pady=15)

model_formats = (('PTH File', '*.pth'), ('PNG File', '*.png'))
select_model = tb.Button(root, bootstyle='info outline', padding=15, text='Select Model', command=lambda:open_file(model_formats))
select_model.pack()

model_name = tb.Label(root, text='Model Name', font=('Helvetica', 10))
model_name.pack(pady=5)

image_formats = (('JPG File', '*.jpg'), ('PNG File', '*.png'))
select_image = tb.Button(root, bootstyle='info outline', padding=15, text='Select Image For Prediction', command=lambda:open_file(image_formats))
select_image.pack()

predict_button = tb.Button(root, bootstyle="info", padding=15, text='Predict Class')
predict_button.pack(pady=15)


root.mainloop()