from helper import prediction

# Make prediction using saved model
top_ps, top_classes = prediction('GTSRB/test/gtsrb/GTSRB/Final_Test/Images/00005.ppm', 'checkpoint.pth')
print(f'Top Probabilities: {top_ps}')
print(f'Top Classes: {top_classes}')