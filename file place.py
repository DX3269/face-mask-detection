import os

file_path = 'C:/Users/Ayush Thakur/Desktop/projects/New folder/saved_model.h5'

if os.path.exists(file_path):
    print("File exists")
else:
    print("File not found")
