import os
import shutil

# Source folders
folder1 = r'/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor/'
folder2 = r'/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor 3/'
folder3 = r'/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor part_1/'
folder4 = r'/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor part_2/'
folder5 = r'/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor part_3/'
folder6 = r'/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor part_4/'

# Destination folder
destination_folder = r'/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Copy and rename images from folder1
for i, filename in enumerate(os.listdir(folder1)):
    src_path = os.path.join(folder1, filename)
    dest_path = os.path.join(destination_folder, f'{i + 1}.jpg')
    shutil.copy(src_path, dest_path)

# Copy and rename images from folder2
for i, filename in enumerate(os.listdir(folder2)):
    src_path = os.path.join(folder2, filename)
    dest_path = os.path.join(destination_folder, f'{i + 1 + len(os.listdir(folder1))}.jpg')
    shutil.copy(src_path, dest_path)

# Copy and rename images from folder3
for i, filename in enumerate(os.listdir(folder3)):
    src_path = os.path.join(folder3, filename)
    dest_path = os.path.join(destination_folder, f'{i + 1 + len(os.listdir(folder1)) + len(os.listdir(folder2))}.jpg')
    shutil.copy(src_path, dest_path)
    
# Copy and rename images from folder3
for i, filename in enumerate(os.listdir(folder4)):
    src_path = os.path.join(folder4, filename)
    dest_path = os.path.join(destination_folder, f'{i + 1 + len(os.listdir(folder1)) + len(os.listdir(folder2))}.jpg')
    shutil.copy(src_path, dest_path)

# Copy and rename images from folder3
for i, filename in enumerate(os.listdir(folder5)):
    src_path = os.path.join(folder5, filename)
    dest_path = os.path.join(destination_folder, f'{i + 1 + len(os.listdir(folder1)) + len(os.listdir(folder2))}.jpg')
    shutil.copy(src_path, dest_path)

# Copy and rename images from folder3
for i, filename in enumerate(os.listdir(folder6)):
    src_path = os.path.join(folder6, filename)
    dest_path = os.path.join(destination_folder, f'{i + 1 + len(os.listdir(folder1)) + len(os.listdir(folder2))}.jpg')
    shutil.copy(src_path, dest_path)


print("Images merged and renamed successfully.")
