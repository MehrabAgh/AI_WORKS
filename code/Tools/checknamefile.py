import os

def list_files(directory):
  try:
    files = os.listdir(directory)
    return files
  except FileNotFoundError:
    print(f"Directory '{directory}' not found.")
    return []

# Example usage:
folder_path_1 = "../oliveproject/base/data/test/Ragham_5"
folder_path_2 = "../oliveproject/base/data/train/Ragham_5"

file_names_test = list_files(folder_path_1)
file_names_train = list_files(folder_path_2)

submit = [os.remove(os.path.join("/AI_WORKS/code/OliveProject/base/data/train/Ragham_5",file)) for file in file_names_train for file2 in file_names_test if file == file2 ]
bb = list_files(folder_path_2)
print(set(bb))
print(len(bb))