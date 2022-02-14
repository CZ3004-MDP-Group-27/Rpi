import os

if __name__ == '__main__':
    source_files = os.fsencode('images/')
    destination_folder = './labels/'

    for file in os.listdir(source_files):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):

            os.rename(os.path.join(source_files.decode(), filename), os.path.join(destination_folder, filename))