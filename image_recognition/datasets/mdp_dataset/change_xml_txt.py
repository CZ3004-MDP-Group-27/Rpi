import os

if __name__ == '__main__':
    directory = os.fsencode('labels/')
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        #if filename.endswith(".txt"): 
        print(os.path.join(directory.decode(), filename))
        if filename.endswith(".txt"):
            image_id = filename.split('.')[0]
            new_name = image_id + '.txt'
            os.rename(os.path.join(directory.decode(), filename), os.path.join(directory.decode(), new_name))