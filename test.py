import os
import shutil

directory='/Users/nach/Downloads/cleaned_zerr_dataset'
public_directory='/Users/nach/Downloads/zerr_dataset_cleaned/public'
private_directory='/Users/nach/Downloads/zerr_dataset_cleaned/private'
n=0
public=[]
private=[]
for inner_directory in os.listdir(directory):
    if not inner_directory.startswith('.'):
        print('Profile Name: '+inner_directory)
        for class_name in os.listdir(directory+'/'+inner_directory):
            if not class_name.startswith('.'):
                print('Class Name: '+class_name)
                for image_name in os.listdir(directory + '/' + inner_directory + '/' + class_name):
                    if not image_name.startswith('.'):
                        print(image_name)
                        if class_name=='private':
                            # shutil.move(directory + '/' + inner_directory + '/' + class_name+'/'+image_name, private_directory)
                            private.append(directory + '/' + inner_directory + '/' + class_name+'/'+image_name)
                        if class_name=='public':
                            public.append(directory + '/' + inner_directory + '/' + class_name + '/' + image_name)
                            # shutil.move(directory + '/' + inner_directory + '/' + class_name + '/' + image_name, public_directory)
                        n+=1
print(n)
