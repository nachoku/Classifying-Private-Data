import torch
from torchvision.transforms import transforms
#from PIL import Image
#from cnn_main import CNNet
from pathlib import Path
from models import Combined
#import os
from pprint import pprint
from torch.autograd import Variable
from PIL import Image, ImageDraw
import os
import glob
import random





def predictor(image):
    #print(pprint(image_output))
    #print(location)
    model = Combined()
    print('----')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    checkpoint = torch.load(Path('/home/nchocka/security/finalProject/best.pt'))
    model.load_state_dict(checkpoint)
    model.to(device)

    trans = transforms.Compose([
                           transforms.Resize((256,256)),
                           #transforms.RandomHorizontalFlip(30),
                           #transforms.RandomRotation(10),
                           #transforms.CenterCrop(300),
                                transforms.ToTensor(),
    #                       transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
                       ])

    #image = Image.open(Path(location))
    image=trans(image)
    image=Variable(image)
    image=image.unsqueeze(0)
#input = image.view(1, 3, 32,32)

    log_results = model.forward(image) # get the log softmax values
    probs = torch.exp(log_results) # exponentiate
    top_probs, top_labs = probs.topk(2) # get the top 5 results
#print(output)
#prediction = int(torch.max(output.data, 1)[1].numpy())
    print(top_probs)
    return label[top_labs[0][0].item()]


split_x = 4
split_y = 4

def crop(im,i,j,width,height):
    box = (j*width,i*height,(j+1)*width,(i+1)*height)
    return im.crop(box)
def splitter(im):
    path='/home/nchocka/security/finalProject/Test_Image_Classifier/split_images'
    index = 0
    hide = random.randint(1,split_x*split_y+1)
    print(hide)
    #imgloc = 'C:/Users/saral/Desktop/Network Security/ImageSplitter/city.jpg'
    #im = Image.open(imgloc)
    dr = im
    imwidth,imheight = im.size
    height = round(imheight/split_x)
    width = round(imwidth/split_y)
    print("height: ",height," width: ",width)
    img = Image.new('RGB',(im.size),color = (0, 0, 0))
    draw = ImageDraw.Draw(dr)
    print('Fail 2')
    for i in range(split_x):
        for j in range(split_y):
            piece = crop(im,i,j,width,height)
            prediction=predictor(piece)
            print (prediction)
            index = index + 1
           ############# Call PCNH here
           # if( index == hide):
           #     draw.rectangle((j*width,i*height,(j+1)*width,(i+1)*height), fill=(0, 0, 0), outline=(255, 255, 255))
           #     continue
            piece.save(path+'/piece_i'+str(i)+'_j'+str(j)+'_'+prediction+'.png')
            if prediction=='public': 
                img.paste(piece,(j*width,i*height))
    path1 = path+"/crop_1.png"
    path2 = path+"/original.png"
    img.save(path1)
    dr.save(path2)










total=1
correct=1
image_output=[]
label={0:'private',
1:'public'}
n=0
directory='/home/nchocka/security/finalProject/zerr_dataset_cleaned/'
for class_name in os.listdir(directory):
    if not class_name.startswith('.'):
        print('Class Name: '+class_name)
        #if class_name=='private':
        #    continue
        for image_name in os.listdir(directory+'/'+class_name):
            if not image_name.startswith('.'):
                #print(image_name)
               # try:
                n+=1
                if n<333:
                    continue
                 
                image = Image.open(Path(directory+'/'+class_name+'/'+image_name))
                prediction=predictor(image)
                    #CALL SPLITTER
                if prediction=='private':
                    splitter(image)
                #except:
                #    print('Could not Calculate Result...')
                #    continue
                image_output.append({'location':directory+'/'+class_name+'/'+image_name, 'label':class_name, 'predicted':prediction})
                print(image_output[-1])
                total=total+1
                if class_name==prediction:
                    correct=correct+1
                print('Accuracy: '+str(correct/total))
            break




