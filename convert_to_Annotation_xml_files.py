
from pascal_voc_writer import Writer
import os

with open('train.txt', 'r') as file_in:
    obj_attributes_string = file_in.readline().strip()
    obj_attributes_split = obj_attributes_string.split(' ')
    print(obj_attributes_split)

n=0
n1=0
with open('train.txt', 'r') as handle:
    c=0
    for line in handle:
        c=c+1
        x = line.split()
        path = 'images\\'+(x[2].split('.')[0]).split('\\')[1]+'.jpg'
        for i in range(int(x[3])):
            if (int(x[6+4*i])>int(x[4+4*i])):
                print('images\\'+(x[2].split('.')[0]).split('\\')[1]+'.jpg')
                n1=n1+1
            n=n+1
        if (2==2):
##                # Writer(path, width, height)
                print(path)
                writer = Writer(path,3680,2760)
        ##
        ### ::addObject(name, xmin, ymin, xmax, ymax)
                for i in range(int(x[3])):
                    writer.addObject('pothole', x[6+4*i],x[7+4*i],x[4+4*i],x[5+4*i])

        ### ::save(path)
        ##
                print(path)
                writer.save('annotations\\'+(x[2].split('.')[0]).split('\\')[1]+'.xml')
####       
