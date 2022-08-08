import os
import cv2


file_dir = '/raid/data/parkingslot/img/'
list = []
for root ,dirs, files in os.walk(file_dir):
    for file in files:
        list.append(file)     
list.sort()
print(list)
print(len(list))

video = cv2.VideoWriter('/raid/data/test.avi',cv2.VideoWriter_fourcc(*'MJPG'),5,(1920,1080))

for i in range(1,len(list)):

    print(i)
    img = cv2.imread('/raid/data/parkingslot/img/'+list[i-1])     
    img = cv2.resize(img,(1920,1080))
    video.write(img)

video.release()
                                                                                                                                                                                                                                                                                                                                                                            
