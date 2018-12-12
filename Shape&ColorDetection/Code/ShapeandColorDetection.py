# -*- coding: utf-8 -*-
"""
ArUco ID Dictionaries: 4X4 = 4-bit pixel, 4X4_50 = 50 combinations of a 4-bit pixel image
List of Dictionaries in OpenCV's ArUco library:
DICT_4X4_50      
DICT_4X4_100     
DICT_4X4_250     
DICT_4X4_1000    
DICT_5X5_50      
DICT_5X5_100     
DICT_5X5_250     
DICT_5X5_1000    
DICT_6X6_50      
DICT_6X6_100     
DICT_6X6_250     
DICT_6X6_1000    
DICT_7X7_50      
DICT_7X7_100     
DICT_7X7_250     
DICT_7X7_1000    
DICT_ARUCO_ORIGINAL

"""

import cv2
import numpy as np
import os
from collections import OrderedDict
import cv2.aruco as aruco
import csv
#classes and subclasses to import
def aruco_detect(path_to_image):
   
    img = cv2.imread(path_to_image)     #give the name of the image with the complete path
    id_aruco_trace = 0
    det_aruco_list = {}
    img2 = img[int(25):int(390),int(25):int(390),:]         #separate out the Aruco image from the whole image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters_create()
    corners,ids,i = aruco.detectMarkers(gray,aruco_dict,parameters = parameters)
    if ids == None:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
        parameters = aruco.DetectorParameters_create()
        corners,ids,i = aruco.detectMarkers(gray,aruco_dict,parameters = parameters)
    if ids == None:
        aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)
        parameters = aruco.DetectorParameters_create()
        corners,ids,i = aruco.detectMarkers(gray,aruco_dict,parameters = parameters)

    filep = open('outputtable.csv','a')
    filep.write(',')
    filep.write(str(ids[0][0]))
    filep.close()
    print('ArUco ID :',ids[0][0])
    return ids 
    cv2.destroyAllWindows()

    
class Colordetector:
        def __init__(self):
                # initialize the colors dictionary, containing the color
                # name as the key and the RGB tuple as the value
                colors = OrderedDict({
                        "Red": (255, 0, 0),
                        "Green": (0, 255, 0),
                        "Blue": (0, 0, 255),
                        "Orange": (255,160,0),
                        "Yellow": (255,255,0),
                        "Black": (0, 0, 0)})
 
                # allocate memory for the L*a*b* image, then initialize
                # the color names list
                self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
                self.colorNames = []
 
                # loop over the colors dictionary
                for (i, (name, rgb)) in enumerate(colors.items()):
                        # update the L*a*b* array and the color names list
                        self.lab[i] = rgb
                        self.colorNames.append(name)
 
                # convert the L*a*b* array from the RGB color space
                # to L*a*b*
                self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)


        def label(self, image, c):
                # construct a mask for the contour, then compute the
                # average L*a*b* value for the masked region
                mask = np.zeros(image.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.erode(mask, None, iterations=2)
                mean = cv2.mean(image, mask=mask)[:3]
 
                # initialize the minimum distance found thus far
                minDist = (np.inf, None)
 
                # loop over the known L*a*b* color values
                for (i, row) in enumerate(self.lab):
                        # compute the distance between the current L*a*b*
                        # color value and the mean of the image
                        d = np.linalg.norm(row[0] -  mean)
 
                        # if the distance is smaller than the current distance,
                        # then update the bookkeeping variable
                        if d < minDist[0]:
                                minDist = (d, i)
 
                # return the name of the color with the smallest distance
                return self.colorNames[minDist[1]]

class Shapedetector:
    def __init__(self):
        pass
    def detect(self,c):
        shape="Unidentified"
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.04*peri,True)
        if len(approx)==3:
            shape="Triangle"
        elif len(approx)==4:
            (x,y,w,h)=cv2.boundingRect(approx)
            ar=w/ float(h)
            shape = "Square" if ar >= 0.95 and ar <= 1.05 else "Rectangle"
            
        else:
            shape = "Circle"
        return shape

#from Counting import Counting



def sizeDetector(area, shape):  #function to detect large size shapes
                        #Detecting size of Circle
                        if shape == "Circle" :
                                        
                                        if area >= 8747:
                                            return 'Large'
                                        

                        #Detecting size of Square
                        if shape == "Square" :
                                       
                                        if area >= 11017:
                                            return 'Large'
                                        
                        #Detecting size of Rectangle
                        if shape == "Rectangle" :
                                       
                                        if area >= 21517:
                                            return 'Large'
                                        

                        #Detecting size of Triangle
                        if shape == "Triangle" :
                                        
                                        if area >= 5969.5:
                                            return 'Large'
                                        

class Counting:
                    counted=0
                    #Initialize the "Counting" class
                    def __init__(self,shape,color):
                                self.shape = shape
                                self.color = color 


#counted=0    
def main(path,sColor1,sShape1,sColor2,sShape2,imId):

                #Received image from given path where path take all image present in this path withextension .jpg
                img_no = path
                #load the image
                image = cv2.imread(img_no)
                #Detecting edges in image
                edges = cv2.Canny(image,100,100)
                #the L*a*b* color spaces
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                #thresholding it to reveal the shapes in the image
                ret,thresh = cv2.threshold(edges, 230, 255, cv2.THRESH_BINARY)

                # find contours in the thresholded image
                _, cnts, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                # initialize the shape detector and color labeler
                SD = Shapedetector() 
                CD = Colordetector()
                count1 = []
                array=[]
                # loop over the contours
                #cnts depects number of contours
                flag=0
                for c in cnts:

                                    # compute the co-ordinates of the contour
                                    x,y,w,h = cv2.boundingRect(c)
                                    area = cv2.contourArea(c)
                                    # detect the shape of the contour
                                    shape = SD.detect(c)
                                    # label the color to contour
                                    color = CD.label(lab, c)
                                    #Finding Size of contour
                                    size = sizeDetector(area,shape)
                                    s=Counting(shape,color)
                                    count1.append(s)
                                    text = "{}-{}-{}".format(color,shape,size)
                                                    
                                     #draw the contours
                                    if size == "Large" :
                                        if (color == sColor1 and shape == sShape1) or (color == sColor2 and shape == sShape2):

                                             if color == "Red" :
                                                     cv2.drawContours(image, [c], -1, (0, 255, 0), 25)
                                             if color == "Green" :
                                                     cv2.drawContours(image, [c], -1, (255, 0, 0), 25)
                                             if color == "Blue":
                                                     cv2.drawContours(image, [c], -1, (0, 0, 255), 25)
                                             
                                             M = cv2.moments(c)
                                             cx = int(M['m10']/M['m00'])
                                             cy = int(M['m01']/M['m00'])
                                             centroid = "({};{})".format(cx,cy)
                                             
                                             if flag%2 == 0 :
                                                 filep = open('outputtable.csv','a')
                                                 filep.write(',')
                                                 filep.write(centroid)
                                                 filep.close()
                                                 print("X :",cx)
                                                 print("Y :",cy)
                                             flag += 1    
                                             text1= "{},{}".format(cx,cy)
                                             text1="("+text1+")"
                                             cv2.putText(image, text1, (cx, cy),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                                           
                if color == "Black" and size == "Large": #printing ArUco id in the output image
                    M = cv2.moments(c)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    disp = imId[0][0]
                    disp = str(disp)
                    cv2.putText(image, disp, (cx, cy),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 0), 1)

                return image
                                   
                
   #main where the path is set for the directory containing the test images
if __name__ == "__main__":
    mypath = '.'
    #getting all files in the directory
    onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if f.endswith(".jpg")]
    #iterate over each file in the directory
    count = 0
    for fp in onlyfiles:
        #Open the csv to write in append mode
        filep = open('outputtable.csv','a')
        #this csv will later be used to save processed data, thus write the file name of the image
        print(fp)
        data = fp.split("\\")
        count += 1
        print("")
        print(data[1])
        print("===========")
        #We replace this  "filep.write(fp)"
        filep.write(data[1])
        #close the file so that it can be reopened again later
        filep.close()
        print("Count")
        print(count)
        requirements = [['Green','Triangle','Blue','Circle'],['Red','Square','None','None'],['Green','Circle','Red','Triangle'],['Blue','Triangle','Blue','Square'],['Red','Circle','Green','Square']]
        #these are basically inputs object1_color,object1_shape,object2_color,object2_shape
        #you can add more inputs by adding more details in requirement matrix !important also add extra image to path
        print(requirements)
        aruco_id = aruco_detect(fp)
        

        
        for i in range(count-1,count):
                
                        print (requirements[i][0])
                        print (requirements[i][1])
                        #process the image
                        data = main(fp,requirements[i][0],requirements[i][1],requirements[i][2],requirements[i][3],aruco_id)
                        #cv2.imshow('image',data)
                        #data is basically an output image
                        cv2.imwrite('image'+str(count)+'.jpg',data) #current input images in a path will automatically converted as a output image
                        #open the csv
                        filep = open('outputtable.csv','a')
                        #make a newline entry so that the next image data is written on a newline
                        filep.write('\n')
                        #close the file
                        filep.close()
                        print("\n")


#please add all input images to present directory for a code test
