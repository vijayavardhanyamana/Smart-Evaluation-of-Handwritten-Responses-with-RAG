import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os


cordinates =[]

# def click_event(event, x, y, flags, params):
#    global cordinates,count
#    if event == cv2.EVENT_LBUTTONDOWN:
#       cordinates.append([x,y])

#       cv2.putText(img, f'({x},{y})',(x,y),
#       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#       cv2.circle(img, (x,y), 3, (0,255,255), -1)
#       count = count + 1
      
      
      
def four_point_transform(image, pts):
    rect = pts
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    rect = np.array(rect, dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def remove_shadow(image):
    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    
    return result,result_norm



    
def analise(image): 
    global line, binary_image1, x_scaling , y_scaling
    kernel = np.ones((1,250),np.uint8)
     
    dilation = cv2.dilate(image, kernel, iterations = 2)
    
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)   
    # cv2.imshow('Image',dilation)
    # cv2.waitKey(0)
    
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in reversed(contours):
            x, y, w, h = cv2.boundingRect(i)
            if cv2.contourArea(i)<20 :
                continue
            elif h < 8:
                continue
            else:
                scaling_factor_in_y = 0.5
                scaling_factor_in_x = 0
                resized_contour = i.copy()
                
                resized_contour = i * [x_scaling, y_scaling] 
                
                resized_contour = resized_contour.astype(int)      
                final_image__ = np.zeros_like(binary_image1)
                cv2.drawContours(final_image__, [resized_contour], 0, (255), -1)
                            
                kernel_dil = np.ones((3,3),np.uint8)
                final_image__ = cv2.dilate(final_image__,kernel_dil,iterations = 3)
                            
                            
                line_image_final = cv2.bitwise_and(final_image__, binary_image1)
                line.append(line_image_final)
                # cv2.namedWindow("Line image", cv2.WINDOW_NORMAL)   
                # cv2.imshow('Line image',line_image_final)
                # cv2.waitKey(0)
  
                
   
def image_resize_and_errosion(image):

    height, width = image.shape[:2]
    height = height + 1 * height
    height = int(height)
    
    resized_image = cv2.resize(image, (width, height))

    kernel = np.ones((13,1),np.uint8)   

    erosion = cv2.erode(resized_image,kernel,iterations = 1)

    return erosion


x_scaling = 0
y_scaling = 0
binary_image1 = 0
line = 0 
line_length = 0
count = 0

def convert_image(img):
    folder_path = 'images'

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
        
        
        
    global x_scaling,y_scaling,binary_image1,line,line_lenght,count
    # img = cv2.imread(image_file)
    img_copy = np.copy(img)
    line_lenght = 250
    rect_image = img

    # removing the shadow in the image
    image1, image2_ = remove_shadow(rect_image)

    # converting into grayscale
    gray_ = cv2.cvtColor(image2_,cv2.COLOR_BGR2GRAY)
    
    # cv2.namedWindow("grayscale image", cv2.WINDOW_NORMAL)   
    # cv2.imshow('grayscale image',gray_)
    # cv2.waitKey(0)

    # convrting into binaryimage
    _, binary_image_ = cv2.threshold(gray_, 200, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow("binary image", cv2.WINDOW_NORMAL)   
    # cv2.imshow('binary image',binary_image_)
    # cv2.waitKey(0)

    inverted_binary_image_ = 255 - binary_image_

    binary_image1 = np.copy(inverted_binary_image_)

    y_height ,x_width= rect_image.shape[:2]

    # print("image width, height =", x_width, y_height)

    # resizing the image
    new_width = 500*5
    new_height = 705*5

    x_scaling = x_width/new_width
    y_scaling = y_height/new_height

    # print("After resizing width, height", new_width , new_height)
    rect_image = cv2.resize(rect_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    # cv2.namedWindow("resized image", cv2.WINDOW_NORMAL)   
    # cv2.imshow('resized image',rect_image)
    # cv2.waitKey(0)
    
    # removing the shadow in the image
    image1, image2 = remove_shadow(rect_image)

    # converting into grayscale
    gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    # cv2.namedWindow("grayscale image", cv2.WINDOW_NORMAL)   
    # cv2.imshow('grayscale image',gray)
    # cv2.waitKey(0)
    
    # convrting into binaryimage
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow("binary image", cv2.WINDOW_NORMAL)   
    # cv2.imshow('binary image',gray)
    # cv2.waitKey(0)

    # inverting the pixel
    inverted_binary_image = 255 - binary_image

    kernel = np.ones((2,2),np.uint8)


    # performing  erosion to remove noise
    erosion = cv2.erode(inverted_binary_image,kernel,iterations = 1)
    # cv2.namedWindow("erosion", cv2.WINDOW_NORMAL)   
    # cv2.imshow('erosion',erosion)
    # cv2.waitKey(0)


    # performing Dilution operatiom
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    # cv2.namedWindow("dilation", cv2.WINDOW_NORMAL)   
    # cv2.imshow('dilation',erosion)
    # cv2.waitKey(0)
    

    new_image = np.copy(dilation)
    new_image = 255 - new_image 


    # defining kernal size
    kernel = np.ones((1,250),np.uint8)


    # performing Dilution operatiom
    dilation_1 = cv2.dilate(dilation,kernel,iterations = 2)
    # cv2.namedWindow("dilation_1", cv2.WINDOW_NORMAL)   
    # cv2.imshow('dilation_1',dilation_1)
    # cv2.waitKey(0)
    
    contours, _ = cv2.findContours(dilation_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    line = []
    # line saparation 
    for i in reversed(contours):
        x, y, w, h = cv2.boundingRect(i)
        if cv2.contourArea(i)<20:
            continue
        elif h < 10:
            continue
        else:
            cv2.drawContours(new_image, [i],-1,(0),2)
            final_image_ = np.zeros_like(binary_image)
            cv2.drawContours(final_image_, [i], 0, (255), -1)
            
            # cv2.namedWindow("final_image_", cv2.WINDOW_NORMAL)   
            # cv2.imshow('final_image_',final_image_)
            # cv2.waitKey(0)
            
            
            line_image = cv2.bitwise_and(final_image_, dilation)
            # cv2.namedWindow("line_image", cv2.WINDOW_NORMAL)   
            # cv2.imshow('line_image',line_image)
            # cv2.waitKey(0)
            
                        
            analise(line_image)         
            

    count = 0
    kernel1 = np.ones((8,8),np.uint8)
    for line_image in line:
            
        dilation_2 = cv2.dilate(line_image,kernel1,iterations = 2)
        
        contours1, _ = cv2.findContours(dilation_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sorted_contours = sorted(contours1, key=lambda c: cv2.boundingRect(c)[0])
        
        for j in sorted_contours:
            x1,y1,w1,h1 = cv2.boundingRect(j)
            final_image = line_image[y1:y1+h1,x1:x1+w1]
            image_name ="images/"+str(count)+".png"
            final_image = 255 - final_image
            cv2.imwrite(image_name, final_image)
            count=count+1

    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    


# img = cv2.imread("ans_image/1.jpg")
# convert_image(img)

