#Plotting and dimensions of the original image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# reading in an image
image = mpimg.imread('C:/Users/Akanksha/Desktop/img1.jpg')
# printing out some stats and plotting the image
print('This image is:', type(image), 'with dimensions:', image.shape)
a = image.shape
##plt.imshow(image)
#plt.show()
##print (a)

height= a[0]
width= a[1]


#Cannyedging

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math



def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 # <-- This line altered for grayscale.
    
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
]
image = mpimg.imread('C:/Users/Akanksha/Desktop/img1.jpg')
#plt.figure()
#plt.imshow(image)
#plt.show()

# Convert to grayscale here.
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Call Canny Edge Detection here.
cannyed_image = cv2.Canny(gray_image, 100, 200)

#Cropping operation
'''cropped_image = region_of_interest(
    cannyed_image,
    np.array([region_of_interest_vertices], np.int32)
)
#plt.figure()
plt.imshow(cropped_image)
#plt.show()'''



'''#Utility function for cropping

import numpy as np
import cv2
def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    #channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    #match_mask_color = (255,) * channel_count

    match_mask_color=255 #since the image is grayscale now
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
'''
#Utility function for rendering
def drawLines(img, lines, color=[255, 0, 0], thickness=3):
        
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    #img = np.copy(imgo)# Create a blank image that matches the original in size.
    #line_img = np.zeros(
        '''(
           img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )'''
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    #img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)
    # Return the modified image.
    return img

#Pipeline to output image with lines
#Hough transform

'''image = mpimg.imread('C:/Users/Akanksha/Desktop/roadlane2.jpg')

plt.figure()
plt.imshow(image)
plt.show()
'''
#Pipeline to output image with lines
def pipeline(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 310,500)
    
    '''cropped_image = region_of_interest(
            cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )'''
    

    lines = cv2.HoughLinesP(
        cannyed_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    print(lines)
    line_image = drawLines(image, lines, color=[255, 0, 0], thickness=3) #calling the fn

    plt.figure()
    ##plt.imshow(line_image)
    ##plt.show() #Use this only to draw, not avg



#Grouping the lines into left and right groups

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])


#Averaging
    min_y = image.shape[0] * (3 / 5)#Just below the horizon
    max_y = image.shape[0] #The bottom of the image

#Operations that generate a linear fn that match x & y for each group
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

#Using the above as input to the fn draw_lines

    line_image = drawLines(
        image,
        [[
            [left_x_start, int(max_y), left_x_end, int(min_y)],
        #[right_x_start, int(max_y), right_x_end, int(min_y)],
        ]],
        color=[255, 0, 0],
        thickness=5
    )
    line_image = drawLines(
        image,
        [[
        #[left_x_start, int(max_y), left_x_end, int(min_y)],
            [right_x_start, int(max_y), right_x_end, int(min_y)],
        ]],
        color=[255, 0, 0],
        thickness=5
    )
    plt.figure()
    ##plt.imshow(line_image)
    ##plt.show()
    return line_image
##pipeline(image)



import numpy as np
import cv2

cap = cv2.VideoCapture('video1.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    print(ret)
    # Our operations on the frame come here
    try :
        img = pipeline(frame)
    except Exception :
        print('')
        img = frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

###Video processing pipeline
##from moviepy.editor import VideoFileClip
##from IPython.display import HTML
##white_output = 'solidWhiteRight_output.mp4'
##clip1 = VideoFileClip("solidWhiteRight_input.mp4")
##white_clip = clip1.fl_image(pipeline)
##white_clip.write_videofile(white_output, audio=False)
