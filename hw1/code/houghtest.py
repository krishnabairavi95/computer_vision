import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import argparse


def myHough(img_name,ce_params,hl_params):
    img= cv2.imread(img_name)
    gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = (gray_img, ce_params[0], ce_params[1], apertureSize=ce_params[2])   ## Try 20, 40
    #plt.imshow(edges)
    #plt.show()
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0

    # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    #ines=cv2.HoughLines(edges, 1, np.pi / 180, 200)
    lines = cv2.HoughLinesP(edges, hl_params[0], hl_params[1], hl_params[2], np.array([]),
                            hl_params[3], hl_params[4])

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)


    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imwrite(img_name +'_hlines.jpg', lines_edges)





    # your function comes here!

if __name__=="__main__":

    # create a list of the params
    # for both your edge detector
    # hough transform

    edge_params = [50, 150,3]
    hl_params = [1,np.pi/180,70,100,20]

    img_name = "/Users/krishnabairavi/Downloads/hw1/data/img01.jpg"

    myHough(img_name, edge_params,hl_params)