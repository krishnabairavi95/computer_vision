import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import pdb
import argparse


def myHough(img_name,ce_params,hl_params):

    img= cv2.imread('../data/'+img_name+'.jpg')
    gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    line_image = np.copy(img) * 0

    edge_img = cv2.Canny(gray_img, ce_params[0], ce_params[1], apertureSize=ce_params[2])
    ## Try 20, 40

    lines = cv2.HoughLinesP(edge_img, hl_params[0], hl_params[1], hl_params[2], np.array([]),
                            hl_params[3], hl_params[4])

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)


    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    cv2.imwrite('../results/'+img_name +'_hlines.jpg', lines_edges)


    # your function comes here!

if __name__=="__main__":

    # create a list of the params 
    # for both your edge detector 
    # hough transform

    edge_params = [50, 150,3]
    hl_params = [1,np.pi/180,100,50,3]

  #  img_path= "/Users/krishnabairavi/Downloads/hw1/data/img09.jpg"

    img_name = "img09"

    myHough(img_name,edge_params,hl_params)