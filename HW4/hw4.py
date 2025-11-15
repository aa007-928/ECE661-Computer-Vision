import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import io, draw

def Haaris_corner_det(img,sig):
    img_op = np.copy(img)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img /= 255.0

    M = int(4*sig)+1 if int(4*sig)%2!=0 else int(4*sig)+2
    filter_dx = np.ones((M,M))
    filter_dx[:,:int(M/2)] = -1
    filter_dy = np.ones((M,M))
    filter_dy[int(M/2):,:] = -1

    dx = cv.filter2D(img,-1,filter_dx)
    dy = cv.filter2D(img,-1,filter_dy)
    dx2 = np.multiply(dx,dx)
    dxdy = np.multiply(dx,dy)
    dy2 = np.multiply(dy,dy)

    nbr_5sig = int(5*sig)+1 #if int(5*sig)%2!=0 else int(5*sig)+2
    nbr_window = np.ones((nbr_5sig,nbr_5sig))

    sum_dx2 = cv.filter2D(dx2,-1,nbr_window)
    sum_dy2 = cv.filter2D(dy2,-1,nbr_window)
    sum_dxdy = cv.filter2D(dxdy,-1,nbr_window)

    det_C = sum_dx2*sum_dy2 - sum_dxdy*sum_dxdy
    Tr_C = sum_dx2 + sum_dy2
    k=0.04  #btw: 0.04-0.06
    R = det_C - k*(Tr_C*Tr_C)  
    R_thres = 0.01*R.max()  #R.mean()+1.0*R.std()
    #R_corner = R>R_thres
    
    N = int(nbr_5sig/2)
    corners = []
    for i in range(N,img.shape[0]-N):
        for j in range(N,img.shape[1]-N):
            if R[i-N:i+N+1,j-N:j+N+1].max() == R[i,j] and R[i,j] > R_thres:
                corners.append([(j,i),R[i,j]])

    p=200   #top p=100 corners
    top_corners = sorted(corners, key=lambda x: x[1], reverse=True)[:p]   

    for c in corners: #(top_corners):
        cv.circle(img_op,c[0],radius=1,thickness=-1,color=(0,0,255))
    
    plt.imshow(img_op[:,:,::-1])
    plt.show()
    
    return top_corners #corners


def SSD(img1,img2,img1_corner,img2_corner,sig):
    img1_op = np.copy(img1)
    img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img1 = img1.astype(np.float32)
    img1 /= 255.0
    img2_op = np.copy(img2)
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    img2 = img2.astype(np.float32)
    img2 /= 255.0
    paired_img = np.concatenate((img1_op,img2_op),axis=1)
    M = int(5*sig)+1
    N = int(M/2)

    shortest_dist_array = []
    for c1 in img1_corner:
        dist_array = []
        for c2 in img2_corner:
            f1_w = img1[c1[0][1]-N:c1[0][1]+N+1,c1[0][0]-N:c1[0][0]+N+1]
            f2_w = img2[c2[0][1]-N:c2[0][1]+N+1,c2[0][0]-N:c2[0][0]+N+1]
            dist = np.sum(np.square(f1_w-f2_w))
            dist_array.append([c2,dist])
        shortest_dist=sorted(dist_array,key=lambda x: x[1])[0]
        shortest_dist_array.append(shortest_dist)

    ssd_threshold = np.percentile([x[1] for x in shortest_dist_array], 30)
    #print(ssd_threshold)
    for p1,p2 in zip(img1_corner,shortest_dist_array):
        if p2[1]<ssd_threshold:#0.7:
            cv.circle(paired_img,p1[0],radius=3,thickness=-1,color=(0,0,255))
            cv.circle(paired_img,(p2[0][0][0]+img1.shape[1],p2[0][0][1]),radius=3,thickness=-1,color=(0,0,255))
            cv.line(paired_img,p1[0],(p2[0][0][0]+img1.shape[1],p2[0][0][1]),(0,255,0),1)

    plt.imshow(paired_img[:,:,::-1])
    plt.show()
    

def NCC(img1,img2,img1_corner,img2_corner,sig):
    img1_op = np.copy(img1)
    img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img1 = img1.astype(np.float32)
    img1 /= 255.0
    img2_op = np.copy(img2)
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    img2 = img2.astype(np.float32)
    img2 /= 255.0
    paired_img = np.concatenate((img1_op,img2_op),axis=1)
    M = int(5*sig)+1
    N = int(M/2)
    shortest_dist_array = []
    for c1 in img1_corner:
        dist_array = []
        for c2 in img2_corner:
            f1_w = img1[c1[0][1]-N:c1[0][1]+N+1,c1[0][0]-N:c1[0][0]+N+1]
            f1 = f1_w-f1_w.mean()
            f2_w = img2[c2[0][1]-N:c2[0][1]+N+1,c2[0][0]-N:c2[0][0]+N+1]
            f2 = f2_w-f2_w.mean()
            dist = np.sum(np.multiply(f1,f2))/np.sqrt(np.multiply(np.sum(np.square(f1)),np.sum(np.square(f2))))
            dist_array.append([c2,dist])
        shortest_dist=sorted(dist_array,key=lambda x: x[1])[-1]
        shortest_dist_array.append(shortest_dist)

    for p1,p2 in zip(img1_corner,shortest_dist_array):
        if p2[1]>0.8:
            cv.circle(paired_img,p1[0],radius=3,thickness=-1,color=(0,0,255))
            cv.circle(paired_img,(p2[0][0][0]+img1.shape[1],p2[0][0][1]),radius=3,thickness=-1,color=(0,0,255))
            cv.line(paired_img,p1[0],(p2[0][0][0]+img1.shape[1],p2[0][0][1]),(0,255,0),1)

    plt.imshow(paired_img[:,:,::-1])
    plt.show()


def SIFT_algo(img1,img2):
    img1_op = np.copy(img1)
    img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2_op = np.copy(img2)
    img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    SIFT = cv.SIFT_create()

    kp1,des1 = SIFT.detectAndCompute(img1,None)
    kp2,des2 = SIFT.detectAndCompute(img2,None)

    bf_matcher = cv.BFMatcher(cv.NORM_L1,crossCheck=True)

    matches = bf_matcher.match(des1,des2)
    matches = sorted(matches,key= lambda x: x.distance)

    p = 100 #top p matches
    op_img = cv.drawMatches(img1_op,kp1,img2_op,kp2,matches[:p],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(op_img[:,:,::-1])    #can remove flag above to see unmatched points
    plt.show()
    

def HarrisCornerDetector(img1, img2, sig):
    corner1 = Haaris_corner_det(img1,sig)
    corner2 = Haaris_corner_det(img2,sig)
    SSD(img1, img2, corner1, corner2,sig)
    NCC(img1, img2, corner1, corner2,sig)


def Harris_detect_sig(image_list):
    sigma_list = [0.8,1.0,1.2,1.4,1.6,1.8,2.0]
    for sig in sigma_list:
        print('SIGMA : ',sig)
        HarrisCornerDetector(image_list[0][0],image_list[0][1],sig)
        HarrisCornerDetector(image_list[1][0],image_list[1][1],sig)
        HarrisCornerDetector(image_list[2][0],image_list[2][1],sig)
        HarrisCornerDetector(image_list[3][0],image_list[3][1],sig)
        HarrisCornerDetector(image_list[4][0],image_list[4][1],sig)
        HarrisCornerDetector(image_list[5][0],image_list[5][1],sig)
        print('/n-------------------------------------------------------------------/n')
        print('/n-------------------------------------------------------------------/n')


if __name__ == "__main__":

    temple1 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images/temple_1.jpg')
    temple2 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images/temple_2.jpg')
    hovde1 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images\hovde_2.jpg')
    hovde2 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images\hovde_3.jpg')

    img1 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images\img1.jpg')
    img2 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images\img2.jpg')
    img3 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images\img3.jpg')
    img4 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images\img4.jpg')
    img7 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images\img7.jpg')
    img8 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images\img8.jpg')
    img9 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images\img9.jpg')
    img10 = cv.imread('D:\Purdue_fall24\Computer_Vision\HW4\HW4_images\img10.jpg')


    image_list = [(temple1,temple2),(hovde1,hovde2),(img1,img2),(img3,img4),(img7,img8),(img9,img10)] 
    Harris_detect_sig(image_list)   #Harris Corner Detection

    #SIFT
    SIFT_algo(hovde1,hovde2)
    SIFT_algo(temple1,temple2)
    SIFT_algo(img1,img2)
    SIFT_algo(img3,img4)
    SIFT_algo(img7,img8)
    SIFT_algo(img9,img10)

    #SuperPoint and SuperGlue: Used git repo (given)


