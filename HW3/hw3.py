import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import io, draw


#function for getting homography
def find_H(img_d,img_r):
    b = img_r.flatten()
    A = np.zeros((8,8))

    for i,pt in enumerate(img_d):
        A[2*i,:] =  np.array([pt[0],pt[1],1,0,0,0,-pt[0]*img_r[i][0],-pt[1]*img_r[i][0]])
        A[2*i+1,:] =  np.array([0,0,0,pt[0],pt[1],1,-pt[0]*img_r[i][1],-pt[1]*img_r[i][1]])

    x = np.linalg.inv(A)@b

    H = np.array([[x[0],x[1],x[2]],[x[3],x[4],x[5]],[x[6],x[7],1]])
    return H

#function for transforming pixel coordinates
def transform(image_d,H):
    img_dims = np.array([[0,0],[image_d.shape[0],0],[image_d.shape[0],image_d.shape[1]],[0,image_d.shape[0]]])
    dim_arr = np.zeros(img_dims.shape,dtype=int)
    for i in range(len(img_dims)):
        x = np.append(img_dims[i],[1])
        x_d = np.linalg.inv(H)@x
        dim_arr[i][0] = int(x_d[0]/x_d[2])
        dim_arr[i][1] = int(x_d[1]/x_d[2])

    x_lim,y_lim = max(dim_arr[:,1])-min(dim_arr[:,1]),max(dim_arr[:,0])-min(dim_arr[:,0])
    image_r = np.zeros((y_lim,x_lim,3),dtype=int)

    for i in range(y_lim):
        for j in range(x_lim):
            x = np.array([i+min(dim_arr[:,0]), j+min(dim_arr[:,1]), 1])
            x_d = H@x
            xi = int(x_d[0]/x_d[2])
            xj = int(x_d[1]/x_d[2])
            if 0<= xj <image_d.shape[0] and 0<= xi < image_d.shape[1]:
                image_r[i,j] = image_d[xj,xi]
    return image_r

#remove projective distortion
def rm_projDist(img_pt,img):
    #pts order -> starting from upper left corner and going anticlockwise
    one_vec = np.ones((len(img_pt),1))
    img_ptHC = np.hstack((img_pt,one_vec))
    l1 = np.cross(img_ptHC[0],img_ptHC[1]) 
    l2 = np.cross(img_ptHC[2],img_ptHC[3])
    l3 = np.cross(img_ptHC[0],img_ptHC[3])
    l4 = np.cross(img_ptHC[1],img_ptHC[2])
    Vp1 = np.cross(l1,l2)
    Vp2 = np.cross(l3,l4)
    VL = np.cross(Vp1,Vp2)
    VL = VL/np.linalg.norm(VL)    #

    H_co = np.array([[1,0,0],[0,1,0],[VL[0],VL[1],VL[2]]])
    # print(H_co)
    # print(f'det: {np.linalg.det(H_co)}')
    H_co = np.linalg.inv(H_co)
    tf = transform(img,H_co)
    plt.imshow(tf[:,:,::-1])
    plt.show()
    
    return H_co

def pd_lines2(pd_points,H):    #2 perpendicular line pairs from i/p points
    one_vec = np.ones((len(pd_points),1))
    img_ptHC = np.hstack((pd_points,one_vec))

    for i,pt in enumerate(img_ptHC):
        img_ptHC[i] = H@pt

    l1 = np.cross(img_ptHC[0],img_ptHC[1])
    l1/=np.linalg.norm(l1) 
    l2 = np.cross(img_ptHC[2],img_ptHC[3])
    l2/=np.linalg.norm(l2) 
    l3 = np.cross(img_ptHC[0],img_ptHC[3])
    l3/=np.linalg.norm(l3) 
    l4 = np.cross(img_ptHC[1],img_ptHC[2])
    l4/=np.linalg.norm(l4) 

    return [l4,l2,l3,l1]

#remove affine distortion
def rm_affDist(lines,img_pt=None,img=None): #perpendicular line pairs given in i/p
    H_matrix = np.zeros((3,3))
    S_matrix = np.zeros((2,2))

    A = np.zeros((2,2))
    b = np.zeros((2,1))
    A[0,0] = lines[0][0]*lines[1][0]
    A[0,1] = lines[0][0]*lines[1][1]+lines[0][1]*lines[1][0]
    A[1,0] = lines[2][0]*lines[3][0]
    A[1,1] = lines[2][0]*lines[3][1]+lines[2][1]*lines[3][0]
    b[0,0] = -lines[0][1]*lines[1][1]
    b[1,0] = -lines[2][1]*lines[3][1]
    x = np.linalg.inv(A)@b

    S_matrix[0,0] = x[0,0]
    S_matrix[0,1] = x[1,0]
    S_matrix[1,0] = x[1,0]
    S_matrix[1,1] = 1

    U,D_s,V = np.linalg.svd(S_matrix)
    D_a = np.sqrt(D_s)
    A_matrix = U@np.diag(D_a)@V

    H_matrix[0:2,0:2] = A_matrix
    H_matrix[2,2] = 1

    return H_matrix

#2-step approach
def rm_dist_2step(img_pt,img,pd_points=None):
    H_proj = rm_projDist(img_pt,img)
    lines = pd_lines2(pd_points,H_proj)
    H_aff = rm_affDist(lines,img_pt=None,img=None)
    H_comb = H_proj@H_aff
    #print(H_comb)
    tf = transform(img,H_comb)
    plt.imshow(tf[:,:,::-1])
    plt.show()


# 5 perpendicular line pairs from points forming a square
def pd_lines5(pd_points):
    one_vec = np.ones((len(pd_points),1))
    img_ptHC = np.hstack((pd_points,one_vec))
    l1 = np.cross(img_ptHC[0],img_ptHC[1])
    l1 /= np.linalg.norm(l1)
    l2 = np.cross(img_ptHC[1],img_ptHC[2])
    l2 /= np.linalg.norm(l2)
    l3 = np.cross(img_ptHC[2],img_ptHC[3])
    l3 /= np.linalg.norm(l3)
    l4 = np.cross(img_ptHC[3],img_ptHC[0])
    l4 /= np.linalg.norm(l4)
    l5 = np.cross(img_ptHC[0],img_ptHC[2])
    l5 /= np.linalg.norm(l5)
    l6 = np.cross(img_ptHC[1],img_ptHC[3])
    l6 /= np.linalg.norm(l6)

    return [(l1,l2),(l2,l3),(l3,l4),(l4,l1),(l5,l6)]

#1-step approach
def rm_dist_1step(img,line_pair):
    H = np.zeros((3,3))
    A = np.zeros((5,5))
    b = np.zeros((5,1))

    for i in range(0,5):
        A[i,0] = line_pair[i][0][0]*line_pair[i][1][0]
        A[i,1] = (line_pair[i][0][0]*line_pair[i][1][1]+line_pair[i][0][1]*line_pair[i][1][0])/2
        A[i,2] = line_pair[i][0][1]*line_pair[i][1][1]
        A[i,3] = (line_pair[i][0][0]*line_pair[i][1][2]+line_pair[i][0][2]*line_pair[i][1][0])/2
        A[i,4] = (line_pair[i][0][1]*line_pair[i][1][2]+line_pair[i][0][2]*line_pair[i][1][1])/2
        b[i,0] = -line_pair[i][0][2]*line_pair[i][1][2]
    
    x = np.linalg.inv(A)@b
    C = np.array([[x[0][0],x[1][0]/2,x[3][0]/2],[x[1][0]/2,x[2][0],x[4][0]/2],[x[3][0]/2,x[4][0]/2,1]])
    C /= np.linalg.norm(C)

    S = C[0:2,0:2]
    Av = C[0:2,2]

    U,D_s,V = np.linalg.svd(S)
    D_a = np.sqrt(D_s)
    A_matrix = V@np.diag(D_a)@V.T
    v = np.linalg.inv(A_matrix)@Av

    H[0:2,0:2] = A_matrix
    H[2,0:2] = v.T
    H[2,2] = 1
    #print('H',H)
    tf = transform(img,H)
    plt.imshow(tf[:,:,::-1])
    plt.show()


if __name__ == "__main__":

    img_board = cv.imread('D:\Purdue_fall24\Computer_Vision\HW3\HW3_images/board_1.jpeg') 
    img_corridor = cv.imread('D:\Purdue_fall24\Computer_Vision\HW3\HW3_images\corridor.jpeg') 
    img_3self = cv.imread('D:\Purdue_fall24\Computer_Vision\HW3\HW3_images\wall_hanging.jpg') 
    img_4self = cv.imread('D:\Purdue_fall24\Computer_Vision\HW3\HW3_images/amp.jpg') 

    #Distorted

    #Board
    x1, y1 = 74,423
    x2, y2 = 424,1783
    x3, y3 = 1348,1946
    x4, y4 = 1217,144
    Img1 = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])

    #Corridor
    x1, y1 = 1078,521
    x2, y2 = 1065,1221
    x3, y3 = 1296,1349 
    x4, y4 = 1307,484
    Img2 = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])

    #wall_hanging
    x1, y1 = 511,875
    x2, y2 = 116,2313
    x3, y3 = 1658,2335
    x4, y4 = 1921,430
    Img3 = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])

    #amp
    x1, y1 = 127,526
    x2, y2 = 383,2130
    x3, y3 = 1822,2120 
    x4, y4 = 2023,471
    Img4 = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])


    #Undistorted

    #Board
    x1, y1 = 0,0
    x2, y2 = 1200,0
    x3, y3 = 1200,800
    x4, y4 = 0,800
    Img1UD= np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])

    #Corridor
    x1, y1 = 0,0
    x2, y2 = 600,0
    x3, y3 = 600,300
    x4, y4 = 0,300
    Img2UD = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])


    #wall_hanging
    x1, y1 = 0,0
    x2, y2 = 500,0
    x3, y3 = 500,500
    x4, y4 = 0,500
    Img3UD= np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])

    #amp
    x1, y1 = 0,0
    x2, y2 = 600,0
    x3, y3 = 600,600
    x4, y4 = 0,600
    Img4UD = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])

    H = find_H(Img1UD,Img1)
    #print(H)
    tf = transform(img_board,H)
    plt.imshow(tf[:,:,::-1])
    plt.show()

    rm_projDist(Img1,img_board)
    pd_points_2Lpair = np.array([(600,692),(895,649),(879,538),(590,586)])
    rm_dist_2step(Img1,img_board,pd_points_2Lpair)

    pd_points_5Lpair = np.array([(148,463),(1076,251),(1169,1252),(330,1241)])
    line_pair = pd_lines5(pd_points_5Lpair)
    rm_dist_1step(img_board,line_pair)

    H = find_H(Img3UD,Img3)
    tf = transform(img_3self,H)
    plt.imshow(tf[:,:,::-1])
    plt.show()

    rm_projDist(Img3,img_3self)
    pd_points_2Lpair = np.array([(116,2313),(1658,2335),(1921,430),(511,875)])
    rm_dist_2step(Img3,img_3self,pd_points_2Lpair)

    pd_points_5Lpair = np.array([(511,875),(1921,430),(1658,2335),(116,2313)])
    line_pair = pd_lines5(pd_points_5Lpair)
    rm_dist_1step(img_3self,line_pair)

    H = find_H(Img4UD,Img4)
    tf = transform(img_4self,H)
    plt.imshow(tf[:,:,::-1])
    plt.show()

    rm_projDist(Img4,img_4self)
    pd_points_2Lpair = np.array([(127,526),(383,2130),(1822,2120),(2023,471)])
    rm_dist_2step(Img4,img_4self,pd_points_2Lpair)

    pd_points_5Lpair = np.array([(127,526),(383,2130),(1822,2120),(2023,471)])
    line_pair = pd_lines5(pd_points_5Lpair)
    rm_dist_1step(img_4self,line_pair)

    H = find_H(Img2UD,Img2)
    tf = transform(img_corridor,H)
    plt.imshow(tf[:,:,::-1])
    plt.show()

    rm_projDist(Img2,img_corridor)
    pd_points = np.array([(600,692),(895,649),(879,538),(590,586)])
    rm_dist_2step(Img2,img_corridor,Img2)

    pd_points_5Lpair = np.array([(932,564),(1301,501),(1291,1330),(929,1129)])
    line_pair = pd_lines5(pd_points_5Lpair)
    rm_dist_1step(img_corridor,line_pair)



