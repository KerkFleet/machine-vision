import cv2
import numpy as np

def DetectFASTCorner(img, delta=20):
    
    img = cv2.GaussianBlur(img,(5,5),0.787)
    offset = np.array([[0, -3], [1, -3], [2, -2], [3, -1], [3, 0], [3, 1], \
                       [2, 2], [1, 3], [0, 3], [-1, 3], [-2, 2], [-3, 1], [-3, 0], [-3, -1], [-2, -2], [-1, -3]])

    b = 4; # b is greater than equal to 4

    # get pixel values for 16 points 
    v0_t = img[b:-b, b:-b].astype(int)+delta
    v0_b = img[b:-b, b:-b].astype(int)-delta
    v1 = img[b+offset[0,1]:offset[0,1]-b,b+offset[0,0]:offset[0,0]-b].astype(int)
    v2 = img[b+offset[1,1]:offset[1,1]-b,b+offset[1,0]:offset[1,0]-b].astype(int)
    v3 = img[b+offset[2,1]:offset[2,1]-b,b+offset[2,0]:offset[2,0]-b].astype(int)
    v4 = img[b+offset[3,1]:offset[3,1]-b,b+offset[3,0]:offset[3,0]-b].astype(int)
    v5 = img[b+offset[4,1]:offset[4,1]-b,b+offset[4,0]:offset[4,0]-b].astype(int)
    v6 = img[b+offset[5,1]:offset[5,1]-b,b+offset[5,0]:offset[5,0]-b].astype(int)
    v7 = img[b+offset[6,1]:offset[6,1]-b,b+offset[6,0]:offset[6,0]-b].astype(int)
    v8 = img[b+offset[7,1]:offset[7,1]-b,b+offset[7,0]:offset[7,0]-b].astype(int)
    v9 = img[b+offset[8,1]:offset[8,1]-b,b+offset[8,0]:offset[8,0]-b].astype(int)
    v10 = img[b+offset[9,1]:offset[9,1]-b,b+offset[9,0]:offset[9,0]-b].astype(int)
    v11 = img[b+offset[10,1]:offset[10,1]-b,b+offset[10,0]:offset[10,0]-b].astype(int)
    v12 = img[b+offset[11,1]:offset[11,1]-b,b+offset[11,0]:offset[11,0]-b].astype(int)
    v13 = img[b+offset[12,1]:offset[12,1]-b,b+offset[12,0]:offset[12,0]-b].astype(int)
    v14 = img[b+offset[13,1]:offset[13,1]-b,b+offset[13,0]:offset[13,0]-b].astype(int)
    v15 = img[b+offset[14,1]:offset[14,1]-b,b+offset[14,0]:offset[14,0]-b].astype(int)
    v16 = img[b+offset[15,1]:offset[15,1]-b,b+offset[15,0]:offset[15,0]-b].astype(int)

    # corner type 1 
    b_1_and_9 = np.logical_and(v0_t < v1, v0_t < v9)
    b_5_or_13 = np.logical_or(v0_t < v5, v0_t < v13)

    b_5_and_13 = np.logical_and(v0_t < v5, v0_t < v13)  
    b_1_or_9 = np.logical_or(v0_t < v1, v0_t < v9)

    b_fast_test1 = np.logical_or(np.logical_and(b_1_or_9, b_5_and_13), np.logical_and(b_1_and_9, b_5_or_13))

    v0_t_tested1 = v0_t[b_fast_test1]

    b1 = v1[b_fast_test1] > v0_t_tested1
    b2 = v2[b_fast_test1] > v0_t_tested1
    b3 = v3[b_fast_test1] > v0_t_tested1
    b4 = v4[b_fast_test1] > v0_t_tested1
    b5 = v5[b_fast_test1] > v0_t_tested1
    b6 = v6[b_fast_test1] > v0_t_tested1
    b7 = v7[b_fast_test1] > v0_t_tested1
    b8 = v8[b_fast_test1] > v0_t_tested1
    b9 = v9[b_fast_test1] > v0_t_tested1
    b10 = v10[b_fast_test1] > v0_t_tested1
    b11 = v11[b_fast_test1] > v0_t_tested1
    b12 = v12[b_fast_test1] > v0_t_tested1
    b13 = v13[b_fast_test1] > v0_t_tested1
    b14 = v14[b_fast_test1] > v0_t_tested1
    b15 = v15[b_fast_test1] > v0_t_tested1
    b16 = v15[b_fast_test1] > v0_t_tested1

    diff1 = v1[b_fast_test1] - v0_t_tested1
    diff2 = v2[b_fast_test1] - v0_t_tested1
    diff3 = v3[b_fast_test1] - v0_t_tested1
    diff4 = v4[b_fast_test1] - v0_t_tested1 
    diff5 = v5[b_fast_test1] - v0_t_tested1 
    diff6 = v6[b_fast_test1] - v0_t_tested1 
    diff7 = v7[b_fast_test1] - v0_t_tested1 
    diff8 = v8[b_fast_test1] - v0_t_tested1 
    diff9 = v9[b_fast_test1] - v0_t_tested1 
    diff10 = v10[b_fast_test1] - v0_t_tested1
    diff11 = v11[b_fast_test1] - v0_t_tested1 
    diff12 = v12[b_fast_test1] - v0_t_tested1 
    diff13 = v13[b_fast_test1] - v0_t_tested1 
    diff14 = v14[b_fast_test1] - v0_t_tested1 
    diff15 = v15[b_fast_test1] - v0_t_tested1 
    diff16 = v16[b_fast_test1] - v0_t_tested1

    c_res1 = diff1*(diff1>0).astype(int)+diff2*(diff2>0).astype(int)+diff3*(diff3>0).astype(int)+diff4*(diff4>0).astype(int) \
            + diff5*(diff5>0).astype(int)+diff6*(diff6>0).astype(int)+diff7*(diff7>0).astype(int)+diff8*(diff8>0).astype(int) \
            + diff9*(diff9>0).astype(int)+diff10*(diff10>0).astype(int)+diff11*(diff11>0).astype(int)+diff12*(diff12>0).astype(int) \
            + diff13*(diff13>0).astype(int)+diff14*(diff14>0).astype(int)+diff15*(diff15>0).astype(int)+diff16*(diff16>0).astype(int) 

    b_1to12 = ((((((((((((b1.astype(int)+b2.astype(int))*b2.astype(int))+b3.astype(int))*b3.astype(int) \
                       +b4.astype(int))*b4.astype(int)+b5.astype(int))*b5.astype(int)+b6.astype(int))*b6.astype(int) \
                    +b7.astype(int))*b7.astype(int)+b8.astype(int))*b8.astype(int)+b9.astype(int))*b9.astype(int) \
                 +b10.astype(int))*b10.astype(int)+b11.astype(int))*b11.astype(int)+b12.astype(int))*b12.astype(int)

    b_2to13 = (b_1to12+b13.astype(int))*b13.astype(int)
    b_3to14 = (b_2to13+b14.astype(int))*b14.astype(int)
    b_4to15 = (b_3to14+b15.astype(int))*b15.astype(int)
    b_5to16 = (b_4to15+b16.astype(int))*b16.astype(int)
    b_6to1 = (b_5to16+b1.astype(int))*b1.astype(int)
    b_7to2 = (b_6to1+b2.astype(int))*b2.astype(int)
    b_8to3 = (b_7to2+b3.astype(int))*b3.astype(int)
    b_9to4 = (b_8to3+b4.astype(int))*b4.astype(int)
    b_10to5 = (b_9to4+b5.astype(int))*b5.astype(int)
    b_11to6 = (b_10to5+b6.astype(int))*b6.astype(int)
    b_12to7 = (b_11to6+b7.astype(int))*b7.astype(int)
    b_13to8 = (b_12to7+b8.astype(int))*b8.astype(int)
    b_14to9 = (b_13to8+b9.astype(int))*b9.astype(int)
    b_15to10 = (b_14to9+b10.astype(int))*b10.astype(int)
    b_16to11 = (b_15to10+b11.astype(int))*b11.astype(int)

    b_corner1 = np.logical_or(np.logical_or(b_1to12 >= 12, b_2to13 >= 12), np.logical_or(b_3to14 >= 12, b_4to15 >= 12))
    b_corner1 = np.logical_or(np.logical_or(b_corner1, b_5to16 >= 12), np.logical_or(b_6to1 >= 12, b_7to2 >= 12))
    b_corner1 = np.logical_or(np.logical_or(b_corner1, b_8to3 >= 12), np.logical_or(b_9to4 >= 12, b_10to5 >= 12))
    b_corner1 = np.logical_or(np.logical_or(b_corner1, b_11to6 >= 12), np.logical_or(b_12to7 >= 12, b_13to8 >= 12))
    b_corner1 = np.logical_or(np.logical_or(b_corner1, b_14to9 >= 12), np.logical_or(b_15to10 >= 12, b_16to11 >= 12))


    # corner type 2
    b_1_and_9 = np.logical_and(v0_b > v1, v0_b > v9)
    b_5_or_13 = np.logical_or(v0_b > v5, v0_b > v13)

    b_5_and_13 = np.logical_and(v0_b > v5, v0_b > v13)  
    b_1_or_9 = np.logical_or(v0_b > v1, v0_b > v9)

    b_fast_test2 = np.logical_or(np.logical_and(b_1_or_9, b_5_and_13), np.logical_and(b_1_and_9, b_5_or_13))

    v0_b_tested2 = v0_b[b_fast_test2]

    b1 = v1[b_fast_test2] < v0_b_tested2
    b2 = v2[b_fast_test2] < v0_b_tested2
    b3 = v3[b_fast_test2] < v0_b_tested2
    b4 = v4[b_fast_test2] < v0_b_tested2
    b5 = v5[b_fast_test2] < v0_b_tested2
    b6 = v6[b_fast_test2] < v0_b_tested2
    b7 = v7[b_fast_test2] < v0_b_tested2
    b8 = v8[b_fast_test2] < v0_b_tested2
    b9 = v9[b_fast_test2] < v0_b_tested2
    b10 = v10[b_fast_test2] < v0_b_tested2
    b11 = v11[b_fast_test2] < v0_b_tested2
    b12 = v12[b_fast_test2] < v0_b_tested2
    b13 = v13[b_fast_test2] < v0_b_tested2
    b14 = v14[b_fast_test2] < v0_b_tested2
    b15 = v15[b_fast_test2] < v0_b_tested2
    b16 = v15[b_fast_test2] < v0_b_tested2

    diff1 = v0_b_tested2 - v1[b_fast_test2]
    diff2 = v0_b_tested2 - v2[b_fast_test2]
    diff3 = v0_b_tested2 - v3[b_fast_test2]
    diff4 = v0_b_tested2 - v4[b_fast_test2]
    diff5 = v0_b_tested2 - v5[b_fast_test2]
    diff6 = v0_b_tested2 - v6[b_fast_test2]
    diff7 = v0_b_tested2 - v7[b_fast_test2]
    diff8 = v0_b_tested2 - v8[b_fast_test2]
    diff9 = v0_b_tested2 - v9[b_fast_test2]
    diff10 = v0_b_tested2 - v10[b_fast_test2]
    diff11 = v0_b_tested2 - v11[b_fast_test2]
    diff12 = v0_b_tested2 - v12[b_fast_test2]
    diff13 = v0_b_tested2 - v13[b_fast_test2]
    diff14 = v0_b_tested2 - v14[b_fast_test2]
    diff15 = v0_b_tested2 - v15[b_fast_test2]
    diff16 = v0_b_tested2 - v16[b_fast_test2]

    c_res2 = diff1*(diff1>0).astype(int) + diff2*(diff2>0).astype(int)+diff3*(diff3>0).astype(int)+diff4*(diff4>0).astype(int) \
            + diff5*(diff5>0).astype(int)+diff6*(diff6>0).astype(int)+diff7*(diff7>0).astype(int)+diff8*(diff8>0).astype(int) \
            + diff9*(diff9>0).astype(int)+diff10*(diff10>0).astype(int)+diff11*(diff11>0).astype(int)+diff12*(diff12>0).astype(int) \
            + diff13*(diff13>0).astype(int)+diff14*(diff14>0).astype(int)+diff15*(diff15>0).astype(int)+diff16*(diff16>0).astype(int) 

    b_1to12 = ((((((((((((b1.astype(int)+b2.astype(int))*b2.astype(int))+b3.astype(int))*b3.astype(int) \
                       +b4.astype(int))*b4.astype(int)+b5.astype(int))*b5.astype(int)+b6.astype(int))*b6.astype(int) \
                    +b7.astype(int))*b7.astype(int)+b8.astype(int))*b8.astype(int)+b9.astype(int))*b9.astype(int) \
                 +b10.astype(int))*b10.astype(int)+b11.astype(int))*b11.astype(int)+b12.astype(int))*b12.astype(int)

    b_2to13 = (b_1to12+b13.astype(int))*b13.astype(int)
    b_3to14 = (b_2to13+b14.astype(int))*b14.astype(int)
    b_4to15 = (b_3to14+b15.astype(int))*b15.astype(int)
    b_5to16 = (b_4to15+b16.astype(int))*b16.astype(int)
    b_6to1 = (b_5to16+b1.astype(int))*b1.astype(int)
    b_7to2 = (b_6to1+b2.astype(int))*b2.astype(int)
    b_8to3 = (b_7to2+b3.astype(int))*b3.astype(int)
    b_9to4 = (b_8to3+b4.astype(int))*b4.astype(int)
    b_10to5 = (b_9to4+b5.astype(int))*b5.astype(int)
    b_11to6 = (b_10to5+b6.astype(int))*b6.astype(int)
    b_12to7 = (b_11to6+b7.astype(int))*b7.astype(int)
    b_13to8 = (b_12to7+b8.astype(int))*b8.astype(int)
    b_14to9 = (b_13to8+b9.astype(int))*b9.astype(int)
    b_15to10 = (b_14to9+b10.astype(int))*b10.astype(int)
    b_16to11 = (b_15to10+b11.astype(int))*b11.astype(int)

    b_corner2 = np.logical_or(np.logical_or(b_1to12 >= 12, b_2to13 >= 12), np.logical_or(b_3to14 >= 12, b_4to15 >= 12))
    b_corner2 = np.logical_or(np.logical_or(b_corner2, b_5to16 >= 12), np.logical_or(b_6to1 >= 12, b_7to2 >= 12))
    b_corner2 = np.logical_or(np.logical_or(b_corner2, b_8to3 >= 12), np.logical_or(b_9to4 >= 12, b_10to5 >= 12))
    b_corner2 = np.logical_or(np.logical_or(b_corner2, b_11to6 >= 12), np.logical_or(b_12to7 >= 12, b_13to8 >= 12))
    b_corner2 = np.logical_or(np.logical_or(b_corner2, b_14to9 >= 12), np.logical_or(b_15to10 >= 12, b_16to11 >= 12))

    #print(c_res1[b_corner1].shape)
    #print(c_res2[b_corner2].shape)

    xy = np.mgrid[4:img.shape[0]-4, 4:img.shape[1]-4]
    xx = xy[1]
    yy = xy[0]

    x_fast1 = xx[b_fast_test1]
    x_corner1 = x_fast1[b_corner1]
    y_fast1 = yy[b_fast_test1]
    y_corner1 = y_fast1[b_corner1]

    x_fast2 = xx[b_fast_test2]
    x_corner2 = x_fast2[b_corner2]
    y_fast2 = yy[b_fast_test2]
    y_corner2 = y_fast2[b_corner2]

    x_corner = np.hstack((x_corner1, x_corner2))
    y_corner = np.hstack((y_corner1, y_corner2))
    c_res = np.hstack((c_res1[b_corner1], c_res2[b_corner2]))

    
    
    # non maximum suppresion 
    result = np.zeros(img.shape[0:2], dtype=int)
    result[y_corner, x_corner] = c_res

    b_non_max = np.amax(np.array([np.pad(result, ((0, 2), (0, 2)), 'constant'), np.pad(result, ((0, 2), (1, 1)), 'constant'), \
              np.pad(result, ((0, 2), (2, 0)), 'constant'), np.pad(result, ((1, 1), (0, 2)), 'constant'), \
              np.pad(result, ((1, 1), (2, 0)), 'constant'), np.pad(result, ((2, 0), (0, 2)), 'constant'), \
              np.pad(result, ((2, 0), (1, 1)), 'constant'), np.pad(result, ((2, 0), (2, 0)), 'constant')]), axis=0) > np.pad(result, ((1, 1), (1, 1)), 'constant')

    b_non_max = b_non_max[1:-1, 1:-1]

    result[b_non_max] = 0

    xy = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    xx = xy[1]
    yy = xy[0]

    b_corner_nms = result > 200
    x_corner = xx[b_corner_nms]
    y_corner = yy[b_corner_nms]
    c_res = result[b_corner_nms]

    return np.vstack((x_corner, y_corner)).T, c_res

def ExtractDescriptors(img, corners, delta = 10):

    # extract descriptors
    img_s = cv2.GaussianBlur(img,(5,5), 1.414)
    img = cv2.GaussianBlur(img,(5,5), 0.787)

    
    v0 = img[corners[:,1], corners[:,0]].astype(int) # pixel values of the corner points
    v0_t = v0 + delta
    v0_b = v0 - delta
    v1 = img[corners[:,1]-1, corners[:,0]-1].astype(int)
    v2 = img[corners[:,1]-1, corners[:,0]].astype(int)
    v3 = img[corners[:,1]-1, corners[:,0]+1].astype(int)
    v4 = img[corners[:,1], corners[:,0]-1].astype(int)
    v5 = img[corners[:,1], corners[:,0]+1].astype(int)
    v6 = img[corners[:,1]+1, corners[:,0]-1].astype(int)
    v7 = img[corners[:,1]+1, corners[:,0]].astype(int)
    v8 = img[corners[:,1]+1, corners[:,0]+1].astype(int)

    v0_s = img_s[corners[:,1], corners[:,0]].astype(int)
    v9 = img_s[corners[:,1]-2, corners[:,0]-2].astype(int)
    v10 = img_s[corners[:,1]-2, corners[:,0]].astype(int)
    v11 = img_s[corners[:,1]-2, corners[:,0]+2].astype(int)
    v12 = img_s[corners[:,1], corners[:,0]-2].astype(int)
    v13 = img_s[corners[:,1], corners[:,0]+2].astype(int)
    v14 = img_s[corners[:,1]+2, corners[:,0]-2].astype(int)
    v15 = img_s[corners[:,1]+2, corners[:,0]].astype(int)
    v16 = img_s[corners[:,1]+2, corners[:,0]+2].astype(int)

    v17 = img_s[corners[:,1]-3, corners[:,0]-3].astype(int)
    v18 = img_s[corners[:,1]-3, corners[:,0]].astype(int)
    v19 = img_s[corners[:,1]-3, corners[:,0]+3].astype(int)
    v20 = img_s[corners[:,1], corners[:,0]-3].astype(int)
    v21 = img_s[corners[:,1], corners[:,0]+3].astype(int)
    v22 = img_s[corners[:,1]+3, corners[:,0]-3].astype(int)
    v23 = img_s[corners[:,1]+3, corners[:,0]].astype(int)
    v24 = img_s[corners[:,1]+3, corners[:,0]+3].astype(int)

    b1 = (v1 > v0_b).astype(np.int8) + (v1 > v0_t).astype(np.int8)
    b2 = (v2 > v0_b).astype(np.int8) + (v2 > v0_t).astype(np.int8)
    b3 = (v3 > v0_b).astype(np.int8) + (v3 > v0_t).astype(np.int8)
    b4 = (v4 > v0_b).astype(np.int8) + (v4 > v0_t).astype(np.int8)
    b5 = (v5 > v0_b).astype(np.int8) + (v5 > v0_t).astype(np.int8)
    b6 = (v6 > v0_b).astype(np.int8) + (v6 > v0_t).astype(np.int8)
    b7 = (v7 > v0_b).astype(np.int8) + (v7 > v0_t).astype(np.int8)
    b8 = (v8 > v0_b).astype(np.int8) + (v8 > v0_t).astype(np.int8)
    b9 = (v9 > v0_s - delta).astype(np.int8) + (v9 > v0_s + delta).astype(np.int8)
    b10 = (v10 > v0_s - delta).astype(np.int8) + (v10 > v0_s + delta).astype(np.int8)
    b11 = (v11 > v0_s - delta).astype(np.int8) + (v11 > v0_s + delta).astype(np.int8)
    b12 = (v12 > v0_s - delta).astype(np.int8) + (v12 > v0_s + delta).astype(np.int8)
    b13 = (v13 > v0_s - delta).astype(np.int8) + (v13 > v0_s + delta).astype(np.int8)
    b14 = (v14 > v0_s - delta).astype(np.int8) + (v14 > v0_s + delta).astype(np.int8)
    b15 = (v15 > v0_s - delta).astype(np.int8) + (v15 > v0_s + delta).astype(np.int8)
    b16 = (v16 > v0_s - delta).astype(np.int8) + (v16 > v0_s + delta).astype(np.int8)
    b17 = (v17 > v0_s - delta).astype(np.int8) + (v17 > v0_s + delta).astype(np.int8)
    b18 = (v18 > v0_s - delta).astype(np.int8) + (v18 > v0_s + delta).astype(np.int8)
    b19 = (v19 > v0_s - delta).astype(np.int8) + (v19 > v0_s + delta).astype(np.int8)
    b20 = (v20 > v0_s - delta).astype(np.int8) + (v20 > v0_s + delta).astype(np.int8)
    b21 = (v21 > v0_s - delta).astype(np.int8) + (v21 > v0_s + delta).astype(np.int8)
    b22 = (v22 > v0_s - delta).astype(np.int8) + (v22 > v0_s + delta).astype(np.int8)
    b23 = (v23 > v0_s - delta).astype(np.int8) + (v23 > v0_s + delta).astype(np.int8)
    b24 = (v24 > v0_s - delta).astype(np.int8) + (v24 > v0_s + delta).astype(np.int8)

    desc = np.vstack((b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24)).T

    return desc

def FindHomography(pts1, pts2):
    
    n = pts1.shape[0]
    A = np.zeros((0, 9), dtype=float)
    for i in range(0, n):
        
        x = pts1[i, 0]
        y = pts1[i, 1]
        
        xp = pts2[i, 0]
        yp = pts2[i, 1]
        
        a1 = np.array([[-x, -y, -1, 0.0, 0.0, 0.0, xp*x, xp*y, xp], [0, 0, 0, -x, -y, -1, yp*x, yp*y, yp]])
        A = np.vstack((A, a1))
    
    #print(A)
    u, d, vh = np.linalg.svd(A)
    
    #print(v)
    H = vh[-1]
    H = H/H[-1]
    
    return H

def DrawCorrespondences(corr, img1, corner1, img2, corner2):

    # corr: correspondences, a list of pairs (i, j) 
    # the 1st column is indices of features(corners) in img1, 2nd column is indices in img2
    # corner1: a list of (x, y)s 

    keypoints1 = []
    for i in range(0, corner1.shape[0]):
        float(corner1[i, 0])
        float(corner1[i, 1])
        keypoints1.append(cv2.KeyPoint(corner1[i, 0], corner1[i, 1], 1))

    keypoints2 = []
    for i in range(0, corner2.shape[0]):
        float(corner2[i, 0])
        float(corner2[i, 1])
        keypoints2.append(cv2.KeyPoint(corner2[i, 0], corner2[i, 1], 1))

    matches_draw = []
    for i in range(0, corr.shape[0]):
        matches_draw.append(cv2.DMatch(corr[i,0], corr[i,1], 1))

    res_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches_draw, outImg=None)

    return res_img
        