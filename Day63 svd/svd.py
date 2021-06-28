# -*- coding: utf-8 -*-
'''
author@cclplus
date:2019/11/3
'''
import cv2
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
#Change to u8 type
def restore1(u, sigma, v, k):
    m = len(u)
    n = len(v)
    a = np.zeros((m, n))
    a = np.dot(u[:, :k], np.diag(sigma[:k])).dot(v[:k, :])
    a[a < 0] = 0
    a[a > 255] = 255
    return np.rint(a).astype('uint8')
def SVD(frame,K=10):
    a = np.array(frame)
    #Because it is a color image, so 3 channels. The innermost array of a is three numbers, each representing RGB, used to represent a pixel
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])
    R = restore1(u_r, sigma_r, v_r, K)
    G = restore1(u_g, sigma_g, v_g, K)
    B = restore1(u_b, sigma_b, v_b, K)
    I = np.stack((R, G, B), axis = 2)
    return I
      

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    frame = cv2.imread("wbb.jpg")
    I = SVD(frame,400)
    plt.imshow(I)
    cv2.imwrite("out400.jpg",I)
