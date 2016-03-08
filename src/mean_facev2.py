__author__ = 'Sukrit'
import numpy as np
from PIL import Image as IM
import pylab


def PCA(X):
    # Principal Component Analysis
    # input: X, face matrix with training data as flattened arrays in rows
    # return: projection matrix having most variance first

    # get dimensions
    num_data, dim = X.shape

    # get mean face
    mean_X = X.mean(axis=0)
    for i in range(num_data):
        X[i] -= mean_X

    COV = np.dot(X, X.T)  # covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(COV)  # eigenvalues and eigenvectors
    tmp = np.dot(X.T, eigenvectors).T

    # sorting eigenvalues in descending order
    index_arr = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[index_arr]

    # sorting eigenvectors in ascending order of their eigenvalues
    eigenvectors = eigenvectors[:, index_arr]

    V = tmp[::-1]  # reverse since last eigenvectors are the ones we want


    # return the projection matrix, the variance and the mean
    return V, mean_X, eigenvalues, eigenvectors


def mode_normalize(mode, high=255, low=0):
    MAX = np.max(mode)
    MIN = np.min(mode)

    # print "Mode MIN : " , MIN
    # print "Mode MAX : " , MAX

    mode_255 = mode

    mode_255 -= float(MIN)
    mode_255 /= float(MAX - MIN)

    mode_255 *= (high - low)
    mode_255 += low

    return mode_255


i = 0
m = 10 # number of images used to calculate mean

faces = []

for i in range (0,m,1) : # m images
    faces.append( IM.open("../data/att_faces/s1/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

for i in range (0,m,1) : # m images
    faces.append( IM.open("../data/att_faces/s2/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image
'''
for i in range(0, m, 1):  # GW Bush
    faces.append(IM.open("../data/gwb_cropped/" + str(i + 1) + ".jpg"))



for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s2/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s3/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s4/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s5/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s6/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s7/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s8/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image
'''

w, h = faces[0].size  # get size (WxH) of first image in list, assuming all are same size!
# print w,h
N = len(faces)  # number of faces

print "------------------------------------------------------------"
print "Number of face images : ", N

'''
PCA
'''

X = np.array([np.array(faces[i]).flatten() for i in range(N)],
             'f')  # converting NxHxW 3D array to Nx(HxW) 2D array i.e N rows for each face, of HxW columns

V, mean_face, evalues, evectors = PCA(X)
mean_face = mean_face.reshape(h,w).copy()
mode1 = V[0].reshape(h, w)  # using first principal component
mode2 = V[1].reshape(h, w)
mode3 = V[2].reshape(h, w)

mean_255 = mode_normalize(mean_face)
mode_255_1 = mode_normalize(mode1)
mode_255_2 = mode_normalize(mode2)
mode_255_3 = mode_normalize(mode3)

print "------------------------------------------------------------"
print "Number of Eigenvalues: " , len(evalues)
print "Eigenvalues: \n", evalues

print "------------------------------------------------------------"
print "Eigenvectors: \n", evectors

print "------------------------------------------------------------"
print "Mean Face: \n" , mean_255
pylab.figure()
pylab.gray()
pylab.imshow(mean_255)


print "------------------------------------------------------------"
print "Mode 1: \n" , mode_255_1
pylab.figure()
pylab.gray()
pylab.imshow(mode_255_1)

print "------------------------------------------------------------"
print "Mode 2: \n" , mode_255_2
pylab.figure()
pylab.gray()
pylab.imshow(mode_normalize(faces[10]-mean_face))

print "------------------------------------------------------------"
print "Mode 3: \n" , mode_255_3
pylab.figure()
pylab.gray()
pylab.imshow(mode_normalize(faces[0]-mean_face))
print "------------------------------------------------------------"

pylab.show()

'''
# Round values in array and cast as 8-bit integer
arr=np.array(np.round(diff_array[0]),dtype=np.uint8)
#arr=np.array(np.round(mean_array),dtype=np.uint8)

# Generate, save and preview final image
out=IM.fromarray(arr,mode="L")
#out.save("../output/diff_face_s1_minus_s1234.png")
out.show()
'''
