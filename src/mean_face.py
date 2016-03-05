import numpy as np
from PIL import Image as IM
i = 1

m = 10 # number of images used to calculate mean

faces = []

for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s1/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s2/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s3/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

for i in range (0,m,1) : # m images
	faces.append( IM.open("../data/att_faces/s4/" + str(i+1) + ".pgm") ) # 92 x 112 PGM image

	

w,h = faces[0].size # get size of first image in list, assuming all are same size!
N = len(faces) # number of faces of 1 person

mean_array=np.zeros((h,w),np.float) # filling empty numpy array 92x112 with zeros

for face in faces :
    imarr=np.array(face,dtype=np.float) # converting each 'face image' into numpy array
    mean_array += imarr #adding up all the faces

mean_array /= N #calculating mean face

diff_array = [m] # diff_array = face - mean_array
for i in range(0,m,1) :
    diff_array.append(np.zeros((h,w),np.float)) # filling with zeros

for i in range(0,m,1) :
    diff_array[i] -= mean_array


# Round values in array and cast as 8-bit integer
arr=np.array(np.round(diff_array[0]),dtype=np.uint8)
#arr=np.array(np.round(mean_array),dtype=np.uint8)

# Generate, save and preview final image
out=IM.fromarray(arr,mode="L")
out.save("../output/diff_face_s1_minus_s1234.png")
out.show()

