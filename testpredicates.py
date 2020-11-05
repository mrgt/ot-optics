import numpy as np
import pypower


# test with q(x,y,z) = z - (x^2 + y^2) 
qxx = -1.0
qyy = -1.0
qzz = 0.0
bx = 0.0
by = 0.0
bz = 1.0
c = 0.0


# x = np.random.rand()
# y = np.random.rand()
# z = x**2 + y**2
# print(pypower.point_above_quadric(qxx,qyy,qzz,bx,by,bz,c,x,y,z))

x0 = np.random.rand()
y0 = np.random.rand()
z0 = x0**2 + y0**2 - .1
x1 = np.random.rand()
y1 = np.random.rand()
z1 = x1**2 + y1**2 + .1
print(pypower.point_above_quadric(qxx,qyy,qzz,bx,by,bz,c,
                                  False,0,0,0,0,
                                  x0,y0,z0))
print(pypower.point_above_quadric(qxx,qyy,qzz,bx,by,bz,c,
                                  False,0,0,0,0,
                                  x1,y1,z1))
print(pypower.intersect_segment_with_quadric(qxx,qyy,qzz,bx,by,bz,c,
                                             False,0,0,0,0,
                                             x0,y0,z0,x1,y1,z1))


x0 = np.random.rand()
y0 = np.random.rand()
z0 = x0**2 + y0**2 + .1
x1 = -x0
y1 = -y0
z1 = x0**2 + y0**2 + .1
print(pypower.intersect_segment_with_quadric(qxx,qyy,qzz,bx,by,bz,c,
                                             False,0,0,0,0,
                                             x0,y0,z0,x1,y1,z1))
