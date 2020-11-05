from geometry import *
quad = Quadric(np.array([-1,-1,0]), np.array([0,0,1]), 1)
# v = np.array([-1., -1., 0.67542021])
# w = np.array([-1., -1., 100.])
# print(quad.relative_position(v))
# print(quad.relative_position(w))
# print(quad(v))
# print(quad(w))
# print(quad.intersect_with_ray(v, w-v))
# print(quad.intersect_with_segment(v, w))



# v = np.array([-1., -1., 0.42779099])
# w = np.array([-1., -1., 20.23843379])
v = np.array([ 1.00000000000001        , -1.        ,  1.94163358])
w = np.array([ 1.        , -1.        ,  0.40137248])
print(quad.intersect_with_segment(v, w))
