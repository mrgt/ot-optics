import numpy as np
import pypower


# this class represents an implicit quadric of the form <Qx|x> + <b|x> + c = 0
class Quadric:
    def __init__(self,Q,b,c, truncating_plane=False, p=np.array([0,0,0]), d=0):
        assert(Q.ndim == 1 and len(Q) == 3)
        self.Q = Q
        self.b = b
        self.c = c
        self.truncating_plane = truncating_plane
        self.p = p
        self.d = d
        
        
    def __call__(self,x):
        return np.dot(x,self.Q*x) + np.dot(self.b,x) + self.c

    def relative_position(self,x):
        return pypower.point_above_quadric(self.Q[0], self.Q[1], self.Q[2],
                                           self.b[0], self.b[1], self.b[2],
                                           self.c,
                                           self.truncating_plane,
                                           self.p[0], self.p[1], self.p[2],
                                           self.d,
                                           x[0], x[1], x[2],)
    
    # compute the intersection of the quadric with the ray a+tv, t a real number.
    # returns the values of t corresponding to the intersection(s)
    def intersect_with_ray(self, a, v):
        S = pypower.intersect_ray_with_quadric(self.Q[0], self.Q[1], self.Q[2],
                                               self.b[0], self.b[1], self.b[2],
                                               self.c,
                                               self.truncating_plane,
                                               self.p[0], self.p[1], self.p[2],
                                               self.d,
                                               a[0], a[1], a[2],
                                               v[0], v[1], v[2])
        return np.array(S)
            
    
    # computes the intersection of the quadric with the segment [p,q], i.e.
    # points of the form (1-t) p + t q with t in [0,1]
    # returns the values of t corresponding to the intersection(s)
    def intersect_with_segment(self,p, q):
        S = pypower.intersect_segment_with_quadric(self.Q[0], self.Q[1], self.Q[2],
                                                   self.b[0], self.b[1], self.b[2],
                                                   self.c,
                                                   self.truncating_plane,
                                                   self.p[0], self.p[1], self.p[2],
                                                   self.d,
                                                   p[0], p[1], p[2],
                                                   q[0], q[1], q[2])
        return np.array(S)
    
    # find the first intersection point of the quadric with the ray a+tv, t>=0 
    # assumption: there exists an intersection
    def compute_first_intersection(self, a, v):
        S = self.intersect_with_ray(a, v)
        S = S[S>=-1e-5]
        if len(S) < 1:
            if abs(self(a)) <= 1e-5:
                return a
            print("len(S)<1: {}".format(self.intersect_with_ray(a, v)))
            print("quad(a): {}".format(self(a)))
        t = np.min(S)
        return a + t * v
    
    # returns a curve corresponding to the intersection of the triangle [p,q,bary] with the quadric
    # assumption: quad(p) = quad(q) = 0, quad(bary)>0
    def trace_curve(self, p, q, bary, eta):
        if np.linalg.norm(p-q) <= eta:
            return [p,q]
        m = (p+q)/2
        b = self.compute_first_intersection(m, m - bary)
        # t = self.intersect_with_segment(bary, m)
        # if len(t) != 1:
        #     print("qq=%g"%self(q))
        #     print("qp=%g"%self(p))
        #     print("qm=%g"%self(m))
        #     print("qbary=%g"%self(bary))
        #     print(t)
        # assert(len(t) == 1)
        # b = (1 - t[0]) * bary + t[0] * m
        return (self.trace_curve(p, b, bary, eta) +
                self.trace_curve(b, q, bary, eta))

def find_point_above_quadric_in_polygon(quad, P):
    nv = len(P)
    bary = np.mean(P,0)
    
    # check if one vertex of the polygon is above
    for p in P:
        if quad.relative_position(p) <= 0:
            continue
        t = 1e-5
        for i in range(5):
            above = p + t*(bary - p)
            if (quad.relative_position(above) > 0):
                return above
                break
            t /= 2

    # check if one segment intersects the quadric
    for v in range(nv):
        w = (v+1)%nv
        T = quad.intersect_with_segment(P[v], P[w])
        if len(T) == 0:
            continue
        if len(T) == 1:
            print("len(T) = 1", quad(P[v]), quad(P[w]))
            continue
        assert(len(T) == 2)
        q1 = P[v] + T[0]*(P[w] - P[v])
        q2 = P[v] + T[1]*(P[w] - P[v])
        above = (q1+q2)/2 + 1e-7*(np.mean(P,0) - P[v,:])
        if (quad.relative_position(above) <= 0):
            continue # segment is barely above....
        assert(quad.relative_position(above)>0)
        return above
    
    # otherwise, the polygon does not intersect the quadric
    # BEWARE: we are neglecting the possibility that P intersects quad in its interior only...
    #print("found no point above")
    return None

# intersects the triangle [v,w,above] with the quadric
# assumption: quad(above) > 0
def intersect_triangle_with_quadric(quad, v, w, above, eta):
    signv = quad.relative_position(v)
    signw = quad.relative_position(w)
    if (signv == 1) and (signw == 1): # everything is above: nothing to do
        return []
    elif (signv == 1) and (signw == -1): 
        T = quad.intersect_with_segment(v, w)
        assert len(T) == 1
        p = v + T[0]*(w - v)
        q = quad.compute_first_intersection(w, above-w)
        return [quad.trace_curve(p,q,above,eta)]
    elif signv == -1 and signw == 1: 
        p = quad.compute_first_intersection(v, above-v)
        T = quad.intersect_with_segment(v, w)
        assert(len(T) == 1)
        q = v + T[0]*(w - v)
        return [quad.trace_curve(p,q,above,eta)]
    elif signv == -1 and signw == -1: 
        pv = quad.compute_first_intersection(v, above-v)
        pw = quad.compute_first_intersection(w, above-w)
        T = quad.intersect_with_segment(v, w)
        if len(T) == 0:
            # all the segment is below the quadric
            return [quad.trace_curve(pv,pw, above, eta)]
        
        assert len(T) == 2, "T={}".format(T)
        q1 = v + T[0]*(w - v)
        q2 = v + T[1]*(w - v)
        return [quad.trace_curve(pv,q1,above,eta),
                quad.trace_curve(q2,pw,above,eta)]

# returns the intersection between the polygon P with the quadric 
# assumption: P is flat, i.e. contained in a plane
# returns a list of curves (list of numpy arrays) describing the intersection
def intersect_polygon_with_quadric(P,quad, eta):
    nv = len(P)
    
    # find point above quadric in P; if there is no such point, intersection is empty
    above = find_point_above_quadric_in_polygon(quad,P)
    if above is None:
        return []
    
    # compute the intersection between quad and the triangles of the form [v,w,above],
    # where v,w are vertices of P
    curves = []
    for v in range(nv):
        w = (v+1)%nv
        curves += intersect_triangle_with_quadric(quad, P[v], P[w], above, eta)
    return curves        


# computes the intersection of a cell = cell[i] of a Power diagram with the quadric
# returns a dictionnary mapping a neighbor j to a list of curves Pow_i \cap Pow_j \cap quad, 
# as returned by intersect_polygon_with_quadric
def intersect_cell_with_quadric(X, cell, quad, eta):
    curves = {}
    for j,P in cell.items(): # iterate over all facets
        P = np.array(P)
        curves[j] = intersect_polygon_with_quadric(P,quad, eta)
    return curves


def volume_of_cell(cell):
    vol = 0
    for j,P in cell.items(): # iterate over all facets
        P = np.array(P)
        for k in range(1,P.shape[0]-1):
            vol += np.linalg.det(np.array([P[0], P[k], P[k+1]]))/6
    return vol
