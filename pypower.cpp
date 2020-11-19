#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>
#include <CGAL/Regular_triangulation_vertex_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <cassert>
#include <vector>

#include <cstdlib>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT                                               Weight;
typedef K::Point_3                                          Point;
typedef K::Vector_3                                         Vector;
typedef K::Weighted_point_3                                 Weighted_point;
typedef CGAL::Regular_triangulation_vertex_base_3<K>        Vb0;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, K, Vb0> Vb;
typedef CGAL::Regular_triangulation_cell_base_3<K>          Cb;
typedef CGAL::Triangulation_data_structure_3<Vb,Cb>         Tds;
typedef CGAL::Regular_triangulation_3<K, Tds>               Rt;
typedef Rt::Vertex_handle Vertex_handle;

template <class T, class OutputIterator>
void
tessellate_facet(const T &dt,
		 typename T::Vertex_handle v,
		 typename T::Vertex_handle w,
		 OutputIterator it)
{
   typename T::Cell_handle cell;
   int i,j;
   if (!dt.is_edge(v,w,cell,i,j))
     return;
   typename T::Cell_circulator c = dt.incident_cells (cell,i,j), done(c);    
   do
     {
       assert (!dt.is_infinite(c));
       auto p = dt.dual(c);
       *it++ = Eigen::Vector3d(p.x(), p.y(), p.z());
     }
   while (++c != done);
}

template <class V>
double squared_norm(const V &v)
{
  return v.x()*v.x() + v.y()*v.y() + v.z()*v.z();
}

double rand01()
{
  return  ((double) rand() / (RAND_MAX));
}

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector< std::map<int, std::vector<Eigen::Vector3d> > >
power_diagram(const Eigen::MatrixXd &X, const Eigen::VectorXd &w,
	      const Eigen::MatrixXd &As, const Eigen::VectorXd &bs)
{
  std::vector< std::pair<Weighted_point, int> > weighted_points;
  int N = X.rows();
  assert(w.size() == N);
  assert(X.cols() == 3);

  for(size_t i = 0; i < N; ++i)
    {
      Point p(X(i,0), X(i,1), X(i,2));
      double ww = w(i);
      weighted_points.push_back(std::make_pair(Weighted_point(p, w(i)),i));
    }
  Rt rt( weighted_points.begin(), weighted_points.end() );

  // domain is defined by A x <= b
  int Nc = As.rows();
  assert(bs.size() == Nc);
  assert(As.cols() == 3);

  
  std::vector<Vector> A;
  std::vector<double> b;
  for (size_t i = 0; i < Nc; ++i)
    {
      A.push_back(Vector(As(i,0),As(i,1),As(i,2)));
      b.push_back(bs(i));
    }

  std::vector< std::map<int, std::vector<Eigen::Vector3d> > > res(N);
  //double v = 0;
  for(auto it = rt.finite_vertices_begin(); it != rt.finite_vertices_end(); ++it)
    {
      // we build a "local" regular triangulation with the neighbors of it, and additional
      // neighbors to clip the cell of it to the domain.
      std::vector< std::pair<Weighted_point, int> > points_loc;
      points_loc.push_back(std::make_pair(it->point(), it->info()));

      // insert neighbors
      std::list<Rt::Vertex_handle> vertices;
      rt.incident_vertices(it, std::back_inserter(vertices));
      for (auto jt = vertices.begin(); jt != vertices.end(); ++jt)
	{
	  if (rt.is_infinite(*jt))
	    continue;
	  points_loc.push_back(std::make_pair((*jt)->point(), (*jt)->info()));
	}

      // insert point corresponding to domain boundary
      Weighted_point p = it->point();

      // we want to write <a_i|x> <= b_i
      // under the form |x-p|^2 - w_p <= |x - z_i|^2 - c_i
      // <=> |x|^2 + |p|^2 - 2<x|p> - w_p <= |x|^2 + |z_i|^2 - 2 <x|z_i> - c_i
      // <=> <x|2(z_i-p)> <=  |z_i|^2 - |p|^2 + w_p - c_i
      // A solution is given by 2(z_i-p) = a_i  (=> z_i = p + a_i/2)
      // b_i = |z_i|^2 - |p|^2 + w_p - c_i
      // c_i = |z_i|^2 - |p|^2 + w_p - b_i
      for  (size_t i = 0; i < A.size(); ++i)
	{
	  Point z = p.point() + A[i]/2;
	  double c =  squared_norm(z) - squared_norm(p) +  p.weight() - b[i];
	  points_loc.push_back(std::make_pair(Weighted_point(z,c), -(i+1)));
	}

      // we find the point corresponding to it in the local triangulation
      Rt rt_loc(points_loc.begin(), points_loc.end());
      Vertex_handle it_loc = 0;
      for (auto jt = rt_loc.finite_vertices_begin(); jt != rt_loc.finite_vertices_end(); ++jt)
	{
	  if (jt->info() == it->info())
	    {
	      it_loc = jt;
	      break;
	    }
	}
      if (it_loc == 0)
	continue;
      // ... and we tessellate the facets of the power cell of p0
      std::list<Rt::Vertex_handle> vertices_loc;
      rt_loc.incident_vertices(it_loc, std::back_inserter(vertices_loc));

      //double vloc = 0;
      for (auto jt = vertices_loc.begin(); jt != vertices_loc.end(); ++jt)
	{
	  std::vector<Eigen::Vector3d> points;
	  tessellate_facet(rt_loc, it_loc, *jt, std::back_inserter(points));
	  // for (size_t i = 1; i < points.size(); ++i)
	  //   {
	  //     vloc += CGAL::volume(Point(0,0,0), points[0], points[i], points[(i+1)%points.size()]);
	  //   }
	  res[it->info()][(*jt)->info()] = points;
	}
      //std::cerr << "vloc = " << vloc << "\n";
      //v += vloc;
    }
  //std::cerr << v << "\n";
  return res;
}


#include <CGAL/Interval_nt.h>
#include <CGAL/Gmpq.h>
#include <CGAL/Sqrt_extension.h>
#include <CGAL/Root_of_traits.h>

typedef double FT;
typedef CGAL::Interval_nt<true> IT;
typedef CGAL::Gmpq ET;


// returns -1 if 
// q(x,y,z) = qxx x^2 + qyy y^2 + qzz z^2 + bx x + by y + bz z + c < 0
// or  h(x,y,z) = px x + py y + pz z + d < 0
// 0, if one of them equals 0
// and 1 if both are positive
template <class NT>
int
point_above_quadric(NT qxx, NT qyy, NT qzz, // diagonal
		    NT bx, NT by, NT bz,    // linear term
		    NT c,                   // constant
		    bool truncating_plane,  // is the quadric truncated (e.g. half-hyperboloid)
		    NT px, NT py, NT pz,    // plane normal
		    NT d,                   // distance to origin
		    NT x, NT y, NT z)       // point
{
  if (truncating_plane)
    {
      auto signq = CGAL::sign(qxx*x*x + qyy*y*y + qzz*z*z + bx*x + by*y + bz*z + c);
      auto signh = CGAL::sign(px*x + py*y + pz*z + d);
      if (signq == CGAL::POSITIVE)
	return signh;
      if (signq == CGAL::NEGATIVE)
	return CGAL::NEGATIVE;
      if (signh == CGAL::POSITIVE || signh == CGAL::ZERO)
	return CGAL::ZERO;
      return CGAL::NEGATIVE;
    }
  else
    {
      // std::cerr << "paq: "
      // 		<< CGAL::to_double(qxx*x*x + qyy*y*y + qzz*z*z + bx*x + by*y + bz*z + c) << "\n";
      return CGAL::sign(qxx*x*x + qyy*y*y + qzz*z*z + bx*x + by*y + bz*z + c);
    }
}

int
point_above_quadric_exact(FT qxx, FT qyy, FT qzz,
			  FT bx, FT by, FT bz,
			  FT c,
			  bool truncating_plane,  // is the quadric truncated (e.g. half-hyperboloid)
			  FT px, FT py, FT pz,    // plane normal
			  FT d,                   // distance to origin
			  FT x, FT y, FT z)
{
  try
    {
      CGAL::Interval_nt_advanced::Protector P;
      return point_above_quadric(IT(qxx), IT(qyy), IT(qzz),
				 IT(bx), IT(by), IT(bz),
				 IT(c),
				 truncating_plane,
				 IT(px), IT(py), IT(pz),
				 IT(d),
				 IT(x), IT(y), IT(z));
    }
  catch (...)
    {
      std::cerr << "exact computation";
      return point_above_quadric(ET(qxx), ET(qyy), ET(qzz),
				 ET(bx), ET(by), ET(bz),
				 ET(c),
				 truncating_plane,
				 ET(px), ET(py), ET(pz),
				 ET(d),
				 ET(x), ET(y), ET(z));
    }
}

#define DEBUG_SHOW(x) std::cerr << #x << " = " << x << "\n"

template <class NT>
class Sqrt
{
public:
  typedef typename CGAL::Sqrt_extension<NT,NT,
					CGAL::Boolean_tag<true>,CGAL::Boolean_tag<true>> Extension;
  static Extension make_sqrt(NT r)
  {
    return -CGAL::make_root_of_2(0,1,r);
  }
};

template <>
class Sqrt<IT>
{
public:
  typedef IT Extension;
  static Extension make_sqrt(IT r)
  {
    return CGAL::sqrt(r);
  }
};

template <class NT, class RT>
bool _is_above_plane(NT px, NT py, NT pz,
		     NT d,
		     NT x0, NT y0, NT z0,
		     RT r,
		     NT vx, NT vy, NT vz)
{
  auto x = x0 + r*vx, y = y0 + r*vy, z = z0 + r*vz;
  return px*x+py*y+pz*z+d >= 0;
}

// intersects the ray {[x0,y0,z0] + t[vx,vy,vz]}
// with {(x,y,z) | q(x,y,z) <= 0 and h(x,y,z) <= 0}
// where  q(x,y,z) = qxx x^2 + qyy y^2 + qzz z^2 + bx x + by y + bz z + c >= 0
// and h(x,y,z) = px x + py y + pz z + d >= 0
template <class NT>
std::vector<typename Sqrt<NT>::Extension>
intersect_ray_with_quadric (NT qxx, NT qyy, NT qzz, // diagonal
			    NT bx, NT by, NT bz,    // linear term
			    NT c,                   // constant
			    bool truncating_plane, 
			    NT px, NT py, NT pz,    // plane normal
			    NT d,                   // distance to origin
			    NT x0, NT y0, NT z0,
			    NT vx, NT vy, NT vz) 
{
  typedef typename Sqrt<NT>::Extension SNT;
  
  // quad(x + tv) = <q(x + tv)|x+tv> + <b|x+tv> + c
  //  =  <qx|x> + t^2 <qv|v> + 2t<qx|v> + <b|x> + t<b|v> + c
  //  = A t^2 + B t + C
  // A = <qv|v>, B = 2<qx|v> + <b|v>, C = <b|x> + c + <qx|x>
  NT A = qxx*vx*vx + qyy*vy*vy + qzz*vz*vz;
  NT B = NT(2)*(x0*qxx*vx + y0*qyy*vy + z0*qzz*vz) + bx*vx+by*vy+bz*vz;
  NT C = c + bx*x0 + by*y0 + bz*z0 + qxx*x0*x0 + qyy*y0*y0 + qzz*z0*z0;

  std::vector<SNT> res;
  //DEBUG_SHOW(A);
  if(CGAL::abs(A) > 1e-7) // ugly
    {      
      NT Delta = B*B - NT(4)*A*C;
      //std::cerr << "non-linear case" << "\n";
      //DEBUG_SHOW(Delta);
      if (Delta >= 0)
	{
	  auto sqrtDelta = Sqrt<NT>::make_sqrt(Delta);
	  auto r0  = (-B-sqrtDelta)/(2*A);
	  auto r1 = (-B+sqrtDelta)/(2*A);
	  if (A < 0)
	    std::swap(r0,r1);
	  if (truncating_plane == false || _is_above_plane(px,py,pz,d,x0,y0,z0,r0,vx,vy,vz))
	    res.push_back(r0);
	  if (truncating_plane == false || _is_above_plane(px,py,pz,d,x0,y0,z0,r1,vx,vy,vz))
	    res.push_back(r1);
	}

    }
  else   //  linear case
    {
      //std::cerr << "linear case\n";
      // DEBUG_SHOW(B);
      if (B > 0 || B < 0)
	{
	  auto r = -C/B;
	  if (truncating_plane == false || _is_above_plane(px,py,pz,d,x0,y0,z0,r,vx,vy,vz))
	    res.push_back(r);
	}
    }
  return res;
}

// intersects the segment [x0,y0,z0],[x1,y1,z1]
// with {(x,y,z) | q(x,y,z) <= 0 and h(x,y,z) <= 0}
// where  q(x,y,z) = qxx x^2 + qyy y^2 + qzz z^2 + bx x + by y + bz z + c >= 0
// and h(x,y,z) = px x + py y + pz z + d >= 0
template <class NT>
std::vector<typename Sqrt<NT>::Extension>
intersect_segment_with_quadric (NT qxx, NT qyy, NT qzz, // diagonal
				NT bx, NT by, NT bz,    // linear term
				NT c,                   // constant
				bool truncating_plane, 
				NT px, NT py, NT pz,    // plane normal
				NT d,                   // distance to origin
				NT x0, NT y0, NT z0,
				NT x1, NT y1, NT z1)    // point a 
{
  typedef typename Sqrt<NT>::Extension SNT;
    
  NT vx = x1 - x0, vy = y1 - y0, vz = z1 - z0;
  auto res = intersect_ray_with_quadric(qxx, qyy, qzz, bx, by, bz, c,
					truncating_plane, px, py, pz, d,
					x0, y0, z0, vx, vy, vz);
  std::vector<SNT> res2;
  for (size_t i = 0; i < res.size(); ++i)
    {
      if (res[i] > 1 || res[i] < 0)
	continue;
      res2.push_back(res[i]);
    }
  return res2;
}

std::vector<FT>
intersect_segment_with_quadric_exact (FT qxx, FT qyy, FT qzz, // diagonal
				      FT bx, FT by, FT bz,    // linear term
				      FT c,                   // constant
				      bool truncating_plane, 
				      FT px, FT py, FT pz,    // plane normal
				      FT d,                   // distance to origin
				      FT x0, FT y0, FT z0,
				      FT x1, FT y1, FT z1)    // point a 
{
  try
    {
      CGAL::Interval_nt_advanced::Protector P;
      auto res = intersect_segment_with_quadric(IT(qxx), IT(qyy), IT(qzz),
  						IT(bx), IT(by), IT(bz),
  						IT(c),
  						truncating_plane,
  						IT(px), IT(py), IT(pz),
  						IT(d),
  						IT(x0), IT(y0), IT(z0),
  						IT(x1), IT(y1), IT(z1));
      std::vector<FT> ft_res;
      for (size_t i = 0; i < res.size(); ++i)
  	ft_res.push_back(CGAL::to_double(res[i]));
      return ft_res;
    }
  catch (...)
    {
      std::cerr << "exact intersection";
      auto res = intersect_segment_with_quadric(ET(qxx), ET(qyy), ET(qzz),
						ET(bx), ET(by), ET(bz),
						ET(c),
						truncating_plane,
						ET(px), ET(py), ET(pz),
						ET(d),
						ET(x0), ET(y0), ET(z0),
						ET(x1), ET(y1), ET(z1));
      std::vector<FT> ft_res;
      for (size_t i = 0; i < res.size(); ++i)
	ft_res.push_back(CGAL::to_double(res[i]));
      return ft_res;
    }
}

std::vector<FT>
intersect_ray_with_quadric_exact (FT qxx, FT qyy, FT qzz, // diagonal
				  FT bx, FT by, FT bz,    // linear term
				  FT c,                   // constant
				  bool truncating_plane, 
				  FT px, FT py, FT pz,    // plane normal
				  FT d,                   // distance to origin
				  FT x0, FT y0, FT z0,
				  FT vx, FT vy, FT vz) 
{
  try
    {
      CGAL::Interval_nt_advanced::Protector P;
      auto res = intersect_ray_with_quadric(IT(qxx), IT(qyy), IT(qzz),
					    IT(bx), IT(by), IT(bz),
					    IT(c),
					    truncating_plane,
					    IT(px), IT(py), IT(pz),
					    IT(d),
					    IT(x0), IT(y0), IT(z0),
					    IT(vx), IT(vy), IT(vz));
      std::vector<FT> ft_res;
      for (size_t i = 0; i < res.size(); ++i)
  	ft_res.push_back(CGAL::to_double(res[i]));
      return ft_res;
    }
  catch (...)
    {
      std::cerr << "exact intersection";
      auto res = intersect_ray_with_quadric(ET(qxx), ET(qyy), ET(qzz),
					    ET(bx), ET(by), ET(bz),
					    ET(c),
					    truncating_plane,
					    ET(px), ET(py), ET(pz),
					    ET(d),
					    ET(x0), ET(y0), ET(z0),
					    ET(vx), ET(vy), ET(vz));
      std::vector<FT> ft_res;
      for (size_t i = 0; i < res.size(); ++i)
	ft_res.push_back(CGAL::to_double(res[i]));
      return ft_res;
    }
}


PYBIND11_PLUGIN(pypower)
{
  py::module m("pypower", "pypower");
  m.def("power_diagram", &power_diagram, "Computes a power diagram");
  m.def("point_above_quadric", &point_above_quadric_exact);
  m.def("intersect_segment_with_quadric", &intersect_segment_with_quadric_exact);
  m.def("intersect_ray_with_quadric", &intersect_ray_with_quadric_exact);
  return m.ptr();
}
