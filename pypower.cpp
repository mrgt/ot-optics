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
power_diagram(const Eigen::MatrixXd &X, const Eigen::VectorXd &w)
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
  std::vector<Vector> A;
  std::vector<double> b;
  A.push_back(Vector(0,0,1));  b.push_back(100);
  A.push_back(Vector(0,0,-1)); b.push_back(100);
  A.push_back(Vector(0,1,0));  b.push_back(1);
  A.push_back(Vector(0,-1,0)); b.push_back(1);
  A.push_back(Vector(1,0,0));  b.push_back(1);
  A.push_back(Vector(-1,0,0)); b.push_back(1);

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

PYBIND11_PLUGIN(pypower) {
    py::module m("pypower", "pypower");
    m.def("power_diagram", &power_diagram, "Computes a power diagram");
    return m.ptr();
}
