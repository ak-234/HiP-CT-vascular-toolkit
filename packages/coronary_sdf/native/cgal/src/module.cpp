#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/iterator.h>
#include <CGAL/facets_in_complex_3_to_triangle_mesh.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/version.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;
#define CORONARY_SDF_STRINGIFY_DETAIL(value) #value
#define CORONARY_SDF_STRINGIFY(value) CORONARY_SDF_STRINGIFY_DETAIL(value)
#if defined(_MSC_FULL_VER)
static constexpr const char* compiler_string =
    "MSVC " CORONARY_SDF_STRINGIFY(_MSC_FULL_VER);
#elif defined(__clang__)
static constexpr const char* compiler_string = "Clang " __clang_version__;
#elif defined(__GNUC__)
static constexpr const char* compiler_string = "GCC " __VERSION__;
#else
static constexpr const char* compiler_string = "unknown";
#endif
static constexpr const char* extension_version = "0.3.0";
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = K::Point_3;
using Mesh_domain = CGAL::Labeled_mesh_domain_3<K>;
using Tr = CGAL::Mesh_triangulation_3<
    Mesh_domain, CGAL::Default, CGAL::Sequential_tag>::type;
using C3t3 = CGAL::Mesh_complex_3_in_triangulation_3<Tr>;
using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;
namespace params = CGAL::parameters;

struct Vec3 {
  double x{}, y{}, z{};
};

static Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
static Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static Vec3 operator*(Vec3 a, double s) { return {a.x*s,a.y*s,a.z*s}; }
static double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
static double norm(Vec3 a) { return std::sqrt(dot(a,a)); }
static Vec3 from_point(const Point& p) { return {p.x(),p.y(),p.z()}; }

struct Capsule {
  Vec3 start, end;
  double r0{}, r1{}, max_radius{};
  std::int64_t segment{};
  std::array<Vec3,2> clip_normals{};
  std::array<double,2> clip_offsets{};
  std::int64_t clip_count{};
};

struct Junction {
  Vec3 position;
  double radius{}, blend_fraction{}, support_factor{}, core_fraction{};
  std::vector<std::vector<std::size_t>> groups;
};

struct PrimitiveValue {
  double value{};
  double radius{};
};

static PrimitiveValue round_cone(Vec3 p, const Capsule& capsule) {
  const Vec3 ba = capsule.end - capsule.start;
  const Vec3 pa = p - capsule.start;
  const double l2 = dot(ba, ba);
  const double y = dot(pa, ba);
  const double rr = capsule.r0 - capsule.r1;
  const double a2 = l2 - rr * rr;
  if (l2 <= 1e-30 || a2 <= 1e-30) {
    const bool use_start = capsule.r0 >= capsule.r1;
    const Vec3 center = use_start ? capsule.start : capsule.end;
    const double radius = use_start ? capsule.r0 : capsule.r1;
    return {norm(p-center)-radius, radius};
  }
  const double z = y-l2;
  const Vec3 cross_scaled = pa*l2-ba*y;
  const double x2 = dot(cross_scaled,cross_scaled);
  const double y2 = y*y*l2;
  const double z2 = z*z*l2;
  const double k = (rr >= 0.0 ? 1.0 : -1.0)*rr*rr*x2;
  const double inv_l2 = 1.0/l2;
  const bool end_region = (z >= 0.0 ? 1.0 : -1.0)*a2*z2 > k;
  const bool start_region = (y >= 0.0 ? 1.0 : -1.0)*a2*y2 < k;
  double t = std::clamp(y*inv_l2,0.0,1.0);
  double radius = capsule.r0+t*(capsule.r1-capsule.r0);
  double result = (std::sqrt(std::max(x2*a2*inv_l2,0.0))+y*rr)*inv_l2-capsule.r0;
  if (start_region) {
    result=std::sqrt(x2+y2)*inv_l2-capsule.r0;
    radius=capsule.r0;
  } else if (end_region) {
    result=std::sqrt(x2+z2)*inv_l2-capsule.r1;
    radius=capsule.r1;
  }
  return {result,radius};
}

static PrimitiveValue clipped_round_cone(Vec3 p, const Capsule& capsule) {
  PrimitiveValue result=round_cone(p,capsule);
  for(std::int64_t i=0;i<capsule.clip_count;++i)
    result.value=std::max(result.value,dot(p,capsule.clip_normals[i])-capsule.clip_offsets[i]);
  return result;
}

struct BvhNode {
  Vec3 lo, hi;
  double max_radius{};
  int left{-1}, right{-1};
  std::vector<std::size_t> indices;
};

class GraphField {
 public:
  GraphField(std::vector<Capsule> capsules, std::vector<Junction> junctions,
             std::size_t leaf_size=8)
      : capsules_(std::move(capsules)), junctions_(std::move(junctions)),
        leaf_size_(std::max<std::size_t>(leaf_size,1)) {
    if (capsules_.empty()) throw std::invalid_argument("at least one capsule is required");
    minimum_positive_radius_=std::numeric_limits<double>::infinity();
    std::vector<std::size_t> indices(capsules_.size());
    std::iota(indices.begin(),indices.end(),0);
    for (const auto& c:capsules_) {
      if (c.r0<0.0 || c.r1<0.0) throw std::invalid_argument("radii must be non-negative");
      if (c.r0>0.0) minimum_positive_radius_=std::min(minimum_positive_radius_,c.r0);
      if (c.r1>0.0) minimum_positive_radius_=std::min(minimum_positive_radius_,c.r1);
    }
    if (!std::isfinite(minimum_positive_radius_))
      throw std::invalid_argument("at least one positive radius is required");
    root_=build(indices);
  }

  PrimitiveValue evaluate(Vec3 point) const {
    ++queries_;
    PrimitiveValue best=hard_min(point);
    for (const auto& junction:junctions_) {
      const Vec3 delta=point-junction.position;
      const double distance=norm(delta);
      const double support=junction.support_factor*junction.radius;
      const double weight=support_weight(distance,support);
      if (weight<=0.0 || junction.groups.size()<2) continue;
      std::vector<double> branch_values;
      branch_values.reserve(junction.groups.size());
      for (const auto& group:junction.groups) {
        double value=std::numeric_limits<double>::infinity();
        for (const auto index:group) value=std::min(value,clipped_round_cone(point,capsules_[index]).value);
        branch_values.push_back(value);
      }
      const double hard=*std::min_element(branch_values.begin(),branch_values.end());
      const double depth=junction.blend_fraction*junction.radius;
      double candidate=hard;
      if (depth>0.0) {
        const double tau=depth/std::log(static_cast<double>(branch_values.size()));
        double sum=0.0;
        for (double value:branch_values) sum+=std::exp(-(value-hard)/tau);
        const double soft=hard-tau*std::log(sum);
        candidate=hard+weight*(soft-hard);
      }
      if (junction.core_fraction>0.0)
        candidate=std::min(candidate,distance-junction.core_fraction*junction.radius);
      if (candidate<best.value) best={candidate,junction.radius};
    }
    if (best.radius<=0.0) best.radius=minimum_positive_radius_;
    return best;
  }

  double size_at(Vec3 point,double cells) const {
    return 2.0*evaluate(point).radius/cells;
  }
  std::size_t query_count() const { return queries_.load(); }

 private:
  int build(std::vector<std::size_t>& indices) {
    BvhNode node;
    node.lo={std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
    node.hi={-node.lo.x,-node.lo.y,-node.lo.z};
    std::array<double,3> centroid_lo{node.lo.x,node.lo.y,node.lo.z};
    std::array<double,3> centroid_hi{node.hi.x,node.hi.y,node.hi.z};
    for (auto index:indices) {
      const auto& c=capsules_[index];
      node.lo.x=std::min({node.lo.x,c.start.x,c.end.x}); node.hi.x=std::max({node.hi.x,c.start.x,c.end.x});
      node.lo.y=std::min({node.lo.y,c.start.y,c.end.y}); node.hi.y=std::max({node.hi.y,c.start.y,c.end.y});
      node.lo.z=std::min({node.lo.z,c.start.z,c.end.z}); node.hi.z=std::max({node.hi.z,c.start.z,c.end.z});
      node.max_radius=std::max(node.max_radius,c.max_radius);
      const std::array<double,3> centroid{0.5*(c.start.x+c.end.x),0.5*(c.start.y+c.end.y),0.5*(c.start.z+c.end.z)};
      for(int a=0;a<3;++a){centroid_lo[a]=std::min(centroid_lo[a],centroid[a]);centroid_hi[a]=std::max(centroid_hi[a],centroid[a]);}
    }
    const int id=static_cast<int>(nodes_.size()); nodes_.push_back(node);
    if(indices.size()<=leaf_size_){nodes_[id].indices=indices;return id;}
    const std::array<double,3> extent{
      centroid_hi[0]-centroid_lo[0],centroid_hi[1]-centroid_lo[1],centroid_hi[2]-centroid_lo[2]};
    const auto axis=static_cast<int>(
      std::max_element(extent.begin(),extent.end())-extent.begin());
    auto coordinate=[&](std::size_t i){const auto& c=capsules_[i];return axis==0?0.5*(c.start.x+c.end.x):axis==1?0.5*(c.start.y+c.end.y):0.5*(c.start.z+c.end.z);};
    const auto middle=indices.begin()+indices.size()/2;
    std::nth_element(indices.begin(),middle,indices.end(),[&](auto a,auto b){return coordinate(a)<coordinate(b);});
    std::vector<std::size_t> left(indices.begin(),middle),right(middle,indices.end());
    nodes_[id].left=build(left); nodes_[id].right=build(right); return id;
  }

  double lower_bound(const BvhNode& node,Vec3 p) const {
    const double dx=std::max({node.lo.x-p.x,0.0,p.x-node.hi.x});
    const double dy=std::max({node.lo.y-p.y,0.0,p.y-node.hi.y});
    const double dz=std::max({node.lo.z-p.z,0.0,p.z-node.hi.z});
    return std::sqrt(dx*dx+dy*dy+dz*dz)-node.max_radius;
  }
  PrimitiveValue hard_min(Vec3 point) const {
    using Entry=std::pair<double,int>;
    std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> queue;
    queue.push({lower_bound(nodes_[root_],point),root_});
    PrimitiveValue best{std::numeric_limits<double>::infinity(),minimum_positive_radius_};
    while(!queue.empty()){
      const auto [bound,id]=queue.top();queue.pop(); if(bound>best.value) break;
      const auto& node=nodes_[id];
      if(!node.indices.empty()){
        for(auto index:node.indices){auto value=clipped_round_cone(point,capsules_[index]);if(value.value<best.value)best=value;}
      }else{
        for(int child:{node.left,node.right}){const double child_bound=lower_bound(nodes_[child],point);if(child_bound<=best.value)queue.push({child_bound,child});}
      }
    }
    return best;
  }
  static double support_weight(double distance,double support){
    const double inner=0.75*support;
    if(distance<=inner)return 1.0;if(distance>=support)return 0.0;
    const double t=(distance-inner)/(support-inner);return 1.0-t*t*(3.0-2.0*t);
  }
  std::vector<Capsule> capsules_;
  std::vector<Junction> junctions_;
  std::vector<BvhNode> nodes_;
  std::size_t leaf_size_{};
  int root_{};
  double minimum_positive_radius_{};
  mutable std::atomic<std::size_t> queries_{0};
};

template <typename T>
static py::array_t<T,py::array::c_style|py::array::forcecast> checked_array(
    py::handle value,const char* name,int ndim){
  auto result=py::array_t<T,py::array::c_style|py::array::forcecast>::ensure(value);
  if(!result || result.ndim()!=ndim)throw std::invalid_argument(std::string(name)+" has invalid rank");
  return result;
}

static std::shared_ptr<GraphField> make_field(
    py::array starts_obj,py::array ends_obj,py::array r0_obj,py::array r1_obj,
    py::array segments_obj,py::array clip_normals_obj,py::array clip_offsets_obj,
    py::array clip_count_obj,
    py::array junction_positions_obj,py::array junction_radii_obj,
    py::array junction_blends_obj,py::array junction_supports_obj,py::array junction_cores_obj,
    py::array junction_group_offsets_obj,py::array group_capsule_offsets_obj,
    py::array group_capsules_obj,std::size_t leaf_size){
  auto starts=checked_array<double>(starts_obj,"starts",2),ends=checked_array<double>(ends_obj,"ends",2);
  auto r0=checked_array<double>(r0_obj,"radii_start",1),r1=checked_array<double>(r1_obj,"radii_end",1);
  auto segments=checked_array<std::int64_t>(segments_obj,"segment_ids",1);
  auto clip_normals=checked_array<double>(clip_normals_obj,"clip_plane_normals",3);
  auto clip_offsets=checked_array<double>(clip_offsets_obj,"clip_plane_offsets",2);
  auto clip_count=checked_array<std::int64_t>(clip_count_obj,"clip_plane_count",1);
  if(starts.shape(1)!=3 || ends.shape(1)!=3 || starts.shape(0)!=ends.shape(0) || starts.shape(0)!=r0.shape(0) || starts.shape(0)!=r1.shape(0) || starts.shape(0)!=segments.shape(0) || clip_normals.shape(0)!=starts.shape(0) || clip_normals.shape(1)!=2 || clip_normals.shape(2)!=3 || clip_offsets.shape(0)!=starts.shape(0) || clip_offsets.shape(1)!=2 || clip_count.shape(0)!=starts.shape(0))throw std::invalid_argument("inconsistent capsule arrays");
  auto s=starts.unchecked<2>(),e=ends.unchecked<2>(),a=r0.unchecked<1>(),b=r1.unchecked<1>(),ids=segments.unchecked<1>(),cn=clip_normals.unchecked<3>(),co=clip_offsets.unchecked<2>(),cc=clip_count.unchecked<1>();
  std::vector<Capsule> capsules;capsules.reserve(starts.shape(0));
  for(py::ssize_t i=0;i<starts.shape(0);++i){Capsule item;item.start={s(i,0),s(i,1),s(i,2)};item.end={e(i,0),e(i,1),e(i,2)};item.r0=a(i);item.r1=b(i);item.max_radius=std::max(a(i),b(i));item.segment=ids(i);item.clip_count=cc(i);if(item.clip_count<0 || item.clip_count>2)throw std::invalid_argument("clip plane count must be in [0,2]");for(std::int64_t k=0;k<item.clip_count;++k){item.clip_normals[k]={cn(i,k,0),cn(i,k,1),cn(i,k,2)};item.clip_offsets[k]=co(i,k);}capsules.push_back(item);}
  auto jp=checked_array<double>(junction_positions_obj,"junction_positions",2);
  auto jr=checked_array<double>(junction_radii_obj,"junction_radii",1);
  auto jb=checked_array<double>(junction_blends_obj,"junction_blends",1);
  auto js=checked_array<double>(junction_supports_obj,"junction_supports",1);
  auto jc=checked_array<double>(junction_cores_obj,"junction_cores",1);
  auto jgo=checked_array<std::int64_t>(junction_group_offsets_obj,"junction_group_offsets",1);
  auto gco=checked_array<std::int64_t>(group_capsule_offsets_obj,"group_capsule_offsets",1);
  auto gc=checked_array<std::int64_t>(group_capsules_obj,"group_capsules",1);
  if(jp.shape(1)!=3 || jp.shape(0)!=jr.shape(0) || jp.shape(0)!=jb.shape(0) || jp.shape(0)!=js.shape(0) || jp.shape(0)!=jc.shape(0) || jgo.shape(0)!=jp.shape(0)+1)throw std::invalid_argument("inconsistent junction arrays");
  auto p=jp.unchecked<2>(),rad=jr.unchecked<1>(),blend=jb.unchecked<1>(),support=js.unchecked<1>(),core=jc.unchecked<1>(),jo=jgo.unchecked<1>(),go=gco.unchecked<1>(),ci=gc.unchecked<1>();
  std::vector<Junction> junctions;junctions.reserve(jp.shape(0));
  for(py::ssize_t j=0;j<jp.shape(0);++j){Junction item;item.position={p(j,0),p(j,1),p(j,2)};item.radius=rad(j);item.blend_fraction=blend(j);item.support_factor=support(j);item.core_fraction=core(j);
    for(std::int64_t g=jo(j);g<jo(j+1);++g){std::vector<std::size_t> group;for(std::int64_t k=go(g);k<go(g+1);++k){if(ci(k)<0 || ci(k)>=starts.shape(0))throw std::invalid_argument("junction capsule index out of range");group.push_back(static_cast<std::size_t>(ci(k)));}item.groups.push_back(std::move(group));}junctions.push_back(std::move(item));}
  return std::make_shared<GraphField>(std::move(capsules),std::move(junctions),leaf_size);
}

struct ImplicitFunction {
  using FT=K::FT;
  std::shared_ptr<GraphField> field;
  FT operator()(const Point& point) const{return field->evaluate(from_point(point)).value;}
};
struct SizeField {
  using FT=K::FT;using Point_3=Point;using Index=Mesh_domain::Index;
  std::shared_ptr<GraphField> field;double cells{},factor{1.0};
  FT operator()(const Point_3& point,const int,const Index&)const{return factor*field->size_at(from_point(point),cells);}
};

static py::dict mesh_native(
    std::shared_ptr<GraphField> field,py::array center_obj,double sphere_radius,
    double cells,double facet_angle,double facet_distance_fraction,
    double cell_size_factor,double cell_radius_edge_ratio){
  auto center=checked_array<double>(center_obj,"bounding_center",1);
  if(center.shape(0)!=3 || sphere_radius<=0.0 || cells<=0.0)throw std::invalid_argument("invalid meshing bounds/criteria");
  auto c=center.unchecked<1>();const Point centre(c(0),c(1),c(2));
  ImplicitFunction function{field};
  Mesh_domain domain=Mesh_domain::create_implicit_mesh_domain(function,K::Sphere_3(centre,sphere_radius*sphere_radius));
  SizeField facet_size{field,cells,1.0};
  SizeField facet_distance{field,cells,facet_distance_fraction};
  SizeField cell_size{field,cells,cell_size_factor};
  Mesh_criteria criteria(params::facet_angle(facet_angle).facet_size(facet_size).facet_distance(facet_distance).cell_radius_edge_ratio(cell_radius_edge_ratio).cell_size(cell_size));
  const auto started=std::chrono::steady_clock::now();
  C3t3 c3t3;
  {py::gil_scoped_release release;c3t3=CGAL::make_mesh_3<C3t3>(domain,criteria,params::no_exude().no_perturb());}
  const auto meshing_finished=std::chrono::steady_clock::now();
  const double elapsed=std::chrono::duration<double>(meshing_finished-started).count();
  using Surface_mesh=CGAL::Surface_mesh<Point>;
  Surface_mesh surface;
  CGAL::facets_in_complex_3_to_triangle_mesh(c3t3,surface);
  std::vector<Vec3> vertices;
  vertices.reserve(surface.number_of_vertices());
  std::vector<std::int64_t> vertex_ids(surface.number_of_vertices(),-1);
  for(const auto vertex:surface.vertices()){
    const auto id=static_cast<std::int64_t>(vertices.size());
    vertex_ids[vertex.idx()]=id;
    vertices.push_back(from_point(surface.point(vertex)));
  }
  std::vector<std::array<std::int64_t,3>> faces;
  faces.reserve(surface.number_of_faces());
  for(const auto face_index:surface.faces()){
    std::array<std::int64_t,3> face{};
    int cursor=0;
    for(const auto vertex:CGAL::vertices_around_face(surface.halfedge(face_index),surface)){
      if(cursor>=3)throw std::runtime_error("CGAL returned a non-triangular surface face");
      face[cursor++]=vertex_ids[vertex.idx()];
    }
    if(cursor!=3)throw std::runtime_error("CGAL returned an incomplete surface face");
    const Vec3 a=vertices[face[0]],b=vertices[face[1]],d=vertices[face[2]];
    Vec3 normal=cross(b-a,d-a);
    const double length=norm(normal);
    if(length<=1e-30)continue;
    normal=normal*(1.0/length);
    const Vec3 centroid=(a+b+d)*(1.0/3.0);
    const double eps=std::max(field->size_at(centroid,cells)*1e-4,1e-12);
    // Negative is lumen/interior. Make the normal point toward increasing field.
    if(field->evaluate(centroid+normal*eps).value<field->evaluate(centroid-normal*eps).value)
      std::swap(face[1],face[2]);
    faces.push_back(face);
  }
  py::array_t<double> vertex_array({static_cast<py::ssize_t>(vertices.size()),3});auto vo=vertex_array.mutable_unchecked<2>();for(std::size_t i=0;i<vertices.size();++i){vo(i,0)=vertices[i].x;vo(i,1)=vertices[i].y;vo(i,2)=vertices[i].z;}
  py::array_t<std::int64_t> face_array({static_cast<py::ssize_t>(faces.size()),3});auto fo=face_array.mutable_unchecked<2>();for(std::size_t i=0;i<faces.size();++i)for(int j=0;j<3;++j)fo(i,j)=faces[i][j];
  const auto extraction_finished=std::chrono::steady_clock::now();
  const double extraction_elapsed=std::chrono::duration<double>(extraction_finished-meshing_finished).count();
  py::dict telemetry;telemetry["cgal_version"]=CGAL_VERSION_STR;telemetry["extension_version"]=extension_version;telemetry["compiler"]=compiler_string;telemetry["meshing_seconds"]=elapsed;telemetry["surface_extraction_seconds"]=extraction_elapsed;telemetry["native_total_seconds"]=elapsed+extraction_elapsed;telemetry["field_queries"]=field->query_count();telemetry["volume_vertices"]=c3t3.triangulation().number_of_vertices();telemetry["surface_vertices"]=vertices.size();telemetry["surface_faces"]=faces.size();
  py::dict result;result["vertices"]=vertex_array;result["triangles"]=face_array;result["telemetry"]=telemetry;return result;
}

PYBIND11_MODULE(coronary_sdf_cgal,module){
  module.attr("API_VERSION")=3;
  module.attr("CGAL_VERSION")=CGAL_VERSION_STR;
  module.attr("EXTENSION_VERSION")=extension_version;
  module.attr("COMPILER")=compiler_string;
  module.def("create_field",&make_field,py::arg("starts"),py::arg("ends"),py::arg("radii_start"),py::arg("radii_end"),py::arg("segment_ids"),py::arg("clip_plane_normals"),py::arg("clip_plane_offsets"),py::arg("clip_plane_count"),py::arg("junction_positions"),py::arg("junction_radii"),py::arg("junction_blends"),py::arg("junction_supports"),py::arg("junction_cores"),py::arg("junction_group_offsets"),py::arg("group_capsule_offsets"),py::arg("group_capsules"),py::arg("leaf_size")=8);
  py::class_<GraphField,std::shared_ptr<GraphField>>(module,"NativeGraphField")
    .def("evaluate",[](const GraphField& field,py::array points_obj){auto points=checked_array<double>(points_obj,"points",2);if(points.shape(1)!=3)throw std::invalid_argument("points must have shape (N,3)");auto p=points.unchecked<2>();py::array_t<double> values(points.shape(0)),radii(points.shape(0));auto v=values.mutable_unchecked<1>(),r=radii.mutable_unchecked<1>();for(py::ssize_t i=0;i<points.shape(0);++i){auto sample=field.evaluate({p(i,0),p(i,1),p(i,2)});v(i)=sample.value;r(i)=sample.radius;}return py::make_tuple(values,radii);})
    .def("sizing",[](const GraphField& field,py::array points_obj,double cells){auto points=checked_array<double>(points_obj,"points",2);if(points.shape(1)!=3)throw std::invalid_argument("points must have shape (N,3)");auto p=points.unchecked<2>();py::array_t<double> sizes(points.shape(0));auto s=sizes.mutable_unchecked<1>();for(py::ssize_t i=0;i<points.shape(0);++i)s(i)=field.size_at({p(i,0),p(i,1),p(i,2)},cells);return sizes;});
  module.def("mesh_implicit_surface",&mesh_native,py::arg("field"),py::arg("bounding_center"),py::arg("bounding_radius"),py::arg("cells_across_diameter"),py::arg("facet_angle_deg")=30.0,py::arg("facet_distance_fraction")=0.25,py::arg("cell_size_factor")=2.0,py::arg("cell_radius_edge_ratio")=2.0);
}
