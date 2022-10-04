/*
This code has been used to prepare experiments 
on relative orientation published in:

S. Segvic, G. Schweighofer and A. Pinz.
Performance evaluation of the five-point relative pose with emphasis on planar scenes. 
Proceedings of AAPR/OAGM 2007 

S. Segvic, G. Schweighofer and A. Pinz. 
Influence of the numerical conditioning on the accuracy of relative orientation. 
Proceedings of ISPRS BenCOS (CVPR workshop), 2007. 

This software is released under GPL 3.0.
http://www.gnu.org/copyleft/gpl.html
*/

#include <boost/random/linear_congruential.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <complex>
#include <algorithm>
#include <vector>
#include <limits>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace ublas = boost::numeric::ublas;


////////////////////////////////////////////
// Point2D

class Point2D: 
  public std::complex<double>
{
public:
  Point2D(){}
  Point2D(double x, double y): 
    std::complex<double>(x,y){}
  Point2D(const std::complex<double>& c):
    std::complex<double>(c) {}
  const Point2D& operator=(const std::complex<double>& c){
    *this=Point2D(c);return *this;}
  double x() const {return real();}
  double y() const {return imag();}
};


////////////////////////////////////////////
// VectorN

template <int N> 
class VectorN : 
  public boost::numeric::ublas::vector<
    double, boost::numeric::ublas::bounded_array<double,N> >
{
  typedef boost::numeric::ublas::vector<
    double, boost::numeric::ublas::bounded_array<double,N> > 
    VectorNBase;
public:
	VectorN(): VectorNBase(3)	{}

	// Construction and assignment from a uBLAS vector expression or copy assignment
	template <class R> 
  VectorN (const boost::numeric::ublas::vector_expression<R>& r) : 
    VectorNBase(r)
	{}
	template <class R> 
  void operator=(const boost::numeric::ublas::vector_expression<R>& r)
	{
		VectorNBase::operator=(r);
	}
	template <class R> 
  void operator=(const VectorNBase& r)
	{
		VectorNBase::operator=(r);
	}
};
class Vector3: public  VectorN<3>{
public:
	Vector3(){}
	Vector3(double x, double y, double z){
    (*this)[0]=x; (*this)[1]=y; (*this)[2]=z;}
	template <class R> 
  Vector3 (const boost::numeric::ublas::vector_expression<R>& r) : 
    VectorN<3>(r)
	{}
};

////////////////////////////////////////////
// MatrixNM

template <int N, int M>
class MatrixNM : 
  public boost::numeric::ublas::matrix<
    double, boost::numeric::ublas::row_major, 
            boost::numeric::ublas::bounded_array<double,N*M> >
{
  typedef boost::numeric::ublas::matrix<
    double, boost::numeric::ublas::row_major, 
            boost::numeric::ublas::bounded_array<double,N*M> > 
    MatrixNMBase;
public:
	MatrixNM(): MatrixNMBase(N,M)	{}

	// Construction and assignment from a uBLAS vector expression or copy assignment
	template <class R> 
  MatrixNM (const boost::numeric::ublas::matrix_expression<R>& r) : 
    MatrixNMBase(r)
	{}
	template <class R> 
  void operator=(const boost::numeric::ublas::matrix_expression<R>& r)
	{
		MatrixNMBase::operator=(r);
	}
	template <class R> 
  void operator=(const MatrixNMBase& r)
	{
		MatrixNMBase::operator=(r);
	}
};
typedef MatrixNM<3,3> Matrix3x3;
typedef MatrixNM<3,4> Matrix3x4;
typedef MatrixNM<9,9> Matrix9x9;


////////////////////////////////////////////
// functions

inline Vector3 cross_prod(
  const Vector3& v1,
  const Vector3& v2)
{
  return Vector3(
    v1(1)*v2(2)-v1(2)*v2(1),
   -v1(0)*v2(2)+v1(2)*v2(0),
    v1(0)*v2(1)-v1(1)*v2(0));
}

inline Matrix3x4 pmatrix(
  const Matrix3x3& R, 
  const Vector3& T)
{
  Matrix3x4 P;
  subrange(P, 0,3, 0,3)=R;
  column(P,3)=T;
  return P;
}

inline Matrix3x3 skewsym(
  const Vector3& T)
{
  Matrix3x3 St(boost::numeric::ublas::zero_matrix<double>(3,3));
  St(0,1)=-T(2);
  St(0,2)=T(1);
  St(1,2)=-T(0);
  St(1,0)=T(2);
  St(2,0)=-T(1);
  St(2,1)=T(0);
  return St;
}

Matrix3x3 rotxz(double phi){
  Matrix3x3 R(ublas::zero_matrix<double>(3,3));
  R(0,0)=cos(phi);
  R(0,2)=-sin(phi);
  R(2,0)=sin(phi);
  R(2,2)=cos(phi);
  R(1,1)=1;
  return R;
}

Matrix3x4 pmatrix(
  double rotangle, double direction)
{
  Matrix3x3 Rrot  (rotxz(-rotangle));
  Matrix3x3 Rtrans(rotxz(direction));
  Vector3 T_1(prod(Rtrans,Vector3(0,0,1)));
  Vector3 T_2(-prod(Rrot,T_1));
  Matrix3x4 P;
  subrange(P, 0,3, 0,3)=Rrot;
  column(P, 3)=T_2;
  return P;  
}


double rotDefault(double direction, 
  double distance, double depth, 
  double slant)
{
  double zCb=cos(direction);
  double xCb=sin(direction);
  double zTarget=distance+.5*depth/cos(slant);
  double xTarget=0;
  return atan((xTarget-xCb)/(zTarget-zCb));
}


////////////////////////////////////////////
// constants and small utils
namespace{
  static const double Pi_=4*atan(1);

  inline int myround(double f){
    return static_cast<int>(f<0?f-.5:f+.5);
  }

  inline double rad2deg(double rad){
    return rad*180/Pi_;
  }
  inline double deg2rad(double deg){
    return deg*Pi_/180;
  }
}

////////////////////////////////////////////
// some algebra

namespace{
  inline Vector3 transform(const Matrix3x4& P, const Vector3& Q){
    return prod(ublas::subrange(P, 0,3, 0,3),Q) + column(P,3);
  }
  inline Point2D myproject(const Vector3& Q){
    return Point2D(Q(0)/Q(2), Q(1)/Q(2));
  }
  inline Point2D myproject(const Matrix3x4& P, const Vector3& Q){
    Vector3 Qt(transform(P,Q));
    return myproject(Qt);
  }
  inline bool pointOK(const Point2D& q, double hfov,double vfov){
    bool ok= fabs(q.x())<tan(hfov/2) && 
             fabs(q.y())<tan(vfov/2);
    return ok;
  }
}



////////////////////////////////////////////
// World

class World{
public:
  int size() const {return Qs_.size();}
  const Vector3& Q(int i) const {return Qs_[i];}
  std::vector<Point2D> projectAll(const Matrix3x4& P, double noise);
  std::vector<Point2D> projectAll(double noise);
protected:
  typedef boost::minstd_rand RndGenType;
  static RndGenType RndGenObj_;

  typedef boost::normal_distribution<> DistribNorm;
  typedef boost::variate_generator<RndGenType,DistribNorm> GenDistNorm;
  static GenDistNorm rndnorm_;

  typedef boost::uniform_real<> DistribUni;
  typedef boost::variate_generator<RndGenType,DistribUni> GenDistUni;
  static GenDistUni rnduni_;
protected:
  std::vector<Vector3> Qs_;
};

World::RndGenType World::RndGenObj_(static_cast<unsigned int>(std::time(0)));
//World::RndGenType World::RndGenObj_(42);
World::GenDistNorm World::rndnorm_(RndGenObj_, DistribNorm(0,1));
World::GenDistUni  World::rnduni_ (RndGenObj_, DistribUni(0,1));

std::vector<Point2D> World::projectAll(double noise){
  Matrix3x4 P(ublas::zero_matrix<double>(3,4));
  P(0,0)=P(1,1)=P(2,2)=1;
  return projectAll(P, noise);
}

std::vector<Point2D> World::projectAll(
  const Matrix3x4& P, double sigma)
{
  std::vector<Point2D> qs(size());
  for (int i=0; i<size(); ++i){
    qs[i]=myproject(P,Q(i));

    // normal components
    Point2D err_norm(rndnorm_(),rndnorm_());
    
    // uniform components
    //Point2D err_uni(rnduni_()-0.5, rnduni_()-0.5);
    //err_uni*=sqrt(12);

    // normal magnitude, uniform direction
    //double r=rndnorm_();
    //double phi=Pi_*rnduni_();
    //Point2D err_norm_mag(r*cos(phi), r*sin(phi));

    qs[i]+=sigma*err_norm;
    ///std::cerr <<"e" <<err <<"\n";
  }
  return qs;
}


// WorldBrick
class WorldBrick: public World{
public:
  WorldBrick(int npoints, 
    double distance, double depth,
    double fov, const Matrix3x4& Pb);
};
WorldBrick::WorldBrick(int npoints, 
    double distance, double depth, 
    double hfov,
    const Matrix3x4& Pb)
{
  Qs_.resize(npoints);
  for (int i=0; i<npoints; ++i){
    Vector3& Qi=Qs_[i];
    while(1) {
      Qi[2]=distance+depth*rnduni_();
      Qi[1]=(distance+depth)*.75*tan(hfov/2)*2*(rnduni_()-0.5);
      Qi[0]=(distance+depth)*tan(hfov/2)*2*(rnduni_()-0.5);
      
      if (pointOK(myproject(Qi),   hfov,.75*hfov) && 
          pointOK(myproject(Pb,Qi),hfov,.75*hfov))
        break;
    }
  }
}

// WorldSlantedBrick
class WorldSlantedBrick: public World{
public:
  WorldSlantedBrick(const Matrix3x4& Pb,
    double distance, double depth, double slant, int npoints, 
    double hfov, double vfov);
};
// Ca ... camera A
// Za ... optical axis of the camera A
// plane1 ... the nearer plae
// plane2 ... the farther plae
WorldSlantedBrick::WorldSlantedBrick(const Matrix3x4& Pb,
    double distance, double depth, double slant, int npoints, 
    double hfov, double vfov)
{
  // distance from Ca to plane1
  double height=distance*cos(slant);
  // distance between the projection of Ca to plane1
  // and the intersection between Za and plane1
  double foot=distance*sin(slant);
  
  // the left and right limits of the volume
  double x_lf=        height*tan(hfov/2-slant)+foot;
  double x_lb=(height+depth)*tan(hfov/2-slant)+foot;
  double x_l=std::max(fabs(x_lf), fabs(x_lb));

  double x_rf=        height*tan(hfov/2+slant)-foot;
  double x_rb=(height+depth)*tan(hfov/2+slant)-foot;
  double x_r=std::max(fabs(x_rf), fabs(x_rb));
  
  // the bottom-top limits of the volume
  double y_l=2*(height+depth)/cos(hfov/2-slant)*tan(vfov/2);
  double y_r=2*(height+depth)/cos(hfov/2+slant)*tan(vfov/2);
  double y_bu=std::max(y_l,y_r);

  Matrix3x3 Rs(rotxz(slant));
  Vector3 Td(0,0,distance);

  Qs_.resize(npoints);
  int i=0;
  while (i<npoints){
    // std::cerr <<"Q" <<i <<"\r";    

    Vector3& Qi=Qs_[i];
    Qi(0)=(x_l+x_r)*rnduni_()-x_l;
    Qi(1)=y_bu*(rnduni_()-0.5);
    Qi(2)=depth*rnduni_();

    Qi=prod(Rs,Qi)+Td;

    if (pointOK(myproject(Qi),   hfov,vfov) && 
        pointOK(myproject(Pb,Qi),hfov,vfov))
    {
      i=i+1;
    }
  }  
}


// WorldGT
class WorldGT: public World{
public:
  WorldGT();
};
WorldGT::WorldGT(){
  Qs_.push_back(Vector3(1,1,10));
  Qs_.push_back(Vector3(0,1,10));
  Qs_.push_back(Vector3(1,0,10));
  Qs_.push_back(Vector3(1,1,15));
  Qs_.push_back(Vector3(0,1,15));
  Qs_.push_back(Vector3(1,0,15));
}




////////////////////////////////////////////
// main

  
void worker(
  std::string strGeom, 
  std::string strImg)
{
  ///////////////////
  //geometry defaults
  double rotangle=0;
  double direction=90;
  double distance=10;
  double depth=5;
  double slant=0;
  int npoints=10000;

  ///////////////////
  //imaging defaults
  double hfov=45;
  int imgwidth=384;
  int imgheight=288;
  double sigmapix=.5;



  ///////////////////
  //parse command line
  std::replace(strGeom.begin(), strGeom.end(), '_', ' ');
  std::istringstream issg(strGeom);
  std::string strrot;
  issg >>strrot >>direction 
       >>distance >>depth >>slant >>npoints;
  if (std::find(strrot.begin(), strrot.end(),'x') != strrot.end()){
    rotangle=rad2deg(rotDefault(deg2rad(direction),
                             distance,depth,deg2rad(slant)));
  } else{
    std::istringstream iss(strrot);
    iss >>rotangle;
  }

  std::replace(strImg.begin(), strImg.end(), '_', ' ');
  std::istringstream issi(strImg);
  issi >>hfov >>imgwidth >>imgheight >>sigmapix;
  sigmapix/=100.0;


  ///////////////////
  //report the experiment specification
  std::ostringstream oss;
  oss <<"M"  <<std::setfill('0') <<std::setw(2) <<myround(rotangle)
      <<":"  <<std::setfill('0') <<std::setw(2) <<myround(direction)
      <<"_D" <<std::setfill('0') <<std::setw(2) <<myround(distance)
      <<"_d" <<std::setw(2) <<myround(depth)
      <<"_s" <<std::setfill('0') <<std::setw(2) <<myround(slant)
      <<"_f" <<std::setfill('0') <<std::setw(3) <<myround(hfov)
      <<"_r" <<std::setw(3) <<imgwidth 
      <<"x"  <<std::setfill('0') <<std::setw(3) <<imgheight
      <<"_n" <<std::setfill('0') <<std::setw(3) <<myround(100*sigmapix);
  std::string expspec(oss.str());
  std::cerr <<expspec <<"\n";


  ///////////////////
  //create geometry
  Matrix3x4 P0(pmatrix(deg2rad(rotangle),deg2rad(direction)));
  //std::cerr <<"P0" <<P0 <<"\n";
  double szpix=2*tan(deg2rad(hfov)/2)/imgwidth;
  double vfov=2*rad2deg(atan(.5*imgheight*szpix));
  double sigmanorm=sigmapix * szpix;
  WorldSlantedBrick w(P0, distance,depth,deg2rad(slant), npoints,
                      deg2rad(hfov),deg2rad(vfov));
  //WorldBrick w(npoints, distance,depth, fov, P0);
  //WorldGT w;



  ///////////////////
  //project points
  std::vector<Point2D> qas(w.projectAll(sigmanorm));
  std::vector<Point2D> qbs(w.projectAll(P0,sigmanorm));




  ///////////////////
  //print everything
  std::cout <<P0 <<"\n";

  std::cout <<"\n";
  for (unsigned i=0; i<qas.size(); ++i){
    std::cout <<qas[i] <<"\n";
  }

  std::cout <<"\n";
  for (unsigned i=0; i<qas.size(); ++i){
    std::cout <<qbs[i] <<"\n";
  }

  std::cout <<"\n";
  for (unsigned i=0; i<qas.size(); ++i){
    std::cout <<w.Q(i) <<"\n";
  }
}

void usage(){
  std::cerr <<"Invoke with: expCreate <geompars> <imgpars>\n";
  std::cerr <<"  <geompars> --> <rotangle>_<direction>_<distance>_<depth>_<slant>_<npoints>\n";
  std::cerr <<"  <imgpars> --> <hfov>_<imgwidth>_<imgheight>_<sigmapix>\n";
}

int main(int argc, char* argv[]){  
  if (argc==3){
    worker(std::string(argv[1]),std::string(argv[2]));
  } else{
    usage();
  }
  return 0;
}  
