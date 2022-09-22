#include<iostream>
#include<cstdio>
#include<cmath>
#include<vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include<pybind11/numpy.h>

namespace py = pybind11;

using namespace std;


const double window = 20.;
const double coin = 130;


bool relajudge(double* points,int a,int b)
{
  //judge 2 hit relative or not
  double d;
  d =((points[4*a+1]-points[4*b+1])*(points[4*a+1]-points[4*b+1]))+((points[4*a+2]-points[4*b+2])*(points[4*a+2]-points[4*b+2]))+((points[4*a+3]-points[4*b+3])*(points[4*a+3]-points[4*b+3]));
  d += ((points[4*a]-points[4*b])*(points[4*a]-points[4*b]))*0.09;
  d = sqrt(d);
  if(d<coin) return 1;
  else return 0;
}

bool timejudge(double* points,int a,int b)
{
  double t;
  t = abs(points[4*a]-points[4*b]);
  if(t<window) return 1;
  else return 0;
}

vector<int> datap(py::array_t<double>& points,int hits_num)
{
    py::buffer_info buf = points.request();
   
    if(!hits_num)
    {
      printf("No hits");
      return {};
    }
    if(hits_num>20000)
    {
      printf("Too many hits, return false.");
      return {};
    }
    int link_num=0;
    vector<int> ts;

    double *ptr = (double*)buf.ptr;
    //cout<<"######"<<endl;
    for(int i=0;i<hits_num;i++)
    {
      for(int j=i+1;j<hits_num;j++)
      {
        if(timejudge(ptr,i,j))
        {
          if(relajudge(ptr,i,j))
          {
            ts.push_back(i);
            ts.push_back(j);
            //if(link_num<=10) cout<<i<<","<<j<<endl;
            link_num++;
          }
        }
        else break;
      }
      if(link_num>5000000)
      {
        printf("Too many edges for this graph, return false.");
        return {};
      }
    }
    if(!link_num) 
    {
      printf("No edges in this event. Event dumped!");
      return {};
    }
   // cout<<"vecsize:"<<ts.size()<<endl;
    cout<<"num:"<<link_num<<endl;
    ts.insert(ts.begin(),link_num);
    cout<<"call:"<<ts.size()<<endl;
    //auto result = py::array_t<int>(ts.size());
    //py::buffer_info bufr = result.request();
//cout<<"@@@@@@@@@@@"<<endl;
    //int *ptrr = (int*)bufr.ptr;
    //ptrr = ts.data(); 

    //cout<<"*****"<<endl;
    //for(int i=0;i<=10;i++) cout<<ts[i]<<endl;
    return ts;

}


PYBIND11_MODULE(datap, m)
{
    m.doc() = "pipeline return vector"; 

    m.def("datap", &datap, "A function that return data vector");
}
