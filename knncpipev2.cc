#include<iostream>
#include<cstdio>
#include<cmath>
#include<vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include<pybind11/numpy.h>
#include<algorithm>

namespace py = pybind11;

using namespace std;


double get_distance(double* points,int a,int b)
{
  //judge 2 hit relative or not
  double d;
  d =((points[4*a+1]-points[4*b+1])*(points[4*a+1]-points[4*b+1]))+((points[4*a+2]-points[4*b+2])*(points[4*a+2]-points[4*b+2]))+((points[4*a+3]-points[4*b+3])*(points[4*a+3]-points[4*b+3]));
  d += ((points[4*a]-points[4*b])*(points[4*a]-points[4*b]))*0.09;
  d = sqrt(d);
  return d;
}

bool causjudge(double* points,int a,int b)
{
  double d;
  d =((points[4*a+1]-points[4*b+1])*(points[4*a+1]-points[4*b+1]))+((points[4*a+2]-points[4*b+2])*(points[4*a+2]-points[4*b+2]))+((points[4*a+3]-points[4*b+3])*(points[4*a+3]-points[4*b+3]));
  d += -((points[4*a]-points[4*b])*(points[4*a]-points[4*b]))*0.09;

  if(d<-0.00001) return 1;
  else return 0;
}
struct sam
{
  int index;
  double distance;
};


bool comp_d(const struct sam &a, const struct sam &b)
{
    if(a.distance<b.distance) return 1;
    else return 0;
}

vector<int> knndatap(py::array_t<double>& points,int hits_num)
{
    cout<<"reach"<<endl;
    py::buffer_info buf = points.request();
   
    if(!hits_num)
    {
            return {};
    }
    /*if(hits_num>20000)
    {
      printf("Too many hits, return false.");
      return {};
    }*/
    int link_num=0;
    int caus_counter;
    int knn_num = 6;
    vector<int> ts;

    struct sam list[1000];
    double *ptr = (double*)buf.ptr;
    for(int i=0;i<hits_num-1;i++)
    {

      //cout<<"heyyyy"<<endl;
      caus_counter =0;

      if(i<hits_num-1000)
      {
          for(int j=i+1;j<i+1001;j++)
          {
            if(causjudge(ptr,i,j))
            {
                list[caus_counter].index = j;
                list[caus_counter].distance = get_distance(ptr,i,j);
                caus_counter++;
            }   
          }
      }
 
      else
      {
          for(int j=i+1;j<hits_num;j++)
          {
            if(causjudge(ptr,i,j))
            {
              list[caus_counter].index = j;
              list[caus_counter].distance = get_distance(ptr,i,j);
              caus_counter++;
            }        
          }
      }
      if(caus_counter>1)
        sort(list, list+caus_counter-1, comp_d);
      if(caus_counter ==0)
        continue;
      if(knn_num>caus_counter) knn_num = caus_counter;
      for(int k=0;k<knn_num;k++)
      {
            ts.push_back(i);
            ts.push_back(list[k].index);
            link_num++;
      }

    }
    
    /*if(!link_num) 
    {
      printf("No edges in this event. Event dumped!");
      return {};
    }*/
   // cout<<"vecsize:"<<ts.size()<<endl;
    cout<<"num:"<<link_num<<endl;
    ts.insert(ts.begin(),link_num);
    //cout<<"call:"<<ts.size()<<endl;
    //auto result = py::array_t<int>(ts.size());
    //py::buffer_info bufr = result.request();
//cout<<"@@@@@@@@@@@"<<endl;
    //int *ptrr = (int*)bufr.ptr;
    //ptrr = ts.data(); 

    //cout<<"*****"<<endl;
    //for(int i=0;i<=10;i++) cout<<ts[i]<<endl;
    return ts;

}


PYBIND11_MODULE(knndatap, m)
{
    m.doc() = "pipeline return vector"; 

    m.def("knndatap", &knndatap, "A function that return data vector");
}
