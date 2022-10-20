#include<iostream>
#include<cstdio>
#include<cmath>
#include<vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include<pybind11/numpy.h>
#include<algorithm>
//Last edition oct.20 Metrics are added

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

vector<double> knndatap(py::array_t<double>& points,int hits_num)
{
    cout<<"reach"<<endl;
    py::buffer_info buf = points.request();
   
    if(hits_num<60.1)
    {
            cout<<"<=  60 hits, cut!"<<endl;
            return {};
    }
    /*if(hits_num>20000)
    {
      printf("Too many hits, return false.");
      return {};
    }*/
    int link_num=0;
    int caus_counter;
    int knn_num = 6;//change 2 place
    vector<double> ts;

    struct sam list[1000];
    double *ptr = (double*)buf.ptr;
    for(int i=hits_num-1;i>0;i--)
    {

      //cout<<"heyyyy"<<endl;
      knn_num = 6;
      caus_counter =0;

      if(i>=1000)
      {
          for(int j=i-1;j>=i-1000;j--)
          {
            if(causjudge(ptr,j,i))
            {
                list[caus_counter].index = j;
                list[caus_counter].distance = get_distance(ptr,j,i);
                caus_counter++;
            }   
          }
      }
 
      else
      {
          for(int j=i-1;j>=0;j--)
          {
            if(causjudge(ptr,j,i))
            {
              list[caus_counter].index = j;
              list[caus_counter].distance = get_distance(ptr,j,i);
              caus_counter++;
            }        
          }
      }
      if(caus_counter>1)
        sort(list, list+caus_counter-1, comp_d);
      if(caus_counter ==0)//link to knn_num nearest time series neighbors
      {
        if(i<=knn_num)
        {
          for(int k=0;k<=i-1;k++)
          {
            ts.push_back(k);
            ts.push_back(i);
            ts.push_back(get_distance(ptr,k,i));

          }
        }
        else
        {
          for(int k=1;k<=knn_num;k++)
          {
            ts.push_back(i-k);
            ts.push_back(i);
            ts.push_back(get_distance(ptr,i-k,i));

          }
        }
        continue;
      }
      if(knn_num>caus_counter)
      {
        for(int k=0;k<caus_counter;k++)
        {
          ts.push_back(list[k].index);
          ts.push_back(i);
          ts.push_back(list[k].distance);
        }
        if(knn_num-caus_counter>i)
        {
          for(int k =0;k<=i-1;k++)
          {
            ts.push_back(k);
            ts.push_back(i);
            ts.push_back(get_distance(ptr,k,i));

          }
          continue;
        }
        for(int k=1;k<=knn_num-caus_counter;k++)
        {
          ts.push_back(i-k);
          ts.push_back(i);
          ts.push_back(get_distance(ptr,i-k,i));

        }
        continue;
      }
      for(int k=0;k<knn_num;k++)
      {
          
            ts.push_back(list[k].index);
            ts.push_back(i);
            ts.push_back(list[k].distance);
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
