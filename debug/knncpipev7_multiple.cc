#include<iostream>
#include<cstdio>
#include<cmath>
#include<vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include<pybind11/numpy.h>
#include<algorithm>
//Last edition oct.20 Metrics are added
//Last edition oct.28 6nn->4nn
//Last edition oct 30 3nn
//Last edition nov 2 4nn
//nov 9 6
//nov 12 1 forE
////nov14 4
//nov27 >0cut
//nov 29 sufcut do pipe for <60
namespace py = pybind11;

using namespace std;


double get_distance(double* points,int a,int b)
{
  //judge 2 hit relative or not
  double d;
  d =((points[5*a+1]-points[5*b+1])*(points[5*a+1]-points[5*b+1]))+((points[5*a+2]-points[5*b+2])*(points[5*a+2]-points[5*b+2]))+((points[5*a+3]-points[5*b+3])*(points[5*a+3]-points[5*b+3]));
  d += ((points[5*a]-points[5*b])*(points[5*a]-points[5*b]))*0.09;
  d = sqrt(d);
  return d;
}
double get_3ddistance(double* points,int a,int b)
{
  
  double d;
  d =((points[5*a+1]-points[5*b+1])*(points[5*a+1]-points[5*b+1]))+((points[5*a+2]-points[5*b+2])*(points[5*a+2]-points[5*b+2]))+((points[5*a+3]-points[5*b+3])*(points[5*a+3]-points[5*b+3]));
  //d += ((points[5*a]-points[5*b])*(points[5*a]-points[5*b]))*0.09;
  d = sqrt(d);
  return d;
}
bool causjudge(double* points,int a,int b)
{
  double d;
  d =((points[5*a+1]-points[5*b+1])*(points[5*a+1]-points[5*b+1]))+((points[5*a+2]-points[5*b+2])*(points[5*a+2]-points[5*b+2]))+((points[5*a+3]-points[5*b+3])*(points[5*a+3]-points[5*b+3]));
  d += -((points[5*a]-points[5*b])*(points[5*a]-points[5*b]))*0.09;

  if(d<-0.00001) return 1;
  else return 0;
}
bool samstrjudge(double* points,int a,int b)
{
  if(points[5*a+4] ==points[5*b+4]) return 1;
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

vector<double> mulknndatap(py::array_t<double>& points,int hits_num)
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
    bool samrela;
    int link_num=0;
    int caus_counter,str_counter,geo_counter;
    int knn_num = 4;//change 2 place
    vector<double> ts;
    int str_limit=2;
    int geo_limit=2;
    struct sam list[1000];
    double *ptr = (double*)buf.ptr;
    for(int i=hits_num-1;i>0;i--)
    {

      //cout<<"heyyyy"<<endl;
      knn_num = 4;
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
            ts.push_back(get_3ddistance(ptr,k,i));

          }
        }
        else
        {
          for(int k=1;k<=knn_num;k++)
          {
            ts.push_back(i-k);
            ts.push_back(i);
            ts.push_back(get_3ddistance(ptr,i-k,i));

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
          ts.push_back(get_3ddistance(ptr,list[k].index,i));
        }
        if(knn_num-caus_counter>i)
        {
          for(int k =0;k<=i-1;k++)
          {
            ts.push_back(k);
            ts.push_back(i);
            ts.push_back(get_3ddistance(ptr,k,i));

          }
          continue;
        }
        for(int k=1;k<=knn_num-caus_counter;k++)
        {
          ts.push_back(i-k);
          ts.push_back(i);
          ts.push_back(get_3ddistance(ptr,i-k,i));

        }
        continue;
      }
      str_counter=0;
      geo_counter=0;
      link_num=0;
      for(int k=0;k<caus_counter;k++)
      {
            //in this case enough causual earlier nodes, try to restrict samstr links
            if(link_num==knn_num) break;
            samrela =  samstrjudge(ptr,list[k].index,i);
            if(samrela&&(str_counter<str_limit))
            {
                ts.push_back(list[k].index);
                ts.push_back(i);
                //ts.push_back(list[k].distance);
                ts.push_back(get_3ddistance(ptr,list[k].index,i));
                link_num++;
                str_counter++;
            }
            if(!samrela&&(geo_counter<geo_limit))
            {
                ts.push_back(list[k].index);
                ts.push_back(i);
                //ts.push_back(list[k].distance);
                ts.push_back(get_3ddistance(ptr,list[k].index,i));
                link_num++;
                geo_counter++;
            }
            
      }
 
    }
    
    /*if(!link_num) 
    {
      printf("No edges in this event. Event dumped!");
      return {};
    }*/
   // cout<<"vecsize:"<<ts.size()<<endl;
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


PYBIND11_MODULE(mulknndatap, m)
{
    m.doc() = "pipeline return vector"; 

    m.def("mulknndatap", &mulknndatap, "A function that return data vector");
}
