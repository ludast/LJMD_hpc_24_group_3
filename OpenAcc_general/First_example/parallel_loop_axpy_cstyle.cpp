#define _USE_MATH_DEFINES
#include<iostream>
#include<vector>
#include<cmath>
#include<iomanip>
#include<openacc.h>
using namespace std; 

int main(){
  constexpr size_t N = 100000; 
  constexpr double  alpha = M_PI/double(N); 
  double* buff_a = (double*) malloc(N*sizeof(double)); 
  double* buff_b = (double*) malloc(N*sizeof(double));  
  double* buff_c = (double*) malloc(N*sizeof(double));
  for (size_t i=0; i< N; ++i){
    buff_a[i] = 1.0;
    buff_b[i] = 1.0;
    buff_c[i] = 0.0;    
  } 
  #pragma acc parallel copyout(buff_c[0:N])  
  #pragma acc loop vector 
  for (size_t i=0; i < N; ++i){
    buff_c[i] = buff_c[i] + i * alpha *  (buff_a[i] + buff_b[i]);  
  }
#if defined(_OPENACC)
  cout << acc_is_present(buff_c,N) << endl;
#endif
  auto c = vector<double>(buff_c, buff_c + N);
  auto start = c.begin(); 
  cout << scientific << setprecision(6); 
  for (int  i =0; i<4; ++i){
    for (double val: vector<double>(start, start+5)){  
      cout << setw(16) << val << " ";
    }
    cout<< endl; 
    start = start + 5;
  }
}

