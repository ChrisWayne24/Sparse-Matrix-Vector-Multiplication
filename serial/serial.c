#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>

void matrix_multiply_seq(float *a, float *b, float *ab, size_t width){
	int i, j, k;
	for(i=0; i<width; i++)
		for(j=0; j<width; j++){
			ab[i]=0.0;
			for(k=0; k<width; k++){
				ab[i] += a[i*width+k] * b[i];
			}
		}
}

int main(void){
  const size_t n = 317;
  //printf("%ld\n",n);
  float *h_a, *h_b, *h_s;
  
  h_a = (float *)malloc(sizeof(float) * n * n);
  h_b = (float *)malloc(sizeof(float) * n);
  h_s = (float *)malloc(sizeof(float) * n);
  //h_res = (float*)malloc(sizeof(float) * n * n);
  
  int z1 = (int)(n*n * 0.01);
  //printf("%d\n", z1);
  int z0 = 0;
  
  for(int i = 0; i < n*n; ++i){
    h_a[i] = 0;
  }
  
  for(int i = 0; i < n; ++i){
    h_b[i] = 1;
  }
  
  
  for(int i = 0; i < z1; ++i){
    int n1 = (int)((rand() % (n*n)) );
    h_a[n1]= 2;
  }
  /*
  fprintf(stdout, "Values Array:\n");
  for (int i = 0; i < n*n; i++) {
         fprintf(stdout, "%.1f ", h_a[i]);
  }
  std::cout<<std::endl;
  
  fprintf(stdout, "Dense Array:\n");
  for (int i = 0; i < n; i++) {
         fprintf(stdout, "%.1f ", h_b[i]);
  }
  std::cout<<std::endl;
  */
  double average_seq_time;
  struct timespec start, end;
  std::cout << "Timing sequential implementation...";
  
  if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
    perror( "clock gettime" );
    exit( EXIT_FAILURE );
  }
  
  for(int i = 0; i < 1; i++){
	  matrix_multiply_seq(h_a, h_b, h_s, n);
  }

  if( clock_gettime( CLOCK_REALTIME, &end) == -1 ) {
	  perror( "clock gettime" );
      exit( EXIT_FAILURE );
  }
  
  //compute the time in s
  average_seq_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e+9;
  //take the average
  average_seq_time /= 1;
  std::cout << " done." << std::endl;
  std::cout << average_seq_time << "s" << std::endl;
  
  fprintf(stdout, "Results Array:\n");
  for (int i = 0; i < n; i++) {
         //if(h_s[i]==0.0){
         fprintf(stdout, "%.1f ", h_s[i]);
         
         //}
  }
  free(h_a);
  free(h_b);
  free(h_s);
  return 0;
}
