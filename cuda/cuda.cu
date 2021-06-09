#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define f_size sizeof(float)
#define i_size sizeof(int)

//Opens a matrix store and represents it using Compressed Sparse Row format
void matrix_read(int **row_pointer, int **column_index, float **values, const char *storename, int *nrows, int *ncols, int *nvals) {

    FILE *store = fopen(storename, "r");
    
    if (store == NULL) {
    
        fprintf(stdout, "File error!\n");
        exit(0);
    }
    
    //first line represents no. rows, columns, nnz values
    fscanf(store, "%d %d %d\n", nrows, ncols, nvals);
    
    int *row_pointer_temp = (int*) malloc( i_size * (*nrows + 1) );
    int *column_index_temp = (int*) malloc( i_size * (*nvals) );
    float *values_temp = (float*) malloc( f_size * (*nvals) );
    
    //count appearances of each row for indices of row_pointer array
    int *r_occ= (int *) malloc(i_size * (*nrows));
    
    for (int i = 0; i < *nrows; i++) {
        r_occ[i] = 0;
    }
    
    int r;
    int c;
    float v;
    
    //C format
    while (fscanf(store, "%d %d %f\n", &r, &c, &v) != EOF) {
        r = r-1;
        c = c-1;
        r_occ[r] = r_occ[r] + 1;
    }
    
    //Fill row_pointer
    int idx = 0;
    
    for (int i = 0; i < *nrows; i++) {
    
        row_pointer_temp[i] = idx;
        idx += r_occ[i];
    }
    
    row_pointer_temp[*nrows] = *nvals;
    free(r_occ);
    
    //Rewind to beginning of file store
    rewind(store);
    
    //Capture column indices and values
    for (int i = 0; i < *nvals; i++) {
    
        column_index_temp[i] = -1;
    }
    
    fscanf(store, "%d %d %d\n", nrows, ncols, nvals);
    int i = 0;
    
    while (fscanf(store, "%d %d %f\n", &r, &c, &v) != EOF) {
    
        r = r - 1;
        c = c - 1;
        
        //Get right index using row information and index i
        while (column_index_temp[i + row_pointer_temp[r]] != -1) {
        
            i++;
        }
        
        column_index_temp[i + row_pointer_temp[r]] = c;
        values_temp[i + row_pointer_temp[r]] = v;
        i = 0;
    }
    
    fclose(store);
    
    *row_pointer = row_pointer_temp;
    *column_index = column_index_temp;
    *values = values_temp;
}

//SpMV kernel using CSR 
__global__ void csr_spMV(const int *row_pointer, const int *column_index, const float *values, const int nrows, const float *x, float *y) {
    // Uses a grid-stride loop to perform dot product
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int a = i ; a < nrows; a = a + (blockDim.x * gridDim.x)) {
    
        float prod = 0;
        const int row_start = row_pointer[a];
        const int row_end = row_pointer[a + 1];
        
        for (int b = row_start; b < row_end; b++) {
        
            prod += values[b] * x[ column_index[b] ];
            
        }
        
        y[a] = (float)prod;
    }
 }
 
int main(int argc, const char * argv[]) {
    if (argc != 5) {
    
        fprintf(stdout, "Invalid, enter:\n1. number of threads, 2. number of repetitions, 3. print mode (1 or 2), 4. test storename\n");
        exit(0);
    }
    
    int *row_pointer, *column_index, nrows, ncols, nvals, numSMs;;
    float *values;
    
    int nthread = atoi(argv[1]);
    int nrepeat = atoi(argv[2]);
    int pmode = atoi(argv[3]);
    const char *storename = argv[4];
    
    matrix_read(&row_pointer, &column_index, &values, storename, &nrows, &ncols, &nvals);
    
    float *x = (float *) malloc(nrows * f_size);
    float *y = (float *) malloc(nrows * f_size);
    
    for (int i = 0; i < nrows; i++) {
    
        x[i] = 1.0;
        y[i] = 0.0;
    }
    
    if (pmode == 1) {
        // val store
        fprintf(stdout, "Value array:\n");
        
        for (int j = 0; j < nvals; j++) {
        
            fprintf(stdout, "%.2f ", values[j]);
            
        }
        
        // Column Indices Array
        fprintf(stdout, "\n\nColumn Indices Array:\n");
        
        for (int j = 0; j < nvals; j++) {
        
            fprintf(stdout, "%d ", column_index[j]);
            
        }
        
        // row pointer store
        fprintf(stdout, "\n\nRow pointer array:\n");
        
        for (int j = 0; j < (nrows + 1); j++) {
        
            fprintf(stdout, "%d ", row_pointer[j]);
            
        }
        
        fprintf(stdout, "\n\nDense vector:\n");
        
        for (int j = 0; j < nrows; j++) {
        
            fprintf(stdout, "%.1f ", x[j]);
            
        }
        
        fprintf(stdout, "\n\nReturned vector:\n");
    }
    
    // Memory allocation
    int *device_row_pointer, *device_column_index;
    float *device_vals, *device_Vx, *device_Vy;
    
    cudaMalloc((void**) &device_row_pointer, (nrows + 1) * i_size);
    cudaMalloc((void**) &device_column_index, nvals * i_size);
    cudaMalloc((void**) &device_vals, nvals * f_size);
    cudaMalloc((void**) &device_Vx, nrows * f_size);
    cudaMalloc((void**) &device_Vy, nrows * f_size);
    
    //SMs in device
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    
    // move data from host to device
    cudaMemcpy(device_row_pointer, row_pointer, (nrows + 1) * i_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_column_index, column_index, nvals * i_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vals, values, nvals * f_size, cudaMemcpyHostToDevice);
    
    // Time kernel
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int i = 0; i < nrepeat; i++) {
    
        cudaMemcpy(device_Vx, x, nrows * f_size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_Vy, y, nrows * f_size, cudaMemcpyHostToDevice);
        
        //kernel
        csr_spMV<<<32 * numSMs, nthread>>>(device_row_pointer, device_column_index, device_vals, nrows, device_Vx, device_Vy);
        
        //move result from device to host
        cudaMemcpy(y, device_Vy, nrows * f_size, cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < nrows; i++) {
        
            x[i] = (float)y[i];
            y[i] = 0.0;
        }
    }
    
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    //Resultant vector
    if (pmode == 1 || pmode == 2) {
    
        for (int j = 0; j < nrows; j++) {
        
            fprintf(stdout, "%.2f\n", x[j]);
        }
        fprintf(stdout, "\n");
    }
    
    // Print time taken
    printf("\nParallel Run time:  %.4f ms\n", elapsed_time);
    
    // Free memory
    cudaFree(device_row_pointer);
    cudaFree(device_column_index);
    cudaFree(device_vals);
    cudaFree(device_Vx);
    cudaFree(device_Vy);
    
    free(row_pointer);
    free(column_index);
    free(values);
    
    return 0;
}
