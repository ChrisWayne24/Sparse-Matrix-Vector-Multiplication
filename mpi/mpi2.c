#include <mpi.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <signal.h>
#include <setjmp.h>
#include <assert.h>
#include <inttypes.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

//Opens a matrix store and represents it using Compressed Sparse Row format
void matrix_read(int **row_pointer, int **column_index, float **values, const char *storename, int *nrows, int *ncols, int *nvals) {

    FILE *store = fopen(storename, "r");
    
    if (store == NULL) {
    
        fprintf(stdout, "File error!\n");
        exit(0);
    }
    
    //first line represents no. rows, columns, nnz values
    fscanf(store, "%d %d %d\n", nrows, ncols, nvals);
    
    int *row_pointer_temp = (int*) malloc( sizeof(int) * (*nrows + 1) );
    int *column_index_temp = (int*) malloc( sizeof(int) * (*nvals) );
    float *values_temp = (float*) malloc( sizeof(float) * (*nvals) );
    
    //count appearances of each row for indices of row_pointer array
    int *r_occ= (int *) malloc(sizeof(int) * (*nrows));
    
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

int vec_prepare(int *Column_index_main, int number_expected, int process_id, int numprocs, int nrows, int myVecSize,
	       	int *r_vector_count, int *r_vector_pointer, int **index_vector_remote, float **denseVec) {
	int i;
	int size = ceil((double)nrows / numprocs);
	int localVecIndStart = process_id * size;
	int localVecIndEnd = (process_id + 1) * size;
	int tempRemoteVecCount[numprocs];

	for (i = 0; i < numprocs; i++)
		r_vector_count[i] = 0;
	*index_vector_remote = (int *)malloc(sizeof(int) * number_expected);

	// Select remote vectors from each remote processor
	int number_vector_remote = 0;
	for (i = 0; i < number_expected; i++) {
		if ((Column_index_main[i] < localVecIndStart) || (Column_index_main[i] >= localVecIndEnd)) {
			r_vector_count[Column_index_main[i]/size]++;
			number_vector_remote++;
		}
	}
	r_vector_pointer[0] = 0;
	tempRemoteVecCount[0] = 0;
	for (i = 1; i < numprocs; i++) {
		r_vector_pointer[i] = r_vector_pointer[i-1] + r_vector_count[i-1];  
		tempRemoteVecCount[i]=0;
	}

	*denseVec = (float *)realloc(*denseVec,sizeof(float) * (myVecSize+number_vector_remote));

	int proc = 0;
	for (i = 0; i < number_expected; i++) {
		if ((Column_index_main[i] < localVecIndStart) || (Column_index_main[i] >= localVecIndEnd)) {
			proc = Column_index_main[i]/size;
			(*index_vector_remote)[r_vector_pointer[proc]+tempRemoteVecCount[proc]] = Column_index_main[i];

			Column_index_main[i] = myVecSize + r_vector_pointer[proc] + tempRemoteVecCount[proc];
			tempRemoteVecCount[proc]++;
		}
		else {
			Column_index_main[i] -= (process_id * size); 
		}
	}



	//return total
	return number_vector_remote;
}

//CRS Matrix-Vector multiplication
void csr_spMV(int process_id, int nrows, int *rowptr, int *colind, float *matval, float *vec, float *res) {
	int i, j, count;
	count = 0;
	
	for (i = 0; i < nrows; i++) {
	
		res[i] = 0.0;
		
		for (j = rowptr[i]; j < rowptr[i+1]; j++) {
		
			res[i] += matval[count] * vec[colind[j]];
			count++;
		}
	}
}


int main(int argc, char * argv[]) {
    if (argc != 4) {
    
        fprintf(stdout, "Invalid command, enter:\n1. number of repetitions, 2. print mode (1 or 2), 3. test filename\n");
        exit(0);
    }
    int total_processes, process_id, name_len, p=0, i=0, vecSize, number_expected, vector_size=0, vector_size2=0; 
    float *denseVec = NULL;
    float *matrix_vals = NULL;
    float *result= NULL; 
    float *final_result= NULL;
    float *Vec_Send_Data= NULL;
    int *Column_index_main= NULL;
    int *Row_pointer_main= NULL;
    int *row_pointer= NULL;
    int *column_index= NULL;
    int num_rows;
    int num_cols; 
    int num_vals;
    float *values= NULL;
    float elapsed_time;
    
    int num_repeat = atoi(argv[1]);
    int print_mode = atoi(argv[2]);
    const char *filename = argv[3];
    
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&total_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Get_processor_name(proc_name, &name_len);

    int r_vector_count[total_processes];
    int r_vector_pointer[total_processes];
    int send_vector_count[total_processes];
    int send_vector_pointer[total_processes];
    int send_total = 0;
    int number_vector_remote = 0;
    int *index_vector_remote;
    int *send_vector_index;
    
    //matrix_read(&row_pointer, &column_index, &values, filename, &num_rows, &num_cols, &num_vals);
    
    //Allocate memory for vectors
    float *x;
    float *y;
    
        
    if(process_id == 0){
    
        matrix_read(&row_pointer, &column_index, &values, filename, &num_rows, &num_cols, &num_vals);
        
        x = (float *) malloc(num_rows * sizeof(float));
        y = (float *) malloc(num_rows * sizeof(float));
        for (int i = 0; i < num_rows; i++) {
              x[i] = 1.0;
              y[i] = 0.0;
       }
        
        //Vector sizes to be shared
        vector_size = ceil((double)num_rows / total_processes);
        //possible vector size -> last process
	vector_size2 = num_rows - (total_processes-1)*vector_size;
	
	//Error case
	if (vector_size2 < 0) { 
	
		printf("Invalid number of processors, exit!\n");
		exit(0);
	}
	if (process_id == total_processes-1) {
	
		vector_size = vector_size2; 
	}
	
	// Data size and displacement vectors for every processor
	int Data_size_vector[total_processes];
	int Data_displacement_vector[total_processes];
	
	for (p = 0; p < total_processes-1; p++) { 
	
		Data_size_vector[p] = vector_size;
		Data_displacement_vector[p] = p*vector_size;
		
	} 
	
	Data_size_vector[total_processes-1] = vector_size2;
	Data_displacement_vector[total_processes-1] = (total_processes-1)*vector_size;
	
	// Sparse matrix entry count and displacement for processors
	int entry_count[total_processes]; 
	int entry_displacement[total_processes];
	
	for (p = 0; p < total_processes; p++) {
	
		entry_count[p] = row_pointer[p*vector_size+Data_size_vector[p]] - row_pointer[p*vector_size];
		entry_displacement[p] = row_pointer[p*vector_size];
	}
	
	//Create storage for local vector entries
	denseVec = (float *)malloc(sizeof(float) * vector_size);

	//Pass vector entries to all processors
	MPI_Bcast(&num_rows,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Scatterv(x, Data_size_vector, Data_displacement_vector, MPI_FLOAT, denseVec, vector_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	MPI_Scatter(entry_count,1,MPI_INT,&number_expected,1,MPI_INT,0,MPI_COMM_WORLD);

	Row_pointer_main = (int *)malloc(sizeof(int) * (vector_size+1));
	MPI_Scatterv(row_pointer, Data_size_vector, Data_displacement_vector,MPI_INT, Row_pointer_main,vector_size,MPI_INT,0,MPI_COMM_WORLD);
	Row_pointer_main[vector_size] = number_expected;

	Column_index_main = (int *)malloc(sizeof(int) * number_expected);
	MPI_Scatterv(column_index, entry_count, entry_displacement, MPI_INT, Column_index_main, number_expected, MPI_INT, 0, MPI_COMM_WORLD);
		
	matrix_vals = (float *)malloc(sizeof(float) * number_expected);
	MPI_Scatterv(values, entry_count, entry_displacement, MPI_FLOAT, matrix_vals, number_expected, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
        clock_t start = clock();
	//decide which remote vectors are needed and make local Column Index array point at the correct Vector Data entry
	number_vector_remote = vec_prepare(Column_index_main,number_expected,process_id,total_processes,num_rows,
	vector_size,r_vector_count,r_vector_pointer,&index_vector_remote,&denseVec);

	//Determine load for each processor
	// Each process shares info with other remote vector entries 
	MPI_Alltoall(r_vector_count,1,MPI_INT,send_vector_count,1,MPI_INT,MPI_COMM_WORLD);

	for (p = 0; p < total_processes; p++)
		send_total = send_total + send_vector_count[p];
		
	send_vector_index = (int *)malloc(sizeof(int) * send_total);
	send_vector_pointer[0] = 0;
	
	for (p = 1; p < total_processes; p++)
		send_vector_pointer[p] = send_vector_pointer[p-1] + send_vector_count[p-1]; 
		
	// Each process send to all others index of remote vector entries needed
	MPI_Alltoallv(index_vector_remote,r_vector_count,r_vector_pointer,MPI_INT,send_vector_index,
		send_vector_count,send_vector_pointer,MPI_INT,MPI_COMM_WORLD);
		
	Vec_Send_Data = (float *)malloc(sizeof(float) * send_total);
	
	for (i = 0; i < send_total; i++) 
		Vec_Send_Data[i] = denseVec[send_vector_index[i]];
 
	float *recvVecData = denseVec + vector_size;
	
	//All processors share actual vector entries with each other
	MPI_Alltoallv(Vec_Send_Data,send_vector_count,send_vector_pointer,MPI_FLOAT,recvVecData,
				r_vector_count,r_vector_pointer,MPI_FLOAT,MPI_COMM_WORLD);

	//Results are computed
	result = (float *)malloc(sizeof(float) * vector_size);
	csr_spMV(process_id, vector_size, Row_pointer_main, Column_index_main, matrix_vals, denseVec, result);
	
	clock_t stop = clock();
        elapsed_time = (((float) (stop - start)) / CLOCKS_PER_SEC) * 1000; // in milliseconds
        // Print elapsed time
        printf("\nMPI Run time:  %.4f ms\n", elapsed_time);
        
	//results gathered from all processors
	final_result = (float *)malloc(sizeof(float) * num_rows);
	MPI_Gatherv(result, vector_size, MPI_FLOAT, final_result, Data_size_vector, Data_displacement_vector,
				MPI_FLOAT,0,MPI_COMM_WORLD);
				
	// Creating output filename
	int a = 0; 
	int temp = num_rows;
	while (temp != 0) {
		temp = temp / 10;
		a = a+1;
	}
	char fname[a+5];
	sprintf(fname,"o%d.vec",num_rows);

	//Resultant vector to output file
	FILE *fpout;
	fpout = fopen(fname,"w");
	for (i = 0; i < num_rows; i++)
		fprintf(fpout,"%.15lg\n",final_result[i]);

	fclose(fpout);
	free(final_result);
    
    }
    else{ //FOR other processors
        int *colInd= NULL;
        int *entry_count= NULL;
        int *entry_displacement= NULL;
        int *rowptr= NULL; 
        int *Data_size_vector= NULL;
        int *Data_displacement_vector= NULL;
	float *matVal= NULL;
	float *allVecData= NULL;
	int size;

	MPI_Bcast(&num_rows,1,MPI_INT,0,MPI_COMM_WORLD);
	vector_size = ceil((double)num_rows / total_processes);
	size = vector_size;
	
	if (process_id == total_processes-1)
		vector_size = num_rows - process_id*vector_size;
		
	//Create storage for local vec entries
	denseVec = (float *)malloc(sizeof(float) * vector_size);
	
	//Get vec entries through processor 0
	MPI_Scatterv(x, Data_size_vector, Data_displacement_vector, MPI_FLOAT,
			denseVec,vector_size,MPI_FLOAT,0,MPI_COMM_WORLD);

	MPI_Scatter(entry_count,1,MPI_INT,&number_expected,1,MPI_INT,0,MPI_COMM_WORLD);
	
	// Receive matrix entries through processor 0
	Row_pointer_main = (int *)malloc(sizeof(int) * (vector_size+1));
	MPI_Scatterv(rowptr,Data_size_vector,Data_displacement_vector,MPI_INT,
			Row_pointer_main,vector_size,MPI_INT,0,MPI_COMM_WORLD);
			
	int rowptrOffset = Row_pointer_main[0];
	for (i = 0; i < vector_size; i++)
		Row_pointer_main[i] = Row_pointer_main[i] - rowptrOffset; //offset
		
	Row_pointer_main[vector_size] = number_expected;
		
	Column_index_main = (int *)malloc(sizeof(int) * number_expected);
	MPI_Scatterv(colInd,entry_count,entry_displacement,MPI_INT,
			Column_index_main,number_expected,MPI_INT,0,MPI_COMM_WORLD);

	
	matrix_vals = (float *)malloc(sizeof(float) * number_expected);
	MPI_Scatterv(matVal, entry_count, entry_displacement, MPI_FLOAT,
	    matrix_vals, number_expected, MPI_FLOAT, 0, MPI_COMM_WORLD);
	    
	number_vector_remote = vec_prepare(Column_index_main,number_expected,process_id,total_processes,num_rows,
	    vector_size,r_vector_count,r_vector_pointer,&index_vector_remote,&denseVec);

	MPI_Alltoall(r_vector_count,1,MPI_INT,send_vector_count,1,MPI_INT,MPI_COMM_WORLD);
	
	for (p = 0; p < total_processes; p++) 
		send_total = send_total + send_vector_count[p];
		
	send_vector_index = (int *)malloc(sizeof(int) * send_total);
	send_vector_pointer[0] = 0;
	
	for (p = 1; p < total_processes; p++)
		send_vector_pointer[p] = send_vector_pointer[p-1] + send_vector_count[p-1];
		
	MPI_Alltoallv(index_vector_remote,r_vector_count,r_vector_pointer,MPI_INT,
			send_vector_index,send_vector_count,send_vector_pointer,MPI_INT,MPI_COMM_WORLD);
	
	Vec_Send_Data = (float *)malloc(sizeof(float) * send_total);
	
	for (i = 0; i < send_total; i++) 
		Vec_Send_Data[i] = denseVec[send_vector_index[i]-process_id*size];
		  

	float *recvVecData = denseVec + vector_size;

	MPI_Alltoallv(Vec_Send_Data,send_vector_count,send_vector_pointer,MPI_FLOAT,recvVecData,
			r_vector_count,r_vector_pointer,MPI_FLOAT,MPI_COMM_WORLD);


	result = (float *)malloc(sizeof(float) * vector_size);
	csr_spMV(process_id, vector_size, Row_pointer_main, Column_index_main, matrix_vals, denseVec, result);


	MPI_Gatherv(result, vector_size, MPI_FLOAT, final_result, Data_size_vector, Data_displacement_vector,
				MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    
    
    MPI_Finalize();
    /*free(Row_pointer_main); free(Column_index_main); free(matrix_vals); free(send_vector_index);
    free(Vec_Send_Data); free(result); free(denseVec); free(index_vector_remote);
    free(x); free(y);
    
    free(row_pointer);
    free(column_index);
    free(values);*/
    return 0;
}


