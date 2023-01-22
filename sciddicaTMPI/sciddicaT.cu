#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <mpi.h>
#include "util.hpp"

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define P_R 0.5
#define P_EPSILON 0.001
#define ADJACENT_CELLS 4
#define STRLEN 256

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ((M)[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))] = (value))
#define BUF_GET(M, rows, columns, n, i, j) (M[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))])

// declaring neighbourhood in constant memory in order to provide faster memory access
__constant__ int Xi[] = {0, -1, 0, 0, 1}; // Xj: von Neuman neighborhood row coordinates (see below)
__constant__ int Xj[] = {0, 0, -1, 1, 0}; // Xj: von Neuman neighborhood col coordinates (see below)

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char *path, int &nrows, int &ncols, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata){
    FILE *f;

    if ((f = fopen(path, "r")) == 0){
        printf("%s configuration header file not found\n", path);
        exit(0);
    }

    // Reading the header
    char str[STRLEN];
    fscanf(f, "%s", &str);
    fscanf(f, "%s", &str);
    ncols = atoi(str); // ncols
    fscanf(f, "%s", &str);
    fscanf(f, "%s", &str);
    nrows = atoi(str); // nrows
    fscanf(f, "%s", &str);
    fscanf(f, "%s", &str); // xllcorner = atof(str);  //xllcorner
    fscanf(f, "%s", &str);
    fscanf(f, "%s", &str); // yllcorner = atof(str);  //yllcorner
    fscanf(f, "%s", &str);
    fscanf(f, "%s", &str); // cellsize = atof(str);   //cellsize
    fscanf(f, "%s", &str);
    fscanf(f, "%s", &str);
    nodata = atof(str); // NODATA_value
}

bool loadGrid2D(double *M, int rows, int columns, char *path)
{
    FILE *f = fopen(path, "r");

    if (!f)
    {
        printf("%s grid file not found\n", path);
        exit(0);
    }

    char str[STRLEN];
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            fscanf(f, "%s", str);
            SET(M, columns, i, j, atof(str));
        }
    }
    fclose(f);

    return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path)
{
    FILE *f;
    f = fopen(path, "w");
    if (!f)
        return false;

    char str[STRLEN];
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            sprintf(str, "%f ", GET(M, columns, i, j));
            fprintf(f, "%s ", str);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    return true;
}

double *addLayer2D(int rows, int columns)
{
    double *tmp = (double *)malloc(sizeof(double) * rows * columns);
    if (!tmp)
        return NULL;
    return tmp;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
// not so useful in terms of performances because it is called just once in the initialization phase
__global__ void sciddicaTSimulationInit(int r, int c, double *Sz, double *Sh){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= 0 && j >= 0 && i < r && j < c)
    {
        double z, h;
        h = GET(Sh, c, i, j);

        if (h > 0.0)
        {
            z = GET(Sz, c, i, j);
            SET(Sz, c, i, j, z - h);
        }
    }
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
__global__ void sciddicaTResetFlows(int r, int c, double nodata, double *Sf, int pid, int np){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(pid == 0){
        if (i > 0 && j > 0 && i < r && j < c - 1){
            int t = i * c + j;
            BUF_SET(Sf, r, c, 0, i, j, 0.0);
            BUF_SET(Sf, r, c, 1, i, j, 0.0);
            BUF_SET(Sf, r, c, 2, i, j, 0.0);
            BUF_SET(Sf, r, c, 3, i, j, 0.0);
        }
    }else if(pid == np -1){
        if (j > 0 && i < r-1 && j < c - 1){
            int t = i * c + j;
            BUF_SET(Sf, r, c, 0, i, j, 0.0);
            BUF_SET(Sf, r, c, 1, i, j, 0.0);
            BUF_SET(Sf, r, c, 2, i, j, 0.0);
            BUF_SET(Sf, r, c, 3, i, j, 0.0);
        }
    }
}

__global__ void sciddicaTFlowsComputation(int r, int c, double nodata, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int min_index, max_index;
    if (i > 0 && j > 0 && i < r - 1 && j < c - 1){
        bool eliminated_cells[5] = {false, false, false, false, false};
        bool again;
        int cells_count;
        double average;
        double m;
        double u[5];
        int n;
        double z, h;
        m = GET(Sh, c, i, j) - p_epsilon;
        u[0] = GET(Sz, c, i, j) + p_epsilon;
        z = GET(Sz, c, i + Xi[1], j + Xj[1]);
        h = GET(Sh, c, i + Xi[1], j + Xj[1]);
        u[1] = z + h;
        z = GET(Sz, c, i + Xi[2], j + Xj[2]);
        h = GET(Sh, c, i + Xi[2], j + Xj[2]);
        u[2] = z + h;
        z = GET(Sz, c, i + Xi[3], j + Xj[3]);
        h = GET(Sh, c, i + Xi[3], j + Xj[3]);
        u[3] = z + h;
        z = GET(Sz, c, i + Xi[4], j + Xj[4]);
        h = GET(Sh, c, i + Xi[4], j + Xj[4]);
        u[4] = z + h;

        do{
            again = false;
            average = m;
            cells_count = 0;

            for (n = 0; n < 5; n++){
                if (!eliminated_cells[n]){
                    average += u[n];
                    cells_count++;
                }
            }

            if (cells_count != 0)
                average /= cells_count;

            for (n = 0; n < 5; n++)
                if ((average <= u[n]) && (!eliminated_cells[n])){
                    eliminated_cells[n] = true;
                    again = true;
                }
        } while (again);

        if (!eliminated_cells[1])
            BUF_SET(Sf, r, c, 0, i, j, (average - u[1]) * p_r);
        if (!eliminated_cells[2])
            BUF_SET(Sf, r, c, 1, i, j, (average - u[2]) * p_r);
        if (!eliminated_cells[3])
            BUF_SET(Sf, r, c, 2, i, j, (average - u[3]) * p_r);
        if (!eliminated_cells[4])
            BUF_SET(Sf, r, c, 3, i, j, (average - u[4]) * p_r);
    }
}

__global__ void sciddicaTFlowsComputationHalos(int r, int c, double nodata, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon, double* Sh_halo, double* Sz_halo, double* Sf_halo, bool top_halo){
    int i;
    if(top_halo){
        i = 0;
    }
    else{
        i = r -1;
    }
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int min_index, max_index;
    if (j > 0 && j < c - 1){
        bool eliminated_cells[5] = {false, false, false, false, false};
        bool again;
        int cells_count;
        double average;
        double m;
        double u[5];
        int n;
        double z, h;
        m = GET(Sh, c, i, j) - p_epsilon;
        u[0] = GET(Sz, c, i, j) + p_epsilon;
        if(top_halo){
            z = GET(Sz_halo, c, 0, j);
            h = GET(Sh_halo, c, 0, j);
        }
        else{
            z = GET(Sz, c, i + Xi[1], j + Xj[1]);
            h = GET(Sh, c, i + Xi[1], j + Xj[1]);
        }
        u[1] = z + h;
        z = GET(Sz, c, i + Xi[2], j + Xj[2]);
        h = GET(Sh, c, i + Xi[2], j + Xj[2]);
        u[2] = z + h;
        z = GET(Sz, c, i + Xi[3], j + Xj[3]);
        h = GET(Sh, c, i + Xi[3], j + Xj[3]);
        u[3] = z + h;
        if(!top_halo){
            z = GET(Sz_halo, c, 0, j);
            h = GET(Sh_halo, c, 0, j);
        }
        else{
            z = GET(Sz, c, i + Xi[4], j + Xj[4]);
            h = GET(Sh, c, i + Xi[4], j + Xj[4]);
        }
        u[4] = z + h;

        do{
            again = false;
            average = m;
            cells_count = 0;

            for (n = 0; n < 5; n++){
                if (!eliminated_cells[n]){
                    average += u[n];
                    cells_count++;
                }
            }

            if (cells_count != 0)
                average /= cells_count;

            for (n = 0; n < 5; n++)
                if ((average <= u[n]) && (!eliminated_cells[n])){
                    eliminated_cells[n] = true;
                    again = true;
                }
        } while (again);

        if (!eliminated_cells[1])
            BUF_SET(Sf, r, c, 0, i, j, (average - u[1]) * p_r);
        if (!eliminated_cells[2])
            BUF_SET(Sf, r, c, 1, i, j, (average - u[2]) * p_r);
        if (!eliminated_cells[3])
            BUF_SET(Sf, r, c, 2, i, j, (average - u[3]) * p_r);
        if (!eliminated_cells[4])
            BUF_SET(Sf, r, c, 3, i, j, (average - u[4]) * p_r);
    }
}

__global__ void sciddicaTWidthUpdate(int r, int c, double nodata, double *Sz, double *Sh, double *Sf){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > 0 && j > 0 && i < r - 1 && j < c - 1){
        double h_next = 0;
        h_next = GET(Sh, c, i, j);
        h_next += BUF_GET(Sf, r, c, 3, i + Xi[1], j + Xj[1]) - BUF_GET(Sf, r, c, 0, i, j);
        h_next += BUF_GET(Sf, r, c, 2, i + Xi[2], j + Xj[2]) - BUF_GET(Sf, r, c, 1, i, j);
        h_next += BUF_GET(Sf, r, c, 1, i + Xi[3], j + Xj[3]) - BUF_GET(Sf, r, c, 2, i, j);
        h_next += BUF_GET(Sf, r, c, 0, i + Xi[4], j + Xj[4]) - BUF_GET(Sf, r, c, 3, i, j);

        SET(Sh, c, i, j, h_next);
    }
}
//#define BUF_GET(M, rows, columns, n, i, j) (M[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))])

__global__ void sciddicaTWidthUpdateHalos(int r, int c, double nodata, double *Sz, double *Sh, double *Sf, double * Sf_halo, bool top_halo){
    int i;
    if(top_halo){
        i = 0;
    }
    else{
        i = r -1;
    }
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(j > 0 && j < c - 1){
        double h_next = 0;
        h_next = GET(Sh, c, i, j);
        //top or bottom neighbours are on the "first" row of the halo buffer
        if(top_halo){
            // int index =  3* 1 * c + j + Xj[1];
            // printf("accessing %d\n ", index);
            h_next += BUF_GET(Sf_halo, 1, c, 3, 0, j) - BUF_GET(Sf, r, c, 0, i, j);
        }else{
           h_next += BUF_GET(Sf, r, c, 3, i + Xi[1], j + Xj[1]) - BUF_GET(Sf, r, c, 0, i, j); 
        }
        h_next += BUF_GET(Sf, r, c, 2, i + Xi[2], j + Xj[2]) - BUF_GET(Sf, r, c, 1, i, j);
        h_next += BUF_GET(Sf, r, c, 1, i + Xi[3], j + Xj[3]) - BUF_GET(Sf, r, c, 2, i, j);
        //top or bottom neighbours are on the "first" row of the halo buffer
        if(!top_halo){
            // int index = j + Xj[1];
            // printf("accessing %d\n ", index);
            h_next += BUF_GET(Sf_halo, 1, c, 0, 0, j) - BUF_GET(Sf, r, c, 3, i, j);
        }else{
            h_next += BUF_GET(Sf, r, c, 0, i + Xi[4], j + Xj[4]) - BUF_GET(Sf, r, c, 3, i, j);
        }
        SET(Sh, c, i, j, h_next);
        //printf("Setting SH to %f\n", h_next);
    }
}
// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int np = -1, pid = -1;
    int rows, cols;
    double nodata;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    // if(pid == 0){
    //     for(unsigned i = 0; i < argc; ++i){
    //         printf("Argument %d %s\n", i, argv[i]);
    //     }
    // }
    readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);
    int master_id = 0;
    int r = rows; // r: grid rows
    int c = cols; // c: grid columns
    int chunk;
    int i_start, i_end;
    // considering a ONLY a 1D partition of the domain
    int j_start = 0, j_end = cols;
    int number_of_rows_to_compute;

    // number of bytes of the data that each process has to compute
    unsigned number_of_bytes_to_compute;
    unsigned number_of_halo_bytes = cols * sizeof(double);
    if (pid == np - 1){
        // last processor must get all the remaining rows
        chunk = (rows*cols) / np + (rows*cols) % np;
        i_end = rows - 1;
        number_of_bytes_to_compute = chunk * sizeof(double);

    }
    else{
        chunk = (rows*cols) / np;
        i_end = (pid + 1) * chunk - 1;
        number_of_bytes_to_compute = chunk * sizeof(double);
    }

    number_of_rows_to_compute = ceil(float(chunk) / float(c));
    i_start = pid * chunk;
    double *Sz; // Sz: substate (grid) containing the cells' altitude a.s.l.
    double *Sh; // Sh: substate (grid) containing the cells' flow thickness
    double *Sf; // Sf: 4 substates containing the flows towards the 4 neighs

    double p_r = P_R;                 // p_r: minimization algorithm outflows dumping factor
    double p_epsilon = P_EPSILON;     // p_epsilon: frictional parameter threshold
    int steps = atoi(argv[STEPS_ID]); // steps: simulation steps

    // The adopted von Neuman neighborhood
    // Format: flow_index:cell_label:(row_index,col_index)
    //
    //   cell_label in [0,1,2,3,4]: label assigned to each cell in the neighborhood
    //   flow_index in   [0,1,2,3]: outgoing flow indices in Sf from cell 0 to the others
    //       (row_index,col_index): 2D relative indices of the cells
    //
    //               |0:1:(-1, 0)|
    //   |1:2:( 0,-1)| :0:( 0, 0)|2:3:( 0, 1)|
    //               |3:4:( 1, 0)|
    //
    //

    // only the master process should delcare enough data structures to contain
    // the entire domain of the automaton. This is needed because only the master will print on file
    if (pid == 0){
        Sz = addLayer2D(r, c);                  // Allocates the Sz substate grid
        Sh = addLayer2D(r, c);                  // Allocates the Sh substate grid
        Sf = addLayer2D(ADJACENT_CELLS * r, c); // Allocates the Sf substates grid,
                                                //   having one layer for each adjacent cell

        loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);    // Load Sz from file
        loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]); // Load Sh from file
    }
    else{
        Sz = (double *)malloc(sizeof(double) * chunk);                      // Allocates the Sz substate grid
        Sh = (double *)malloc(sizeof(double) * chunk);                      // Allocates the Sh substate grid
        Sf = (double *)malloc(sizeof(double) * chunk * ADJACENT_CELLS);     // Allocates the Sf substates grid,
                                                                            //   having one layer for each adjacent cell
    }
    // declaring datastructure that will be used on the device
    double *d_Sz; // Sz: substate (grid) containing the cells' altitude a.s.l.
    double *d_Sh; // Sh: substate (grid) containing the cells' flow thickness
    double *d_Sf; // Sf: 4 substates containing the flows towards the 4 neighs
    // int *d_Xi;
    // int *d_Xj;

    int cuda_error = cudaSetDevice(pid);
    if(cuda_error != cudaSuccess){
        printf("%d\n", cuda_error);
        return 1;
    }
    
    // allocate data structures that will contain only the
    // portion of the domain that has to be computed by each thread
    cudaMalloc((void **)&d_Sz, number_of_bytes_to_compute);
    cudaMalloc((void **)&d_Sh, number_of_bytes_to_compute);
    cudaMalloc((void **)&d_Sf, number_of_bytes_to_compute * ADJACENT_CELLS);
    double *d_Sz_top_halo;
    double *d_Sh_top_halo;
    double *d_Sf_top_halo;
    if(pid!=0){
        cudaHostAlloc((void **)&d_Sz_top_halo, number_of_halo_bytes, cudaHostAllocDefault);
        cudaHostAlloc((void **)&d_Sh_top_halo, number_of_halo_bytes, cudaHostAllocDefault);
        cudaHostAlloc((void **)&d_Sf_top_halo, number_of_halo_bytes * ADJACENT_CELLS, cudaHostAllocDefault);
    }
    double *d_Sz_bottom_halo;
    double *d_Sh_bottom_halo;
    double *d_Sf_bottom_halo;
    if(pid != np -1){
        cudaHostAlloc((void **)&d_Sz_bottom_halo, number_of_halo_bytes, cudaHostAllocDefault);
        cudaHostAlloc((void **)&d_Sh_bottom_halo, number_of_halo_bytes, cudaHostAllocDefault);
        cudaHostAlloc((void **)&d_Sf_bottom_halo, number_of_halo_bytes * ADJACENT_CELLS, cudaHostAllocDefault);
    }
    // compute number of blocks given a fixed dimension for the block
    dim3 blockDimension(32, 32);
    dim3 numBlocks(ceil(float(cols) / 32.0), ceil((float(number_of_rows_to_compute)) / 32.0));

    dim3 blockDimensionHalo(32);
    dim3 numBlocksHalo(ceil(float(cols) / 32.0));

    // each process here has already declared the data structures he needs and now
    // worker processed can receive their portion of data to compute and halos
    MPI_Status status;
    
    //send only chunks so that every process runs its init kernel
    if(pid == 0){
        // send data to all workers
        int number_of_elements_to_compute_last_process = chunk + ((rows*cols) % np);
        for (unsigned i = 1; i < np -1; ++i){
            // send chunks
            //printf("Sending Sh\n");
            MPI_Send(&Sh[i * chunk], chunk, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            //printf("Sending Sz\n");
            MPI_Send(&Sz[i * chunk], chunk, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            //printf("Sending Sf\n");
            MPI_Send(&Sf[i * chunk], chunk, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            MPI_Send(&Sf[rows * cols + i * chunk], chunk, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            MPI_Send(&Sf[2 * rows * cols + i * chunk], chunk, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
            MPI_Send(&Sf[3 * rows * cols + i * chunk], chunk, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
        }
        //send larger chunk for last process
        MPI_Send(&Sh[(np -1) * chunk], number_of_elements_to_compute_last_process, MPI_DOUBLE, np -1, np -1, MPI_COMM_WORLD);
        //printf("Sending Sz\n");
        MPI_Send(&Sz[(np -1) * chunk], number_of_elements_to_compute_last_process, MPI_DOUBLE, np -1, np -1, MPI_COMM_WORLD);
        //printf("Sending Sf\n");
        MPI_Send(&Sf[(np -1) * chunk], number_of_elements_to_compute_last_process, MPI_DOUBLE, np -1, np -1, MPI_COMM_WORLD);
        MPI_Send(&Sf[rows * cols + (np -1) * chunk], number_of_elements_to_compute_last_process, MPI_DOUBLE, np -1, np -1, MPI_COMM_WORLD);
        MPI_Send(&Sf[2 * rows * cols + (np -1) * chunk], number_of_elements_to_compute_last_process, MPI_DOUBLE, np -1, np -1, MPI_COMM_WORLD);        
        MPI_Send(&Sf[3 * rows * cols + (np -1) * chunk], number_of_elements_to_compute_last_process, MPI_DOUBLE, np -1, np -1, MPI_COMM_WORLD);
    }
    else{
        // receive data from master
        // receive chunks
        MPI_Recv(Sh, chunk, MPI_DOUBLE, master_id, pid, MPI_COMM_WORLD, &status);
        //printf("Receiving Sz\n");
        MPI_Recv(Sz, chunk, MPI_DOUBLE, master_id, pid, MPI_COMM_WORLD, &status);
        //printf("Receiving Sf\n");
        MPI_Recv(Sf, chunk, MPI_DOUBLE, master_id, pid, MPI_COMM_WORLD, &status);
        MPI_Recv(&Sf[chunk], chunk, MPI_DOUBLE, master_id, pid, MPI_COMM_WORLD, &status);
        MPI_Recv(&Sf[2 * chunk], chunk, MPI_DOUBLE, master_id, pid, MPI_COMM_WORLD, &status);
        MPI_Recv(&Sf[3 * chunk], chunk, MPI_DOUBLE, master_id, pid, MPI_COMM_WORLD, &status);
        
    }
    // copy data to the device
    cudaMemcpy(d_Sz, Sz, number_of_bytes_to_compute, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sh, Sh, number_of_bytes_to_compute, cudaMemcpyHostToDevice);
    //this is all 0 in the first step
    cudaMemcpy(d_Sf, Sf, number_of_bytes_to_compute * ADJACENT_CELLS, cudaMemcpyHostToDevice);

    // Apply the init kernel (elementary process) to the whole domain grid (cellular space)
    sciddicaTSimulationInit<<<numBlocks, blockDimension>>>(number_of_rows_to_compute, c, d_Sz, d_Sh);
 
    util::Timer cl_timer;


    //  simulation loop
    for (int s = 0; s < steps; ++s){

        // Apply the resetFlow kernel to the whole domain
        sciddicaTResetFlows<<<numBlocks, blockDimension>>>(number_of_rows_to_compute, c, nodata, d_Sf, pid, np);
        if(pid == 0){
            //send bottom halo
            MPI_Send(&d_Sh[chunk - cols], cols, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
            MPI_Send(&d_Sz[chunk - cols], cols, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);

            //receive bottom halo
            MPI_Recv(d_Sh_bottom_halo, cols, MPI_DOUBLE, 1, 7, MPI_COMM_WORLD, &status);
            MPI_Recv(d_Sz_bottom_halo, cols, MPI_DOUBLE, 1, 8, MPI_COMM_WORLD, &status);
        }else{
            //receive top halo
            MPI_Recv(d_Sh_top_halo, cols, MPI_DOUBLE, master_id, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(d_Sz_top_halo, cols, MPI_DOUBLE, master_id, 2, MPI_COMM_WORLD, &status);

             //send top halo
            MPI_Send(d_Sh, cols, MPI_DOUBLE, pid -1, 7, MPI_COMM_WORLD);
            MPI_Send(d_Sz, cols, MPI_DOUBLE, pid -1, 8, MPI_COMM_WORLD);
        }
        // Apply the FlowComputation kernel to the whole domain
        sciddicaTFlowsComputation<<<numBlocks, blockDimension>>>(number_of_rows_to_compute, c, nodata, d_Sz, d_Sh, d_Sf, p_r, p_epsilon);
        if(pid != 0)
            sciddicaTFlowsComputationHalos<<<numBlocksHalo, blockDimensionHalo>>>(number_of_rows_to_compute, c, nodata, d_Sz, d_Sh, d_Sf, p_r, p_epsilon, d_Sh_top_halo, d_Sz_top_halo, d_Sf_top_halo, true);
        if(pid != np -1)
            sciddicaTFlowsComputationHalos<<<numBlocksHalo, blockDimensionHalo>>>(number_of_rows_to_compute, c, nodata, d_Sz, d_Sh, d_Sf, p_r, p_epsilon, d_Sh_bottom_halo, d_Sz_bottom_halo, d_Sf_bottom_halo, false);  

        if(pid == 0){
            //send bottom halo
            MPI_Send(&d_Sf[chunk - cols], cols, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD);
            MPI_Send(&d_Sf[2 * chunk - cols], cols, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD);
            MPI_Send(&d_Sf[3 * chunk - cols], cols, MPI_DOUBLE, 1, 5, MPI_COMM_WORLD);
            MPI_Send(&d_Sf[4 * chunk - cols], cols, MPI_DOUBLE, 1, 6, MPI_COMM_WORLD);

            //receive bottom halo
            MPI_Recv(d_Sf_bottom_halo, cols, MPI_DOUBLE, 1, 9, MPI_COMM_WORLD, &status);
            MPI_Recv(&d_Sf_bottom_halo[cols], cols, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, &status);
            MPI_Recv(&d_Sf_bottom_halo[2 * cols], cols, MPI_DOUBLE, 1, 11, MPI_COMM_WORLD, &status);
            MPI_Recv(&d_Sf_bottom_halo[3 * cols], cols, MPI_DOUBLE, 1, 12, MPI_COMM_WORLD, &status);
        
        }
        else{
            //receive top halo
            MPI_Recv(d_Sf_top_halo, cols, MPI_DOUBLE, master_id, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(&d_Sf_top_halo[cols], cols, MPI_DOUBLE, master_id, 4, MPI_COMM_WORLD, &status);
            MPI_Recv(&d_Sf_top_halo[2 * cols], cols, MPI_DOUBLE, master_id, 5, MPI_COMM_WORLD, &status);
            MPI_Recv(&d_Sf_top_halo[3 * cols], cols, MPI_DOUBLE, master_id, 6, MPI_COMM_WORLD, &status);
            //send top halo
            MPI_Send(d_Sf, cols, MPI_DOUBLE, pid -1, 9, MPI_COMM_WORLD);
            MPI_Send(&d_Sf[chunk], cols, MPI_DOUBLE, pid -1, 10, MPI_COMM_WORLD);
            MPI_Send(&d_Sf[2 * chunk], cols, MPI_DOUBLE, pid -1, 11, MPI_COMM_WORLD);
            MPI_Send(&d_Sf[3 * chunk], cols, MPI_DOUBLE, pid -1, 12, MPI_COMM_WORLD);

        }
        // Apply the WidthUpdate mass balance kernel to the whole domain
        sciddicaTWidthUpdate<<<numBlocks, blockDimension>>>(number_of_rows_to_compute, c, nodata, d_Sz, d_Sh, d_Sf);
        if(pid != 0){
            sciddicaTWidthUpdateHalos<<<numBlocksHalo, blockDimensionHalo>>>(number_of_rows_to_compute, c, nodata, d_Sz, d_Sh, d_Sf, d_Sf_top_halo, true);
        }
        if(pid != np -1)
            sciddicaTWidthUpdateHalos<<<numBlocksHalo, blockDimensionHalo>>>(number_of_rows_to_compute, c, nodata, d_Sz, d_Sh, d_Sf, d_Sf_bottom_halo, false);
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
   
    // copy data back to the host
    cudaMemcpy(Sz, d_Sz, number_of_bytes_to_compute, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sh, d_Sh, number_of_bytes_to_compute, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sf, d_Sf, number_of_bytes_to_compute * ADJACENT_CELLS, cudaMemcpyDeviceToHost);
    //receive computed chunks from all workers
    if(pid == 0){
        int number_of_elements_to_compute_last_process = chunk + ((rows*cols) % np);
        for (unsigned i = 1; i < np -1; ++i){
            // receive data from workers
            // receive chunks
            MPI_Recv(&Sh[i * chunk], chunk, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
            //MPI_Recv(&Sz[i * chunk], chunk, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
            //MPI_Recv(&Sf[i * chunk * ADJACENT_CELLS], chunk * ADJACENT_CELLS, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
        }

        MPI_Recv(&Sh[(np -1) * chunk], number_of_elements_to_compute_last_process, MPI_DOUBLE, np -1, np -1, MPI_COMM_WORLD, &status);
        //MPI_Recv(&Sz[(np -1) * chunk], number_of_elements_to_compute_last_process, MPI_DOUBLE, np -1, np -1, MPI_COMM_WORLD, &status);
        //MPI_Recv(&Sf[(np -1) * chunk * ADJACENT_CELLS], number_of_elements_to_compute_last_process * ADJACENT_CELLS, MPI_DOUBLE, np -1, np -1, MPI_COMM_WORLD, &status);
        // for(unsigned i = 0; i < number_of_rows_to_compute; ++i){
        //     for(unsigned j = 0; j< cols; ++j){
        //         printf("%f ", Sh[i*cols + j]);
        //     }
        //     printf("\n");
        // }
    }
    else{
        // send chunks
        MPI_Send(Sh, chunk, MPI_DOUBLE, master_id, pid, MPI_COMM_WORLD);

        //MPI_Send(&Sz[0], chunk, MPI_DOUBLE, master_id, pid, MPI_COMM_WORLD);
        //MPI_Send(&Sf[0], chunk * ADJACENT_CELLS, MPI_DOUBLE, master_id, pid, MPI_COMM_WORLD);
    }


    if(pid == 0){
        double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
        printf("Elapsed time: %lf [s]\n", cl_time);

        saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]); // Save Sh to file
    }

    printf("Releasing memory...\n");
    delete[] Sz;
    delete[] Sh;
    delete[] Sf;
    // delete[] Sh_top_halo;
    // delete[] Sz_top_halo;
    // delete[] Sf_top_halo;
    // delete[] Sh_bottom_halo;
    // delete[] Sz_bottom_halo;
    // delete[] Sf_bottom_halo;
    // deleting memory on GPU
    cudaFree(d_Sz);
    cudaFree(d_Sh);
    cudaFree(d_Sf);
    if(pid != 0){
        cudaFree(d_Sh_top_halo);
        cudaFree(d_Sz_top_halo);
        cudaFree(d_Sf_top_halo);
    }
    if(pid != np -1){
        cudaFree(d_Sh_bottom_halo);
        cudaFree(d_Sz_bottom_halo);
        cudaFree(d_Sf_bottom_halo);
    }
    MPI_Finalize();
    return 0;
}
