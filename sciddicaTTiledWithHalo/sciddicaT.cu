#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "util.hpp"

// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define NEIGHBOURHOOD_WIDTH 3
#define TILE_SIZE 8
// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )


//declaring neighbourhood in constant memory in order to provide faster memory access
__constant__ int Xi[] = {0, -1,  0,  0,  1};// Xj: von Neuman neighborhood row coordinates (see below)
__constant__ int Xj[] = {0,  0, -1,  1,  0};// Xj: von Neuman neighborhood col coordinates (see below)

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(const char* path, int &nrows, int &ncols, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata){
    #define STRLEN 256
    FILE* f;
  
    if ( (f = fopen(path,"r") ) == 0){
        printf("%s configuration header file not found\n", path);
        exit(0);
    }

    //Reading the header
    char str[STRLEN];
    fscanf(f,"%s",&str); fscanf(f,"%s",&str); ncols = atoi(str);      //ncols
    fscanf(f,"%s",&str); fscanf(f,"%s",&str); nrows = atoi(str);      //nrows
    fscanf(f,"%s",&str); fscanf(f,"%s",&str); //xllcorner = atof(str);  //xllcorner
    fscanf(f,"%s",&str); fscanf(f,"%s",&str); //yllcorner = atof(str);  //yllcorner
    fscanf(f,"%s",&str); fscanf(f,"%s",&str); //cellsize = atof(str);   //cellsize
    fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);     //NODATA_value 
}

bool loadGrid2D(double *M, int rows, int columns, const char *path){
    #define STRLEN 256
    FILE *f = fopen(path, "r");

    if(!f) {
        printf("%s grid file not found\n", path);
        exit(0);
    }

    char str[STRLEN];
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            fscanf(f, "%s", str);
            SET(M, columns, i, j, atof(str));
        }
    }
    fclose(f);

    return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, const char *path){
    #define STRLEN 256
    FILE *f;
    f = fopen(path, "w");
    if(!f)
        return false;

    char str[STRLEN];
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            sprintf(str, "%f ", GET(M, columns, i, j));
            fprintf(f, "%s ", str);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    return true;
}

double* addLayer2D(int rows, int columns){
    double *tmp = (double *)malloc(sizeof(double) * rows * columns);
    if(!tmp)
        return NULL;
    return tmp;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
//not so useful in terms of performances because it is called just once in the initialization phase
__global__  void sciddicaTSimulationInit(int r, int c, double* Sz, double* Sh){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= 0 && j >= 0 && i < r && j < c){
        double z, h;
        h = GET(Sh, c, i, j);

        if (h > 0.0){
            z = GET(Sz, c, i, j);
            SET(Sz, c, i, j, z - h);
        }
    }
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
__global__ void sciddicaTResetFlows(int r, int c, double nodata, double* Sf){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(i > 0 && j > 0 && i < r-1 && j < c-1){
        BUF_SET(Sf, r, c, 0, i, j, 0.0);
        BUF_SET(Sf, r, c, 1, i, j, 0.0);
        BUF_SET(Sf, r, c, 2, i, j, 0.0);
        BUF_SET(Sf, r, c, 3, i, j, 0.0);
    }
}

__global__ void sciddicaTFlowsComputation(int r, int c, double& nodata, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon){
    //determining row and col indexes that each thread has to compute
    int i = TILE_SIZE * blockIdx.y + threadIdx.y;
    int j = TILE_SIZE * blockIdx.x + threadIdx.x;
    const int dim_buffers =  22 + NEIGHBOURHOOD_WIDTH -1;
    //declaring buffers in shared memory 
    __shared__ double Sz_shared[dim_buffers * dim_buffers];
    __shared__ double Sh_shared[dim_buffers * dim_buffers];
    //determining the indexes of the element that each thread has to load in shared memory
    int i_halo = i - NEIGHBOURHOOD_WIDTH/2;
    int j_halo = j - NEIGHBOURHOOD_WIDTH/2;
    if(i_halo >= 0 && i_halo < r && j_halo >= 0 && j_halo < c){
        Sz_shared[threadIdx.y * dim_buffers + threadIdx.x] = Sz[i_halo * c + j_halo];
        Sh_shared[threadIdx.y * dim_buffers + threadIdx.x] = Sh[i_halo * c + j_halo];
    }
    else{
        //ghost cells are not needed since boundaries are not computed
    }
    __syncthreads();

    if(i > 0 && i < r -1 && j > 0 && j < c -1 &&  threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE){
        bool eliminated_cells[5] = {false, false, false, false, false};
        bool again;
        int cells_count;
        double average;
        double m;
        double u[5];
        int n;
        double z, h;
        m = GET(Sh_shared, dim_buffers, 1 + threadIdx.y, 1 + threadIdx.x) - p_epsilon;
        u[0] = GET(Sz_shared, dim_buffers, 1 + threadIdx.y, 1 + threadIdx.x) + p_epsilon;
        z = GET(Sz_shared, dim_buffers, 1 + threadIdx.y + Xi[1], 1 + threadIdx.x + Xj[1]);
        h = GET(Sh_shared, dim_buffers, 1 + threadIdx.y + Xi[1], 1 + threadIdx.x + Xj[1]);
        u[1] = z + h;                                         
        z = GET(Sz_shared, dim_buffers, 1 + threadIdx.y + Xi[2], 1 + threadIdx.x + Xj[2]);
        h = GET(Sh_shared, dim_buffers, 1 + threadIdx.y + Xi[2], 1 + threadIdx.x + Xj[2]);
        u[2] = z + h;                                         
        z = GET(Sz_shared, dim_buffers, 1 + threadIdx.y + Xi[3], 1 + threadIdx.x + Xj[3]);
        h = GET(Sh_shared, dim_buffers, 1 + threadIdx.y + Xi[3], 1 + threadIdx.x + Xj[3]);
        u[3] = z + h;                                         
        z = GET(Sz_shared, dim_buffers, 1 + threadIdx.y + Xi[4], 1 + threadIdx.x + Xj[4]);
        h = GET(Sh_shared, dim_buffers, 1 + threadIdx.y + Xi[4], 1 + threadIdx.x + Xj[4]);
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
        }while(again);

        if (!eliminated_cells[1]) BUF_SET(Sf, r, c, 0, i, j, (average - u[1]) * p_r);
        if (!eliminated_cells[2]) BUF_SET(Sf, r, c, 1, i, j, (average - u[2]) * p_r);
        if (!eliminated_cells[3]) BUF_SET(Sf, r, c, 2, i, j, (average - u[3]) * p_r);
        if (!eliminated_cells[4]) BUF_SET(Sf, r, c, 3, i, j, (average - u[4]) * p_r);
    }
}

__global__ void sciddicaTWidthUpdate(int r, int c, double& nodata, double *Sz, double *Sh, double *Sf){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(i > 0 && j > 0 && i < r-1 && j < c-1){
        double h_next = 0;
        h_next = GET(Sh, c, i, j);
        h_next += BUF_GET(Sf, r, c, 3, i+Xi[1], j+Xj[1]) - BUF_GET(Sf, r, c, 0, i, j);
        h_next += BUF_GET(Sf, r, c, 2, i+Xi[2], j+Xj[2]) - BUF_GET(Sf, r, c, 1, i, j);
        h_next += BUF_GET(Sf, r, c, 1, i+Xi[3], j+Xj[3]) - BUF_GET(Sf, r, c, 2, i, j);
        h_next += BUF_GET(Sf, r, c, 0, i+Xi[4], j+Xj[4]) - BUF_GET(Sf, r, c, 3, i, j);

        SET(Sh, c, i, j, h_next);
    }
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(const int argc, char* const*argv){
    //declaring constants in the main so that kernel threads will not have them in their memory
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
    

    int rows, cols;
    double nodata;
    readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

    int r = rows;                  // r: grid rows
    int c = cols;                  // c: grid columns
    double *Sz;                    // Sz: substate (grid) containing the cells' altitude a.s.l.
    double *Sh;                    // Sh: substate (grid) containing the cells' flow thickness
    double *Sf;                    // Sf: 4 substates containing the flows towards the 4 neighs
    double p_r = P_R;              // p_r: minimization algorithm outflows dumping factor
    double p_epsilon = P_EPSILON;  // p_epsilon: frictional parameter threshold
    int steps = atoi(argv[STEPS_ID]); //steps: simulation steps

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

    Sz = addLayer2D(r, c);                 // Allocates the Sz substate grid
    Sh = addLayer2D(r, c);                 // Allocates the Sh substate grid
    Sf = addLayer2D(ADJACENT_CELLS* r, c); // Allocates the Sf substates grid, 
                                            //   having one layer for each adjacent cell

    loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);   // Load Sz from file
    loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]);// Load Sh from file
    
    //declaring datastructure that will be used on the device
    double *d_Sz;                    // Sz: substate (grid) containing the cells' altitude a.s.l.
    double *d_Sh;                    // Sh: substate (grid) containing the cells' flow thickness
    double *d_Sf;                    // Sf: 4 substates containing the flows towards the 4 neighs
    unsigned numberOfBytes = rows * cols * sizeof(double);
    
    cudaMalloc((void**) &d_Sz, numberOfBytes);
    cudaMalloc((void**) &d_Sh, numberOfBytes);
    cudaMalloc((void**) &d_Sf, numberOfBytes * ADJACENT_CELLS);

    //compute number of blocks given a fixed dimension for the block
    dim3 blockDimensionFlowsComputation(TILE_SIZE+NEIGHBOURHOOD_WIDTH -1, TILE_SIZE + NEIGHBOURHOOD_WIDTH -1);
    dim3 numBlocksFlowsComputation(ceil(float(cols) / float(TILE_SIZE)), ceil(float(rows) / float(TILE_SIZE)));
    
    dim3 blockDimension(32, 32);
    dim3 numBlocks(ceil(float(cols) / 32.0), ceil(float(rows) / 32.0));
    //copy data to the device
    cudaMemcpy(d_Sz, Sz, numberOfBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sh, Sh, numberOfBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sf, Sf, numberOfBytes * ADJACENT_CELLS, cudaMemcpyHostToDevice);

    // Apply the init kernel (elementary process) to the whole domain grid (cellular space)
    sciddicaTSimulationInit<<<numBlocks, blockDimension>>>(r, c, d_Sz, d_Sh);
    util::Timer cl_timer;
    // simulation loop
    for (int s = 0; s < steps; ++s){
        // Apply the resetFlow kernel to the whole domain
        sciddicaTResetFlows<<<numBlocks, blockDimension>>>(r, c, nodata, d_Sf);
        // Apply the FlowComputation kernel to the whole domain
        sciddicaTFlowsComputation<<<numBlocksFlowsComputation, blockDimensionFlowsComputation>>>(r, c, nodata, d_Sz, d_Sh, d_Sf, p_r, p_epsilon);
        // Apply the WidthUpdate mass balance kernel to the whole domain
        sciddicaTWidthUpdate<<<numBlocks, blockDimension>>>(r, c, nodata, d_Sz, d_Sh, d_Sf);
    }
    //copy data back to the host
    cudaMemcpy(Sz, d_Sz, numberOfBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sf, d_Sf, numberOfBytes * ADJACENT_CELLS, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sh, d_Sh, numberOfBytes, cudaMemcpyDeviceToHost);
    
    double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
    printf("Elapsed time: %lf [s]\n", cl_time);

    saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);// Save Sh to file

    printf("Releasing memory...\n");
    delete[] Sz;
    delete[] Sh;
    delete[] Sf;
    //deleting memory on GPU
    cudaFree(d_Sz);
    cudaFree(d_Sh);
    cudaFree(d_Sf);
    return 0;
}
