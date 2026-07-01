//===============================================================================
// Program : test.c
// Purpose : Find local and global extremas in a 3D grid
//
// Compile : mpicc -o test.x test.c -O3
//===============================================================================
/*===============================LIBRARY INCLUDE FILES===========================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include <math.h>

int PX, PY, PZ, NX, NY, NZ, NC;
FILE *fp = NULL;
// DATA: Convert linear index to 3D coordinates
void linearToCoord(int linearIndex, int *x, int *y, int *z) {
    *z = linearIndex / (NX * NY);
    int remainder = linearIndex % (NX * NY);
    *y = remainder / NX;
    *x = remainder % NX;
}

// DATA: Convert 3D coordinates to linear index
int coordToLinear(int x, int y, int z) {
    return x + y * NX + z * NX * NY;
}

// DATA -> RANK: Determine which process owns a specific coordinate
int coordToRank(int x, int y, int z) {
    int process_x = x / (NX / PX);
    int process_y = y / (NY / PY);
    int process_z = z / (NZ / PZ);
    return process_x + process_y * PX + process_z * PX * PY;
}

int coordToRankU(int x, int y, int z) {
    // Calculate which process owns x
    int div_x = NX / PX;
    int rem_x = NX % PX;
    int process_x;
    if (x < (div_x + 1) * rem_x) {
        // In the region of processes with div_x + 1 elements
        process_x = x / (div_x + 1);
    } else {
        // In the region of processes with div_x elements
        process_x = rem_x + (x - (div_x + 1) * rem_x) / div_x;
    }
    
    // Similar calculations for y and z dimensions
    int div_y = NY / PY;
    int rem_y = NY % PY;
    int process_y;
    if (y < (div_y + 1) * rem_y) {
        process_y = y / (div_y + 1);
    } else {
        process_y = rem_y + (y - (div_y + 1) * rem_y) / div_y;
    }
    
    int div_z = NZ / PZ;
    int rem_z = NZ % PZ;
    int process_z;
    if (z < (div_z + 1) * rem_z) {
        process_z = z / (div_z + 1);
    } else {
        process_z = rem_z + (z - (div_z + 1) * rem_z) / div_z;
    }
    
    return process_x + process_y * PX + process_z * PX * PY;
}

// PROCESS: Convert process rank to its position in the process grid
void rankToProcessCoord(int rank, int *process_x, int *process_y, int *process_z) {
    *process_x = rank % PX;
    *process_y = (rank / PX) % PY;
    *process_z = rank / (PX * PY);
}

int CoordToRankProcess(int x, int y, int z) {
    if(x < 0 || x >= PX || y < 0 || y >= PY || z < 0 || z >= PZ)
        return -1;
    return x + y * PX + z * PX * PY;
}

// PROCESS: Convert linear index to rank
int idxToRank(int i) {
    int x, y, z;
    linearToCoord(i, &x, &y, &z);
    return coordToRank(x, y, z);
}

// Function to compute 3D indices in local array with ghost cells
int localIndex(int x, int y, int z, int t, int lnx, int lny, int lnz) {
    return x + (y) * lnx + (z) * lnx * lny + t * lnx * lny * lnz;
}

// Function to determine if a point is a local minimum
// Returns 1 if it's a local minimum, 0 otherwise
int isLocalMinimum(float* local_data, int x, int y, int z, int t, int lnx, int lny, int lnz) {

    // Check all 6 neighbors
    float value = local_data[localIndex(x, y, z, t, lnx, lny, lnz)];
    int offsets[6][3] = {
        {+1, 0, 0},
        {-1, 0, 0},
        {0, +1, 0},
        {0, -1, 0},
        {0, 0, +1},
        {0, 0, -1}
    };

    for(int o = 0; o < 6; o++){
        int dx = offsets[o][0];
        int dy = offsets[o][1];
        int dz = offsets[o][2];
        float n_value = local_data[localIndex(x+dx, y+dy, z+dz, t, lnx, lny, lnz)];
        if(fabs(n_value - FLT_MAX) < FLT_EPSILON) continue; // Skip ghost cells
        if(value >= n_value) return 0; // Not a minimum
    }
    
    return 1; // It's a local minimum
}

// Similarly for local maximum
int isLocalMaximum(float* local_data, int x, int y, int z, int t, int lnx, int lny, int lnz) {
    
    float value = local_data[localIndex(x, y, z, t, lnx, lny, lnz)];
    int offsets[6][3] = {
        {+1, 0, 0},
        {-1, 0, 0},
        {0, +1, 0},
        {0, -1, 0},
        {0, 0, +1},
        {0, 0, -1}
    };

    for(int o = 0; o < 6; o++){
        int dx = offsets[o][0];
        int dy = offsets[o][1];
        int dz = offsets[o][2];
        float n_value = local_data[localIndex(x+dx, y+dy, z+dz, t, lnx, lny, lnz)];
        if(fabs(n_value - FLT_MAX) < FLT_EPSILON) continue; // Skip ghost cells
        if(value <= n_value) return 0; // Not a maximum
    }

    return 1; // It's a local maximum
}

// Find local minima and maxima for a specific time step
void findLocalExtrema(float *local_data, int t, int lnx, int lny, int lnz,
                      int local_nx, int local_ny, int local_nz,
                      int *min_count, int *max_count, float *global_min, float *global_max)
{
    *min_count = 0;
    *max_count = 0;
    *global_min = FLT_MAX;
    *global_max = -FLT_MAX;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int proc_x, proc_y, proc_z;
    rankToProcessCoord(rank, &proc_x, &proc_y, &proc_z);
    for (int z = 1; z <= local_nz ; z++){
        for (int y = 1; y <= local_ny ; y++){
            for (int x = 1; x <= local_nx; x++){
                float value = local_data[localIndex(x, y, z, t, lnx, lny, lnz)];
                // Update global extrema
                if (value < *global_min)
                    *global_min = value;
                if (value > *global_max)
                    *global_max = value;

                // Check for local extrema
                if (isLocalMinimum(local_data, x, y, z, t, lnx, lny, lnz)){
                    (*min_count)++;
                    int x_, y_, z_;
                    x_ = proc_x * local_nx + x;
                    y_ = proc_y * local_ny + y;
                    z_ = proc_z * local_nz + z;
                    //get the neighbouring values
                    float n1 = local_data[localIndex(x+1, y, z, t, lnx, lny, lnz)];
                    float n2 = local_data[localIndex(x-1, y, z, t, lnx, lny, lnz)];
                    float n3 = local_data[localIndex(x, y+1, z, t, lnx, lny, lnz)];
                    float n4 = local_data[localIndex(x, y-1, z, t, lnx, lny, lnz)];
                    float n5 = local_data[localIndex(x, y, z+1, t, lnx, lny, lnz)];
                    float n6 = local_data[localIndex(x, y, z-1, t, lnx, lny, lnz)];
                    fprintf(fp, "Local minimum (%d, %d, %d, %d) with value %.6f with neigbouring values being %f, %f, %f, %f, %f, %f\n", x_, y_, z_,t, value,n1,n2,n3,n4,n5,n6);
                }

                if (isLocalMaximum(local_data, x, y, z, t, lnx, lny, lnz)){
                    (*max_count)++;
                    int x_, y_, z_;
                    x_ = proc_x * local_nx + x;
                    y_ = proc_y * local_ny + y;
                    z_ = proc_z * local_nz + z;
                    float n1 = local_data[localIndex(x+1, y, z, t, lnx, lny, lnz)];
                    float n2 = local_data[localIndex(x-1, y, z, t, lnx, lny, lnz)];
                    float n3 = local_data[localIndex(x, y+1, z, t, lnx, lny, lnz)];
                    float n4 = local_data[localIndex(x, y-1, z, t, lnx, lny, lnz)];
                    float n5 = local_data[localIndex(x, y, z+1, t, lnx, lny, lnz)];
                    float n6 = local_data[localIndex(x, y, z-1, t, lnx, lny, lnz)];
                    fprintf(fp, "Local maximum (%d, %d, %d, %d) with value %.6f with neigbouring values being %f, %f, %f, %f, %f, %f\n", x_, y_, z_,t, value, n1, n2, n3, n4, n5, n6);
                }
            }
        }
    }
}

// Add ownership check to determine which process "owns" each boundary point
int isOwnedByThisProcess(int x, int y, int z, int local_nx, int local_ny, int local_nz,
                         int proc_x, int proc_y, int proc_z)
{
    // Simple ownership rule: process with even rank along each dimension
    // owns the boundary in that dimension
    if (x == 0 && proc_x > 0 && proc_x % 2 != 0)
        return 0;
    if (x == local_nx - 1 && proc_x < PX - 1 && proc_x % 2 != 0)
        return 0;
    if (y == 0 && proc_y > 0 && proc_y % 2 != 0)
        return 0;
    if (y == local_ny - 1 && proc_y < PY - 1 && proc_y % 2 != 0)
        return 0;
    if (z == 0 && proc_z > 0 && proc_z % 2 != 0)
        return 0;
    if (z == local_nz - 1 && proc_z < PZ - 1 && proc_z % 2 != 0)
        return 0;
    return 1;
}

void updateLocalExtrema(float *local_data, int t, int lnx, int lny, int lnz,
                      int local_nx, int local_ny, int local_nz,
                      int *min_count, int *max_count, float *global_min, float *global_max)
{

    //doing only for edge cases
    int dz[2] = {0, local_nz - 1};
    int dy[2] = {0, local_ny - 1};
    int dx[2] = {0, local_nx - 1};

    for(int z = 0; z < 2; z++){
        for(int y = 0; y < local_ny; y++){
            for(int x = 0; x < local_nx; x++){
                if (!isOwnedByThisProcess(x, y, dz[z], local_nx, local_ny, local_nz, PX, PY, PZ))    continue;
                float value = local_data[localIndex(x, y, dz[z], t, lnx, lny, lnz)];
                if(value < *global_min)
                    *global_min = value;
                if(value > *global_max)
                    *global_max = value;
                if(isLocalMinimum(local_data, x, y, dz[z], t, lnx, lny, lnz))
                    (*min_count)++;
                if(isLocalMaximum(local_data, x, y, dz[z], t, lnx, lny, lnz))
                    (*max_count)++;
            }
        }
    }

    for(int y = 0; y < 2; y++){
        for(int z = 0; z < local_nz; z++){
            for(int x = 0; x < local_nx; x++){
                if (!isOwnedByThisProcess(x, dy[y], z, local_nx, local_ny, local_nz, PX, PY, PZ))    continue;
                float value = local_data[localIndex(x, dy[y], z, t, lnx, lny, lnz)];
                if(value < *global_min)
                    *global_min = value;
                if(value > *global_max)
                    *global_max = value;
                if(isLocalMinimum(local_data, x, dy[y], z, t, lnx, lny, lnz))
                    (*min_count)++;
                if(isLocalMaximum(local_data, x, dy[y], z, t, lnx, lny, lnz))
                    (*max_count)++;
            }
        }
    }

    for(int x = 0; x < 2; x++){
        for(int y = 0; y < local_ny; y++){
            for(int z = 0; z < local_nz; z++){
                if (!isOwnedByThisProcess(dx[x], y, z, local_nx, local_ny, local_nz, PX, PY, PZ))    continue;
                float value = local_data[localIndex(dx[x], y, z, t, lnx, lny, lnz)];
                if(value < *global_min)
                    *global_min = value;
                if(value > *global_max)
                    *global_max = value;
                if(isLocalMinimum(local_data, dx[x], y, z, t, lnx, lny, lnz))
                    (*min_count)++;
                if(isLocalMaximum(local_data, dx[x], y, z, t, lnx, lny, lnz))
                    (*max_count)++;
            }
        }
    }

}

int main(int argc, char* argv[]){
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    /* argument checking and setting */
    if(argc < 10){
        if(!rank)
            fprintf(stderr, "Usage: %s dataset PX PY PZ NX NY NZ NC outputfile\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char* datafile = argv[1];
    PX = atoi(argv[2]);
    PY = atoi(argv[3]);
    PZ = atoi(argv[4]);
    NX = atoi(argv[5]);
    NY = atoi(argv[6]);
    NZ = atoi(argv[7]);
    NC = atoi(argv[8]);
    char* outputfile = argv[9];

    if(PX * PY * PZ != size){
        if(!rank)
            fprintf(stderr, "Error: PX*PY*PZ must equal the total number of processes.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int proc_x = rank % PX;
    int proc_y = (rank / PX) % PY;
    int proc_z = rank / (PX * PY);

    int local_nx = NX / PX + (proc_x < NX % PX ? 1 : 0);
    int local_ny = NY / PY + (proc_y < NY % PY ? 1 : 0);
    int local_nz = NZ / PZ + (proc_z < NZ % PZ ? 1 : 0);

    int local_data_size = local_nx * local_ny * local_nz * NC; /* floats */
    
    int lnx = local_nx + 2;
    int lny = local_ny + 2;
    int lnz = local_nz + 2;

    int local_data_array_size = lnx * lny * lnz * NC;
    float* local_data = (float *)malloc(local_data_array_size * sizeof(float));

    if(local_data == NULL){
        fprintf(stderr, "Rank %d: Local data allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    /* assuming divisible grid */
    float* global_data = NULL;
    int sendcounts[size];
    int displs[size];
    if(!rank){
        //read the file and scatter the appropriate data to the processes
        global_data = (float *)malloc(NX * NY * NZ * NC * sizeof(float));
        if(global_data == NULL){
            fprintf(stderr, "Rank 0: Global data allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        FILE *fp = fopen(datafile, "r");
        if(fp == NULL){
            fprintf(stderr, "Rank 0: Cannot open input file %s\n", datafile);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        float* base_ptr[size];
        base_ptr[0] = global_data;
        int belong[NX*NY*NZ];
        
        int displ = 0;
        for(int p = 0; p < size; p++){
            int x, y, z;
            rankToProcessCoord(p, &x, &y, &z);
            int lx = NX / PX + ((x < (NX % PX)) ? 1 : 0);
            int ly = NY / PY + ((y < (NY % PY)) ? 1 : 0);
            int lz = NZ / PZ + ((z < (NZ % PZ)) ? 1 : 0);
            sendcounts[p] = lx * ly * lz * NC;
            displs[p] = displ;
            displ += sendcounts[p];
            if(p) base_ptr[p] = base_ptr[p - 1] + sendcounts[p - 1];
            int s_x = x * (NX / PX) + ((x < (NX % PX)) ? x : (NX % PX));
            int s_y = y * (NY / PY) + ((y < (NY % PY)) ? y : (NY % PY));   
            int s_z = z * (NZ / PZ) + (z < (NZ % PZ) ? z : (NZ % PZ));
            for(int z = 0; z < lz; z++){
                for(int y = 0; y < ly; y++){
                    for(int x = 0; x < lx; x++){
                        int x_, y_, z_;
                        x_ = s_x + x;
                        y_ = s_y + y;
                        z_ = s_z + z;
                        belong[(x_ + y_*NX + z_*NX*NY)] = p;
                    }
                }
            }
        }
        // for(int i = 0; i < NX*NY*NZ; i++){
        //     printf("%d belongs to Rank %d\n", i, belong[i]);
        // }

        for(int i = 0; i < NX*NY*NZ; i++){
            int rowRank = belong[i];
            int x, y, z;
            rankToProcessCoord(rowRank, &x, &y, &z);
            int lx = NX / PX + (x < NX % PX ? 1 : 0);
            int ly = NY / PY + (y < NY % PY ? 1 : 0);
            int lz = NZ / PZ + (z < NZ % PZ ? 1 : 0);
            for(int t = 0; t < NC; t++){
                float* offset = base_ptr[rowRank] + t*lx*ly*lz;
                if(fscanf(fp, "%f", offset) != 1){
                    fprintf(stderr, "Rank 0: Error reading data from file.\n");
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
            }
            base_ptr[rowRank]++;
        }
        fclose(fp);
    }
    //scatter the data to the processes
    //! Update to scatterv
    MPI_Scatterv(global_data, sendcounts, displs, MPI_FLOAT, local_data, local_data_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(!rank)
        free(global_data);
    float* temp_data = (float *)malloc(local_data_size * sizeof(float)); 
    /* can do local_data_array_size and then swap the pointers */
    memcpy(temp_data, local_data, local_data_size * sizeof(float));
    // memset(local_data, FLT_MAX, local_data_array_size * sizeof(float));
    /* Set the ghost cells to FLT_MAX */
    for(int t = 0; t < NC; t++){
        for(int z = 0; z < lnz; z++){
            for(int y = 0; y < lny; y++){
                for(int x = 0; x < lnx; x++){
                    local_data[localIndex(x, y, z, t, lnx, lny, lnz)] = FLT_MAX;
                }
            }
        }
    }
    /* Sets the value of the ghost cells as well */

    int idx = 0;
    for(int t = 0; t < NC; t++){
        for(int z = 0; z < local_nz; z++){
            for(int y = 0; y < local_ny; y++){
                for(int x = 0; x < local_nx; x++){
                //    local_data[localIndex(x, y, z, t, lnx, lny, lnz)] = temp_data[idx++];
                    local_data[localIndex(x + 1, y + 1, z + 1, t, lnx, lny, lnz)] = temp_data[idx++];
                }
            }
        }
    }

    free(temp_data);
    /*============================= WE HAVE GOT THE DATA! =========================================*/
    /* Compute and Communicate for all the time steps */
    
    int local_min_count = 0;
    int local_max_count = 0;
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;


    MPI_Datatype x_slice, y_slice, z_slice;
    MPI_Datatype y_line;
    MPI_Type_vector(local_ny,   // count
                    1,          // blocklength
                    lnx,        // stride between y's
                    MPI_FLOAT,
                    &y_line);
    MPI_Type_commit(&y_line);

    MPI_Type_create_hvector(local_nz,             // count of blocks (z)
                       1,                    // each block is one y_line
                       lnx*lny*sizeof(float), // stride to next z-plane in bytes
                       y_line,
                       &x_slice);
    MPI_Type_free(&y_line);



    MPI_Type_vector(local_nz, local_nx, lnx * lny, MPI_FLOAT, &y_slice);
    MPI_Type_vector(local_ny, local_nx, lnx, MPI_FLOAT, &z_slice);
    MPI_Type_commit(&x_slice);
    MPI_Type_commit(&y_slice);
    MPI_Type_commit(&z_slice);

    // Main computation loop for all time steps
    int *local_min_counts   = (int *)malloc(NC * sizeof(int));
    int *local_max_counts   = (int *)malloc(NC * sizeof(int));
    float *local_global_mins = (float *)malloc(NC * sizeof(float));
    float *local_global_maxs = (float *)malloc(NC * sizeof(float));

    int neighbours[6];

    enum{
        RIGHT = 0,
        LEFT,
        DOWN,
        UP,
        BACK,
        FRONT
    };

    neighbours[RIGHT] = CoordToRankProcess(proc_x + 1, proc_y, proc_z) ;   // right
    neighbours[LEFT] = CoordToRankProcess(proc_x - 1, proc_y, proc_z);   // left
    neighbours[DOWN] = CoordToRankProcess(proc_x, proc_y + 1, proc_z);   //down
    neighbours[UP] = CoordToRankProcess(proc_x, proc_y - 1, proc_z);   // up
    neighbours[BACK] = CoordToRankProcess(proc_x, proc_y, proc_z + 1);   // back
    neighbours[FRONT] = CoordToRankProcess(proc_x, proc_y, proc_z - 1);   // front

    // open a file for each rank to write the local extrema
 
    // if not present create it  
    char filename[100];
    sprintf(filename, "rank_%d.txt", rank);
    fp = fopen(filename, "w");
    if(fp == NULL){
        fprintf(stderr, "Rank %d: Cannot open output file %s\n", rank, filename);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    for(int t = 0; t < NC; t++){

        MPI_Request requests[12];
        MPI_Status statuses[12];
        int req_idx = 0;

        // X direction exchange
        // Send right edge, receive left ghost
        if(neighbours[RIGHT] != -1)
            MPI_Isend(&local_data[localIndex(local_nx, 1, 1, t, lnx, lny, lnz)], 1, x_slice, neighbours[RIGHT], 0, MPI_COMM_WORLD, &requests[req_idx++]);
        if(neighbours[LEFT] != -1)
            MPI_Irecv(&local_data[localIndex(0, 1, 1, t, lnx, lny, lnz)], 1, x_slice, neighbours[LEFT], 0, MPI_COMM_WORLD, &requests[req_idx++]);

        // Send left edge, receive right ghost
        if(neighbours[LEFT] != -1)
            MPI_Isend(&local_data[localIndex(1, 1, 1, t, lnx, lny, lnz)], 1, x_slice, neighbours[LEFT], 1, MPI_COMM_WORLD, &requests[req_idx++]);
        if(neighbours[RIGHT] != -1)
            MPI_Irecv(&local_data[localIndex(local_nx + 1, 1, 1, t, lnx, lny, lnz)], 1, x_slice, neighbours[RIGHT], 1, MPI_COMM_WORLD, &requests[req_idx++]);

        // Y-direction exchange (down-up)
        // Send down edge, receive up ghost
        if(neighbours[DOWN] != -1)
            MPI_Isend(&local_data[localIndex(1, local_ny, 1, t, lnx, lny, lnz)], 1, y_slice, neighbours[DOWN], 2, MPI_COMM_WORLD, &requests[req_idx++]);
        if(neighbours[UP] != -1)
            MPI_Irecv(&local_data[localIndex(1, 0, 1, t, lnx, lny, lnz)], 1, y_slice, neighbours[UP], 2, MPI_COMM_WORLD, &requests[req_idx++]);

        // Send up edge, receive down ghost
        if(neighbours[UP] != -1)
            MPI_Isend(&local_data[localIndex(1, 1, 1, t, lnx, lny, lnz)], 1, y_slice, neighbours[UP], 3, MPI_COMM_WORLD, &requests[req_idx++]);
        if(neighbours[DOWN] != -1)
            MPI_Irecv(&local_data[localIndex(1, local_ny + 1, 1, t, lnx, lny, lnz)], 1, y_slice, neighbours[DOWN], 3, MPI_COMM_WORLD, &requests[req_idx++]);

        // Z-direction exchange (back-front)
        // Send back edge, receive front ghost
        if(neighbours[BACK] != -1)
            MPI_Isend(&local_data[localIndex(1, 1, local_nz, t, lnx, lny, lnz)], 1, z_slice, neighbours[BACK], 4, MPI_COMM_WORLD, &requests[req_idx++]);
        if(neighbours[FRONT] != -1)
            MPI_Irecv(&local_data[localIndex(1, 1, 0, t, lnx, lny, lnz)], 1, z_slice, neighbours[FRONT], 4, MPI_COMM_WORLD, &requests[req_idx++]);
        
        // Send front edge, receive back ghost
        if(neighbours[FRONT] != -1)
            MPI_Isend(&local_data[localIndex(1, 1, 1, t, lnx, lny, lnz)], 1, z_slice, neighbours[FRONT], 5, MPI_COMM_WORLD, &requests[req_idx++]);
        if(neighbours[BACK] != -1)
            MPI_Irecv(&local_data[localIndex(1, 1, local_nz + 1, t, lnx, lny, lnz)], 1, z_slice, neighbours[BACK], 5, MPI_COMM_WORLD, &requests[req_idx++]);

        //overlap the communication and computation
        
        // findLocalExtrema(local_data, t, lnx, lny, lnz, local_nx, local_ny, local_nz,
        //     &local_min_counts[t], &local_max_counts[t], 
        //     &local_global_mins[t], &local_global_maxs[t]);
        // // Wait for all exchanges to complete
        // MPI_Waitall(req_idx, requests, statuses);

        // updateLocalExtrema(local_data, t, lnx, lny, lnz, local_nx, local_ny, local_nz,
        //     &local_min_counts[t], &local_max_counts[t], 
        //     &local_global_mins[t], &local_global_maxs[t]);

        MPI_Waitall(req_idx, requests, statuses);
        findLocalExtrema(local_data, t, lnx, lny, lnz, local_nx, local_ny, local_nz,  &local_min_counts[t], &local_max_counts[t], &local_global_mins[t], &local_global_maxs[t]);
    }

    int *global_min_counts = NULL;
    int *global_max_counts = NULL;
    float *global_mins = NULL;
    float *global_maxs = NULL;

    if (rank == 0) {
        global_min_counts = (int*)malloc(NC * sizeof(int));
        global_max_counts = (int*)malloc(NC * sizeof(int));
        global_mins = (float*)malloc(NC * sizeof(float));
        global_maxs = (float*)malloc(NC * sizeof(float));
    }

    // Reduce min/max counts
    MPI_Reduce(local_min_counts, global_min_counts, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_counts, global_max_counts, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Reduce global min/max values
    MPI_Reduce(local_global_mins, global_mins, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_global_maxs, global_maxs, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if(!rank){
        FILE* fp = fopen(outputfile, "w");
        if(!fp){
            fprintf(stderr, "Rank 0: Cannot open output file %s\n", outputfile);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        for(int t = 0; t < NC; t++){
            fprintf(fp, "(%d, %d)", global_min_counts[t], global_max_counts[t]);
            if(t < NC - 1)
                fprintf(fp, ", ");
        }
        fprintf(fp, "\n");

        for(int t = 0; t < NC; t++){
            fprintf(fp, "(%f, %f)", global_mins[t], global_maxs[t]);
            if(t < NC - 1)
                fprintf(fp, ", ");
        }
        fprintf(fp, "\n");

        // fprintf(fp, "%f, %f, %f\n", read_time, main_time, total_time);
        fclose(fp);

        free(global_min_counts);
        free(global_max_counts);
        free(global_mins);
        free(global_maxs);
    }

    free(local_data);
    free(local_min_counts);
    free(local_max_counts);
    free(local_global_mins);
    free(local_global_maxs);
    //free the datatypes
    MPI_Type_free(&x_slice);
    MPI_Type_free(&y_slice);
    MPI_Type_free(&z_slice);
    fclose(fp);
    MPI_Finalize();
    return 0;
}
