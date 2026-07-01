//===============================================================================
// Program : optimised.c
// Purpose : Find local and global extremas in a 3D grid (optimised)
//
// Compile : mpicc -o optimised.x optimised.c -O3
//===============================================================================
/*===============================LIBRARY INCLUDE FILES===========================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include <math.h>

int PX, PY, PZ, NX, NY, NZ, NC;
int rank, size;
int proc_x, proc_y, proc_z;
int local_nx, local_ny, local_nz;
int lnx, lny, lnz;

int offsets[6][3] = {
    {+1, 0, 0},
    {-1, 0, 0},
    {0, +1, 0},
    {0, -1, 0},
    {0, 0, +1},
    {0, 0, -1}};

enum
{
    RIGHT = 0,
    LEFT,
    DOWN,
    UP,
    BACK,
    FRONT,
    NEIGHBOURS
};

// DATA: Convert linear index to 3D coordinates
void linearToCoord(int linearIndex, int *x, int *y, int *z)
{
    *z = linearIndex / (NX * NY);
    int remainder = linearIndex % (NX * NY);
    *y = remainder / NX;
    *x = remainder % NX;
}

// DATA: Convert 3D coordinates to linear index
int coordToLinear(int x, int y, int z)
{
    return x + y * NX + z * NX * NY;
}

// DATA -> RANK: Determine which process owns a specific coordinate
int coordToRank(int x, int y, int z)
{
    int process_x = x / (NX / PX);
    int process_y = y / (NY / PY);
    int process_z = z / (NZ / PZ);
    return process_x + process_y * PX + process_z * PX * PY;
}

int coordToRankU(int x, int y, int z)
{
    // Calculate which process owns x
    int div_x = NX / PX;
    int rem_x = NX % PX;
    int process_x;
    if (x < (div_x + 1) * rem_x)
    {
        // In the region of processes with div_x + 1 elements
        process_x = x / (div_x + 1);
    }
    else
    {
        // In the region of processes with div_x elements
        process_x = rem_x + (x - (div_x + 1) * rem_x) / div_x;
    }

    // Similar calculations for y and z dimensions
    int div_y = NY / PY;
    int rem_y = NY % PY;
    int process_y;
    if (y < (div_y + 1) * rem_y)
    {
        process_y = y / (div_y + 1);
    }
    else
    {
        process_y = rem_y + (y - (div_y + 1) * rem_y) / div_y;
    }

    int div_z = NZ / PZ;
    int rem_z = NZ % PZ;
    int process_z;
    if (z < (div_z + 1) * rem_z)
    {
        process_z = z / (div_z + 1);
    }
    else
    {
        process_z = rem_z + (z - (div_z + 1) * rem_z) / div_z;
    }

    return process_x + process_y * PX + process_z * PX * PY;
}

// PROCESS: Convert process rank to its position in the process grid
void rankToProcessCoord(int rank, int *process_x, int *process_y, int *process_z)
{
    *process_x = rank % PX;
    *process_y = (rank / PX) % PY;
    *process_z = rank / (PX * PY);
}

int CoordToRankProcess(int x, int y, int z)
{
    if (x < 0 || x >= PX || y < 0 || y >= PY || z < 0 || z >= PZ)
        return MPI_PROC_NULL;
    return x + y * PX + z * PX * PY;
}

// PROCESS: Convert linear index to rank
int idxToRank(int i)
{
    int x, y, z;
    linearToCoord(i, &x, &y, &z);
    return coordToRank(x, y, z);
}

// Function to compute 3D indices in local array with ghost cells
int localIndex(int x, int y, int z, int t, int lnx, int lny, int lnz)
{
    return x + (y)*lnx + (z)*lnx * lny + t * lnx * lny * lnz;
}

// Function to determine if a point is a local minimum
// Returns 1 if it's a local minimum, 0 otherwise
int isLocalMinimum(float *local_data, int x, int y, int z, int t, int lnx, int lny, int lnz)
{

    // Check all 6 neighbors
    float value = local_data[localIndex(x, y, z, t, lnx, lny, lnz)];

    for (int o = 0; o < 6; o++)
    {
        int dx = offsets[o][0];
        int dy = offsets[o][1];
        int dz = offsets[o][2];
        float n_value = local_data[localIndex(x + dx, y + dy, z + dz, t, lnx, lny, lnz)];
        if (fabs(n_value - FLT_MAX) < FLT_EPSILON)
            continue; // Skip ghost cells
        if (value >= n_value)
            return 0; // Not a minimum
    }

    return 1; // It's a local minimum
}

// Similarly for local maximum
int isLocalMaximum(float *local_data, int x, int y, int z, int t, int lnx, int lny, int lnz)
{

    float value = local_data[localIndex(x, y, z, t, lnx, lny, lnz)];

    for (int o = 0; o < 6; o++)
    {
        int dx = offsets[o][0];
        int dy = offsets[o][1];
        int dz = offsets[o][2];
        float n_value = local_data[localIndex(x + dx, y + dy, z + dz, t, lnx, lny, lnz)];
        if (fabs(n_value - FLT_MAX) < FLT_EPSILON)
            continue; // Skip ghost cells
        if (value <= n_value)
            return 0; // Not a maximum
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
    for (int z = 1; z <= local_nz; z++)
    {
        for (int y = 1; y <= local_ny; y++)
        {
            for (int x = 1; x <= local_nx; x++)
            {
                float value = local_data[localIndex(x, y, z, t, lnx, lny, lnz)];
                // Update global extrema
                if (value < *global_min)
                    *global_min = value;
                if (value > *global_max)
                    *global_max = value;

                // Check for local extrema
                if (isLocalMinimum(local_data, x, y, z, t, lnx, lny, lnz))
                {
                    (*min_count)++;
                }

                if (isLocalMaximum(local_data, x, y, z, t, lnx, lny, lnz))
                {
                    (*max_count)++;
                }
            }
        }
    }
}

void read_arrange_data(float **global_data, int **sendcounts, int **displs, char *datafile)
{

    (*global_data) = (float *)malloc(NX * NY * NZ * NC * sizeof(float));
    if ((*global_data) == NULL)
    {
        fprintf(stderr, "Rank 0: Global data allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    (*sendcounts) = (int *)malloc(size * sizeof(int));
    (*displs) = (int *)malloc(size * sizeof(int));
    if ((*sendcounts) == NULL || (*displs) == NULL)
    {
        fprintf(stderr, "Rank 0: Sendcounts or displacements allocation failed.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    FILE *fp = fopen(datafile, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Rank 0: Cannot open input file %s\n", datafile);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    float *base_ptr[size];
    base_ptr[0] = (*global_data); // check if gather is faster

    int displ = 0;
    for (int p = 0; p < size; p++)
    {
        int x, y, z;
        rankToProcessCoord(p, &x, &y, &z);
        int lx = NX / PX + ((x < (NX % PX)) ? 1 : 0);
        int ly = NY / PY + ((y < (NY % PY)) ? 1 : 0);
        int lz = NZ / PZ + ((z < (NZ % PZ)) ? 1 : 0);
        (*sendcounts)[p] = lx * ly * lz * NC;
        (*displs)[p] = displ;
        displ += (*sendcounts)[p];
        if (p)
            base_ptr[p] = base_ptr[p - 1] + (*sendcounts)[p - 1];
    }

    for (int i = 0; i < NX * NY * NZ; i++)
    {
        int dx, dy, dz;
        linearToCoord(i, &dx, &dy, &dz);
        int rowRank = coordToRankU(dx, dy, dz);
        float *offset = base_ptr[rowRank];
        size_t elements_to_read  = NC;
        if(fread(offset, sizeof(float), elements_to_read, fp) != elements_to_read)
        {
            fprintf(stderr, "Rank 0: Error reading data from file.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        base_ptr[rowRank] += elements_to_read;
    }
    fclose(fp);
}

void read_parallel(float *local_data, char *datafile) {
    
    MPI_Datatype filetype;
    int *blocklength = (int*)malloc(local_ny * local_nz * sizeof(int));
    int *displ = (int*)malloc(local_ny * local_nz * sizeof(int));
    
    // Calculate starting indices for this process
    int s_x = proc_x * (NX / PX) + ((proc_x < (NX % PX)) ? proc_x : (NX % PX));
    int s_y = proc_y * (NY / PY) + ((proc_y < (NY % PY)) ? proc_y : (NY % PY));   
    int s_z = proc_z * (NZ / PZ) + ((proc_z < (NZ % PZ)) ? proc_z : (NZ % PZ));
    
    int idx = 0;
    for(int z = 0; z < local_nz; z++) {
        for(int y = 0; y < local_ny; y++) {
            blocklength[idx] = local_nx * NC;
            displ[idx] = ((s_x) + (s_y + y) * NX + (s_z + z) * NX * NY) * NC;
            idx++;
        }
    }
    
    MPI_Type_indexed(local_ny * local_nz, blocklength, displ, MPI_FLOAT, &filetype);
    MPI_Type_commit(&filetype);
    
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, datafile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
    
    // Read data into a temporary buffer without ghost cells
    MPI_File_read_all(fh, local_data, local_nx * local_ny * local_nz * NC, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    //write the data into file
    free(blocklength);
    free(displ);
    MPI_Type_free(&filetype);
}

void set_local_data(float *local_data, float *temp_data, int local_nx, int local_ny, int local_nz,
                    int lnx, int lny, int lnz)
{
    int idx = 0;
    for (int t = 0; t < NC; t++)
    {
        // X‐faces at x=0 and x=lnx-1
        for (int z = 0; z < lnz; z++)
            for (int y = 0; y < lny; y++)
            {
                local_data[localIndex(0, y, z, t, lnx, lny, lnz)] = FLT_MAX;
                local_data[localIndex(lnx - 1, y, z, t, lnx, lny, lnz)] = FLT_MAX;
            }

        // Y‐faces at y=0 and y=lny-1, but skip corners already set
        for (int z = 0; z < lnz; z++)
            for (int x = 1; x < lnx - 1; x++)
            {
                local_data[localIndex(x, 0, z, t, lnx, lny, lnz)] = FLT_MAX;
                local_data[localIndex(x, lny - 1, z, t, lnx, lny, lnz)] = FLT_MAX;
            }

        // Z‐faces at z=0 and z=lnz-1, skip edges already set
        for (int y = 1; y < lny - 1; y++)
            for (int x = 1; x < lnx - 1; x++)
            {
                local_data[localIndex(x, y, 0, t, lnx, lny, lnz)] = FLT_MAX;
                local_data[localIndex(x, y, lnz - 1, t, lnx, lny, lnz)] = FLT_MAX;
            }
    }

    for (int t = 0; t < NC; t++)
    {
        for (int z = 0; z < local_nz; z++)
        {
            for (int y = 0; y < local_ny; y++)
            {
                for (int x = 0; x < local_nx; x++)
                {
                    local_data[localIndex(x + 1, y + 1, z + 1, t, lnx, lny, lnz)] = temp_data[((z * local_ny + y) * local_nx + x) * NC + t];
                }
            }
        }
    }
}

void write_to_file(char *outputfile, int *global_min_counts, int *global_max_counts,
                   float *global_mins, float *global_maxs, double max_read_time, 
                    double max_main_time, double max_total_time)
{
    FILE *fp = fopen(outputfile, "w");
    if (!fp)
    {
        fprintf(stderr, "Rank 0: Cannot open output file %s\n", outputfile);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    for (int t = 0; t < NC; t++)
    {
        fprintf(fp, "(%d, %d)", global_min_counts[t], global_max_counts[t]);
        if (t < NC - 1)
            fprintf(fp, ", ");
    }
    fprintf(fp, "\n");

    for (int t = 0; t < NC; t++)
    {
        fprintf(fp, "(%f, %f)", global_mins[t], global_maxs[t]);
        if (t < NC - 1)
            fprintf(fp, ", ");
    }
    fprintf(fp, "\n");
    fprintf(fp, "%f, %f, %f\n", max_read_time, max_main_time, max_total_time);
    fclose(fp);
}

int main(int argc, char *argv[])
{
    double time1, time2, time3;
    double read_time, main_time, total_time;
    double max_read_time, max_main_time, max_total_time;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    time1 = MPI_Wtime();
    /* argument checking and setting */
    if (argc < 10)
    {
        if (!rank)
            fprintf(stderr, "Usage: %s dataset PX PY PZ NX NY NZ NC outputfile\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char *datafile = argv[1];
    PX = atoi(argv[2]);
    PY = atoi(argv[3]);
    PZ = atoi(argv[4]);
    NX = atoi(argv[5]);
    NY = atoi(argv[6]);
    NZ = atoi(argv[7]);
    NC = atoi(argv[8]);
    char *outputfile = argv[9];

    if (PX * PY * PZ != size)
    {
        if (!rank)
            fprintf(stderr, "Error: PX*PY*PZ must equal the total number of processes.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    proc_x = rank % PX;
    proc_y = (rank / PX) % PY;
    proc_z = rank / (PX * PY);

    local_nx = NX / PX + (proc_x < NX % PX ? 1 : 0);
    local_ny = NY / PY + (proc_y < NY % PY ? 1 : 0);
    local_nz = NZ / PZ + (proc_z < NZ % PZ ? 1 : 0);

    lnx = local_nx + 2;
    lny = local_ny + 2;
    lnz = local_nz + 2;

    int local_data_size = local_nx * local_ny * local_nz * NC; /* floats */
    int local_data_array_size = lnx * lny * lnz * NC;

    float *local_data = (float *)malloc(local_data_array_size * sizeof(float));
    float *temp_data = (float *)malloc(local_data_size * sizeof(float));

    if (local_data == NULL || temp_data == NULL)
    {
        fprintf(stderr, "Rank %d: Local data allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    float *global_data = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;

    if (!rank)
    {
        read_arrange_data(&global_data, &sendcounts, &displs, datafile);
    }

    MPI_Scatterv(global_data, sendcounts, displs, MPI_FLOAT, temp_data, local_data_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // read_parallel(temp_data, datafile);

    time2 = MPI_Wtime();
    read_time = time2 - time1;
    
    if (!rank)
    {
        free(sendcounts);
        free(displs);
        free(global_data);
    }

    /*============================= WE HAVE GOT THE DATA! =========================================*/
    set_local_data(local_data, temp_data, local_nx, local_ny, local_nz, lnx, lny, lnz);

    free(temp_data);
    /*============================= WE HAVE GOT THE DATA! =========================================*/
    /* Compute and Communicate for all the time steps */

    MPI_Datatype x_slice, y_slice, z_slice;
    MPI_Datatype y_line;
    MPI_Type_vector(local_ny, // count
                    1,        // blocklength
                    lnx,      // stride between y's
                    MPI_FLOAT,
                    &y_line);
    MPI_Type_commit(&y_line);

    MPI_Type_create_hvector(local_nz,                  // count of blocks (z)
                            1,                         // each block is one y_line
                            lnx * lny * sizeof(float), // stride to next z-plane in bytes
                            y_line,
                            &x_slice);
    MPI_Type_free(&y_line);

    MPI_Type_vector(local_nz, local_nx, lnx * lny, MPI_FLOAT, &y_slice);
    MPI_Type_vector(local_ny, local_nx, lnx, MPI_FLOAT, &z_slice);
    MPI_Type_commit(&x_slice);
    MPI_Type_commit(&y_slice);
    MPI_Type_commit(&z_slice);

    // Main computation loop for all time steps
    int *local_min_counts = (int *)malloc(NC * sizeof(int));
    int *local_max_counts = (int *)malloc(NC * sizeof(int));
    float *local_global_mins = (float *)malloc(NC * sizeof(float));
    float *local_global_maxs = (float *)malloc(NC * sizeof(float));

    int neighbours[NEIGHBOURS];
    neighbours[RIGHT] = CoordToRankProcess(proc_x + 1, proc_y, proc_z); // right
    neighbours[LEFT] = CoordToRankProcess(proc_x - 1, proc_y, proc_z);  // left
    neighbours[DOWN] = CoordToRankProcess(proc_x, proc_y + 1, proc_z);  // down
    neighbours[UP] = CoordToRankProcess(proc_x, proc_y - 1, proc_z);    // up
    neighbours[BACK] = CoordToRankProcess(proc_x, proc_y, proc_z + 1);  // back
    neighbours[FRONT] = CoordToRankProcess(proc_x, proc_y, proc_z - 1); // front

    for (int t = 0; t < NC; t++)
    {

        MPI_Request requests[12];
        MPI_Status statuses[12];
        int req_idx = 0;

        // X direction exchange
        // Send right edge, receive left ghost
        MPI_Isend(&local_data[localIndex(local_nx, 1, 1, t, lnx, lny, lnz)], 1, x_slice, neighbours[RIGHT], 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(0, 1, 1, t, lnx, lny, lnz)], 1, x_slice, neighbours[LEFT], 0, MPI_COMM_WORLD, &requests[req_idx++]);

        // Send left edge, receive right ghost
        MPI_Isend(&local_data[localIndex(1, 1, 1, t, lnx, lny, lnz)], 1, x_slice, neighbours[LEFT], 1, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(local_nx + 1, 1, 1, t, lnx, lny, lnz)], 1, x_slice, neighbours[RIGHT], 1, MPI_COMM_WORLD, &requests[req_idx++]);

        // Y-direction exchange (down-up)
        // Send down edge, receive up ghost
        MPI_Isend(&local_data[localIndex(1, local_ny, 1, t, lnx, lny, lnz)], 1, y_slice, neighbours[DOWN], 2, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(1, 0, 1, t, lnx, lny, lnz)], 1, y_slice, neighbours[UP], 2, MPI_COMM_WORLD, &requests[req_idx++]);

        // Send up edge, receive down ghost
        MPI_Isend(&local_data[localIndex(1, 1, 1, t, lnx, lny, lnz)], 1, y_slice, neighbours[UP], 3, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(1, local_ny + 1, 1, t, lnx, lny, lnz)], 1, y_slice, neighbours[DOWN], 3, MPI_COMM_WORLD, &requests[req_idx++]);

        // Z-direction exchange (back-front)
        // Send back edge, receive front ghost
        MPI_Isend(&local_data[localIndex(1, 1, local_nz, t, lnx, lny, lnz)], 1, z_slice, neighbours[BACK], 4, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(1, 1, 0, t, lnx, lny, lnz)], 1, z_slice, neighbours[FRONT], 4, MPI_COMM_WORLD, &requests[req_idx++]);

        // Send front edge, receive back ghost
        MPI_Isend(&local_data[localIndex(1, 1, 1, t, lnx, lny, lnz)], 1, z_slice, neighbours[FRONT], 5, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_data[localIndex(1, 1, local_nz + 1, t, lnx, lny, lnz)], 1, z_slice, neighbours[BACK], 5, MPI_COMM_WORLD, &requests[req_idx++]);

        MPI_Waitall(req_idx, requests, statuses);
        findLocalExtrema(local_data, t, lnx, lny, lnz, local_nx, local_ny, local_nz, &local_min_counts[t], &local_max_counts[t], &local_global_mins[t], &local_global_maxs[t]);
    }

    int *global_min_counts = NULL;
    int *global_max_counts = NULL;
    float *global_mins = NULL;
    float *global_maxs = NULL;

    if (rank == 0)
    {
        global_min_counts = (int *)malloc(NC * sizeof(int));
        global_max_counts = (int *)malloc(NC * sizeof(int));
        global_mins = (float *)malloc(NC * sizeof(float));
        global_maxs = (float *)malloc(NC * sizeof(float));
    }

    // Reduce min/max counts
    MPI_Reduce(local_min_counts, global_min_counts, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_counts, global_max_counts, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Reduce global min/max values
    MPI_Reduce(local_global_mins, global_mins, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_global_maxs, global_maxs, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    time3 = MPI_Wtime();
    main_time = time3 - time2;
    total_time = time3 - time1;
    
    // Find maximum time across all processes
    MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&main_time, &max_main_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (!rank)
    {
        write_to_file(outputfile, global_min_counts, global_max_counts, global_mins, global_maxs, 
                      max_read_time, max_main_time, max_total_time);

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
    // free the datatypes
    MPI_Type_free(&x_slice);
    MPI_Type_free(&y_slice);
    MPI_Type_free(&z_slice);
    MPI_Finalize();
    return 0;
}
