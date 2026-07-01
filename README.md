# Parallel 3D Grid Processing with MPI

An MPI-based implementation for distributed processing of large 3D grid datasets using domain decomposition and ghost-cell communication. The project focuses on parallel detection of local and global extrema while exploring performance and scalability on distributed-memory systems.

> Developed as part of CS633 (Parallel Computing) at IIT Kanpur.

---

## Features

- 3D domain decomposition across MPI processes
- Ghost-cell (halo) exchange for boundary communication
- Non-blocking communication using `MPI_Isend` / `MPI_Irecv`
- MPI derived datatypes for efficient data transfers
- Support for uneven domain partitioning
- Strong scaling experiments and performance benchmarking
- Multiple implementations (baseline, optimized, final)

---

## Repository Structure

```
.
├── src.c
├── optmised_src.c
├── final_src.c
├── src_test.c
├── Assignment.pdf
├── LICENSE
├── scaling_results.csv
├── *.png
└── datasets/
```

---

## Parallelization Strategy

The implementation distributes a 3D grid across a Cartesian process grid (`PX × PY × PZ`).

Each process:

- owns a local subdomain
- exchanges ghost cells with neighboring processes
- computes local minima and maxima
- participates in global reductions for extrema computation

Communication uses:

- MPI Cartesian decomposition
- Non-blocking point-to-point communication
- MPI derived datatypes for slice exchange

---

## Building

Requirements

- MPI (OpenMPI or MPICH)
- GCC (C99)
- Linux

Compile:

```bash
mpicc -O3 -o final.x final_src.c -lm
```

---

## Running

```bash
mpirun -np <processes> ./final.x \
    <dataset> \
    <PX> <PY> <PZ> \
    <NX> <NY> <NZ> \
    <TIMESTEPS> \
    <output>
```

Example:

```bash
mpirun -np 8 ./final.x \
data_64_64_64_3.txt \
2 2 2 \
64 64 64 \
3 \
output.txt
```

---

## Output

The program reports:

- Local extrema counts
- Global minimum and maximum values
- Read time
- Computation time
- Total runtime

Example:

```
(125,143), (98,156), (110,132)
(0.125,9.875), (0.250,9.750), (0.313,9.625)
0.125,0.875,1.000
```

---

## Performance

The project evaluates strong scaling across multiple MPI process counts.

Highlights:

- 3D decomposition improves load balancing.
- Communication dominates for smaller datasets.
- Larger datasets demonstrate better scalability.
- Ghost-cell exchange implemented using MPI derived datatypes minimizes packing overhead.

Performance plots are included in the repository.

---

## Technologies

- C
- MPI
- OpenMPI / MPICH
- Linux
- Distributed-memory parallel computing

---

## License

Released under the MIT License.
