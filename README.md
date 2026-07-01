# Parallel 3D Grid Processing with MPI

An MPI-based implementation for distributed processing of large 3D grid datasets using domain decomposition and ghost-cell communication. The project focuses on parallel detection of local and global extrema while exploring performance and scalability on distributed-memory systems.

> Developed as part of **CS633: Parallel Computing** at IIT Kanpur.

---

## Features

- 3D domain decomposition across MPI processes
- Ghost-cell (halo) exchange for boundary communication
- Non-blocking communication using `MPI_Isend` / `MPI_Irecv`
- MPI derived datatypes for efficient data transfers
- Support for uneven domain partitioning
- Strong scaling experiments and performance benchmarking
- Multiple implementations (baseline, optimized, and final)

---

## Repository Structure

```text
.
├── Assignment.pdf
├── LICENSE
├── README.md
├── baseline.c
├── optimised.c
├── final.c
├── src_c.c
├── test.c
├── datasets/
│   ├── data_64_64_64_3.bin
│   ├── data_64_64_64_3.txt
│   ├── data_64_64_96_7.bin
│   └── data_64_64_96_7.txt
└── results/
    ├── scaling_results.csv
    ├── data_64_64_64_3_scaling_boxplot.png
    ├── data_64_64_96_7_scaling_boxplot.png
    ├── strong_scaling_boxplot.png
    └── strong_scaling_plot.png
```

---

## Parallelization Strategy

The implementation distributes a 3D grid across a Cartesian process grid (`PX × PY × PZ`).

Each MPI process:

- Owns a local subdomain of the global grid
- Exchanges ghost cells with neighboring processes
- Computes local minima and maxima
- Participates in global reductions to determine overall extrema

The implementation uses:

- Cartesian domain decomposition
- Non-blocking point-to-point communication (`MPI_Isend` / `MPI_Irecv`)
- MPI derived datatypes for efficient boundary exchange
- Support for uneven grid partitioning across processes

---

## Building

### Requirements

- MPI (OpenMPI or MPICH)
- GCC (C99 compatible)
- Linux

### Compile

```bash
mpicc -O3 -o final.x final.c -lm
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

### Example

```bash
mpirun -np 8 ./final.x \
datasets/data_64_64_64_3.txt \
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

```text
(125,143), (98,156), (110,132)
(0.125,9.875), (0.250,9.750), (0.313,9.625)
0.125,0.875,1.000
```

---

## Performance

The project includes strong scaling experiments across multiple MPI process counts.

Key observations:

- 3D decomposition improves load balancing compared to lower-dimensional partitioning.
- Communication overhead dominates for smaller problem sizes.
- Larger datasets demonstrate improved scalability.
- Ghost-cell exchange using MPI derived datatypes minimizes packing overhead.

Performance plots and scaling results are available in the `results/` directory.

---

## Technologies

- C
- MPI
- OpenMPI / MPICH
- Linux
- Distributed-memory parallel computing

---

## License

Released under the MIT License. See the [LICENSE](LICENSE) file for details.