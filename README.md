# Parallel Cosine Similarity for Document Matching (MPI + GCP)

This project implements a parallelized version of cosine similarity computation for document matching using MPI (OpenMPI) on distributed environments, specifically Google Cloud Platform. By leveraging distributed memory parallelism, the system significantly improves performance and scalability over the serial baseline.

---

## Abstract

Computing pairwise cosine similarity between all documents in a large corpus has O(n²) time complexity, making it inefficient for large datasets. This project parallelizes the computation using OpenMPI, distributing documents and workloads across multiple nodes and cores. Both intra-regional and infra-regional cluster configurations were tested on GCP to evaluate performance, scalability, and latency behavior.

---

## 🔍 Key Features

- **TF-IDF Vectorization** of all documents in the corpus
- **Pairwise Cosine Similarity** between document vectors
- **Parallel Implementation** using **MPI** with distributed message passing
- **Deployment & Testing on Google Cloud Platform (GCP)** with:
  - Light & Fat clusters
  - Intra- and Infra-regional topologies
- **Performance Metrics**: Speedup, strong/weak scalability, latency impact
- **Testing**: Serial vs Parallel consistency validation, identity tests, and debug strategies

---

## How It Works

### Serial Version
- Tokenize documents line by line from `.txt` file
- Compute term frequency (TF) and document frequency (DF)
- Calculate TF-IDF vectors
- Compute all pairwise cosine similarity scores

### Parallel Version (MPI)
- Master (rank 0) reads and distributes documents
- Workers compute TF, DF, and local TF-IDF
- Global vocabulary broadcasted using `MPI_Bcast`
- Document frequency vectors merged with `MPI_Allreduce`
- Final cosine similarity is computed in parallel by dividing index pairs across processes

---

## Running the Code

### Prerequisites

- C++ compiler
- OpenMPI installed
- Google Cloud CLI (if running on GCP)

### Compile & Run (Locally or on VM)

```bash
# Compile
mpic++ -std=c++17 -o main main_parallelized.cpp

# Run with 4 processes
mpirun -np 4 ./main


