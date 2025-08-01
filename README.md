# Parallel Cosine Similarity for Document Matching (MPI + GCP)

This project implements a parallelized version of cosine similarity computation for document matching using MPI (OpenMPI) on distributed environments, specifically Google Cloud Platform. By leveraging distributed memory parallelism, the system significantly improves performance and scalability over the serial baseline.

---

## Abstract

Computing pairwise cosine similarity between all documents in a large corpus has O(n²) time complexity, making it inefficient for large datasets. This project parallelizes the computation using OpenMPI, distributing documents and workloads across multiple nodes and cores. Both intra-regional and infra-regional cluster configurations were tested on GCP to evaluate performance, scalability, and latency behavior.

---

## Key Features

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

## Cluster Configurations

### Light Cluster (Intra-Regional)
- **Nodes**: 8 VMs
- **vCPUs**: 2 per node
- **RAM**: 8GB per node
- **Region**: europe-west1-d

### Fat Cluster (Intra-Regional)
- **Nodes**: 2 VMs
- **vCPUs**: 8 per node
- **RAM**: 32GB per node
- **Region**: us-central1-c

### Light Cluster (Infra-Regional)
- **Nodes**: 8 VMs in different continents
- **vCPUs**: 2 per node
- **RAM**: 8GB per node
- **Regions**:
  - northamerica-northeast2-b
  - australia-southeast1-b
  - me-central1-a
  - asia-northeast3-c
  - europe-north2-b
  - africa-south1-c
  - us-south1-a
  - southamerica-east1-c

### Fat Cluster (Infra-Regional)
- **Nodes**: 2 high-performance VMs
- **vCPUs**: 8 per node
- **RAM**: 32GB per node
- **Regions**:
  - europe-southwest1-b
  - northamerica-northeast1-c
    
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


