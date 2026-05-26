#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>        /* used only for rank/size and host-side scatter/gather */

/* Pull in your existing host-side helpers (gifenc, orbium placement, etc.) */
extern "C" {
#include "gifenc.h"     /* ge_new_gif / ge_add_frame / ge_close_gif          */
#include "lenia.h"      /* orbium_coo, evolve_lenia, evolve_lenia_mpi        */
#include "orbium.h"     /* place_orbium, inferno_pallete                     */
double *generate_kernel(double *K, const unsigned int size);
}

/* -------------------------------------------------------------------------
 * Error-checking macros
 * ---------------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define NCCL_CHECK(call)                                                     \
    do {                                                                     \
        ncclResult_t _r = (call);                                            \
        if (_r != ncclSuccess) {                                             \
            fprintf(stderr, "NCCL error %s:%d  %s\n",                       \
                    __FILE__, __LINE__, ncclGetErrorString(_r));             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)


__global__ void convolve_kernel(double       *result,
                                const double *buf,
                                int           local_rows,
                                int           cols,
                                const double *kernel_w,
                                int           kernel_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= local_rows || j >= cols) return;

    int halo = kernel_size / 2;
    double sum = 0.0;

    for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++) {
        for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++) {
            int row_idx = i + kri;                          /* within buf     */
            int col_idx = (j + kcj - halo + cols) % cols;  /* wrap columns   */
            sum += kernel_w[ki * kernel_size + kj] *
                   buf[(size_t)row_idx * cols + col_idx];
        }
    }
    result[(size_t)i * cols + j] = sum;
}


__device__ __forceinline__ double growth_lenia_dev(double u)
{
    const double mu    = 0.135;
    const double sigma = 0.015;
    return 2.0 * exp(-((u - mu) * (u - mu)) / (2.0 * sigma * sigma)) - 1.0;
}

__global__ void apply_growth_kernel(double       *local_buf,
                                    const double *conv_result,
                                    int           local_rows,
                                    int           cols,
                                    int           halo,
                                    double        dt)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= local_rows || j >= cols) return;

    double *cell = &local_buf[(size_t)(halo + i) * cols + j];
    double  val  = *cell + dt * growth_lenia_dev(conv_result[(size_t)i * cols + j]);
    *cell = fmin(1.0, fmax(0.0, val));
}


static void exchange_halos_nccl(double      *local_buf,
                                int          local_rows,
                                int          cols,
                                int          halo,
                                int         *sendcounts,   /* host array */
                                int         *displs,       /* host array */
                                int          my_start,
                                int          total_rows,
                                int          myid,
                                int          procs,
                                ncclComm_t   comm,
                                cudaStream_t stream)
{
    int prev = (myid - 1 + procs) % procs;
    int next = (myid + 1) % procs;

    if (local_rows >= halo) {
        /*
         * Two paired send/recv rounds inside a single NCCL group.
         * NCCL guarantees deadlock-free execution when all sends and
         * receives are enclosed in ncclGroupStart / ncclGroupEnd.
         */
        NCCL_CHECK(ncclGroupStart());

        /* pass 0 --------------------------------------------------------- */
        /* send bottom real rows to next */
        NCCL_CHECK(ncclSend(
            local_buf + (size_t)local_rows * cols,
            (size_t)halo * cols, ncclDouble, next, comm, stream));
        /* recv top ghost from prev */
        NCCL_CHECK(ncclRecv(
            local_buf,
            (size_t)halo * cols, ncclDouble, prev, comm, stream));

        /* pass 1 --------------------------------------------------------- */
        /* send top real rows to prev */
        NCCL_CHECK(ncclSend(
            local_buf + (size_t)halo * cols,
            (size_t)halo * cols, ncclDouble, prev, comm, stream));
        /* recv bottom ghost from next */
        NCCL_CHECK(ncclRecv(
            local_buf + (size_t)(halo + local_rows) * cols,
            (size_t)halo * cols, ncclDouble, next, comm, stream));

        NCCL_CHECK(ncclGroupEnd());

    } else {
        /*
         * Fallback: local_rows < halo (tiny grid / many GPUs).
         * Use ncclAllGather to broadcast every rank's real rows to all ranks,
         * then copy the required ghost rows from the gathered buffer.
         *
         * ncclAllGather requires equal send counts, so we use the maximum
         * local_rows across all ranks and pad with zeros where needed.
         * For simplicity we use a host-side staging area (this path is rare).
         */
        int max_local = 0;
        for (int r = 0; r < procs; r++) {
            int lr = sendcounts[r] / cols;
            if (lr > max_local) max_local = lr;
        }

        /* Allocate a full device gather buffer */
        double *d_full;
        CUDA_CHECK(cudaMalloc(&d_full,
                              (size_t)procs * max_local * cols * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_full, 0,
                              (size_t)procs * max_local * cols * sizeof(double)));

        /* ncclAllGather: each rank contributes its real rows */
        NCCL_CHECK(ncclAllGather(
            local_buf + (size_t)halo * cols,       /* sendbuf (real rows)   */
            d_full,                                /* recvbuf               */
            (size_t)max_local * cols, ncclDouble,
            comm, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        /*
         * Reconstruct the global grid on the device so we can index it.
         * We copy ranks' contributions back using their actual sendcounts.
         * Using a host staging buffer here is acceptable – this path is rare.
         */
        size_t full_size = (size_t)total_rows * cols * sizeof(double);
        double *h_full = (double *)malloc(full_size);
        double *h_gathered;
        CUDA_CHECK(cudaMallocHost(&h_gathered,
                                  (size_t)procs * max_local * cols * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(h_gathered, d_full,
                              (size_t)procs * max_local * cols * sizeof(double),
                              cudaMemcpyDeviceToHost));

        for (int r = 0; r < procs; r++) {
            int lr      = sendcounts[r] / cols;
            int rstart  = displs[r] / cols;
            memcpy(h_full + (size_t)rstart * cols,
                   h_gathered + (size_t)r * max_local * cols,
                   (size_t)lr * cols * sizeof(double));
        }

        /* Copy required ghost rows back to device buffer */
        double *h_local_buf;
        int buf_rows = local_rows + 2 * halo;
        CUDA_CHECK(cudaMallocHost(&h_local_buf,
                                  (size_t)buf_rows * cols * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(h_local_buf, local_buf,
                              (size_t)buf_rows * cols * sizeof(double),
                              cudaMemcpyDeviceToHost));

        for (int gi = 0; gi < halo; gi++) {
            int grow = (my_start - halo + gi + total_rows) % total_rows;
            memcpy(h_local_buf + (size_t)gi * cols,
                   h_full      + (size_t)grow * cols,
                   (size_t)cols * sizeof(double));
        }
        for (int gi = 0; gi < halo; gi++) {
            int grow = (my_start + local_rows + gi) % total_rows;
            memcpy(h_local_buf + (size_t)(halo + local_rows + gi) * cols,
                   h_full      + (size_t)grow * cols,
                   (size_t)cols * sizeof(double));
        }

        CUDA_CHECK(cudaMemcpy(local_buf, h_local_buf,
                              (size_t)buf_rows * cols * sizeof(double),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaFreeHost(h_local_buf));
        CUDA_CHECK(cudaFreeHost(h_gathered));
        free(h_full);
        CUDA_CHECK(cudaFree(d_full));
    }
}

/* -------------------------------------------------------------------------
 * Manual scatter / gather helpers using NCCL point-to-point
 *
 * NCCL has no native Scatterv/Gatherv, so rank 0 sends each slice
 * individually inside a group operation.
 * ---------------------------------------------------------------------- */
static void nccl_scatterv(const double *d_src,       /* rank 0 only, device */
                          double       *d_dst,        /* all ranks,  device */
                          int          *sendcounts,
                          int          *displs,
                          int           local_rows,
                          int           cols,
                          int           halo,
                          int           myid,
                          int           procs,
                          ncclComm_t    comm,
                          cudaStream_t  stream)
{
    NCCL_CHECK(ncclGroupStart());
    if (myid == 0) {
        for (int r = 0; r < procs; r++) {
            NCCL_CHECK(ncclSend(
                d_src + displs[r],
                (size_t)sendcounts[r], ncclDouble, r, comm, stream));
        }
    }
    /* Every rank (including 0) receives its own slice */
    NCCL_CHECK(ncclRecv(
        d_dst + (size_t)halo * cols,
        (size_t)local_rows * cols, ncclDouble, 0, comm, stream));
    NCCL_CHECK(ncclGroupEnd());
}

static void nccl_gatherv(const double *d_src,       /* all ranks,  device */
                         double       *d_dst,        /* rank 0 only, device */
                         int          *sendcounts,
                         int          *displs,
                         int           local_rows,
                         int           cols,
                         int           halo,
                         int           myid,
                         int           procs,
                         ncclComm_t    comm,
                         cudaStream_t  stream)
{
    NCCL_CHECK(ncclGroupStart());
    /* Every rank sends its real rows to rank 0 */
    NCCL_CHECK(ncclSend(
        d_src + (size_t)halo * cols,
        (size_t)local_rows * cols, ncclDouble, 0, comm, stream));
    if (myid == 0) {
        for (int r = 0; r < procs; r++) {
            NCCL_CHECK(ncclRecv(
                d_dst + displs[r],
                (size_t)sendcounts[r], ncclDouble, r, comm, stream));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
}

/* -------------------------------------------------------------------------
 * Main entry point – NCCL+CUDA port of evolve_lenia_mpi()
 * ---------------------------------------------------------------------- */
extern "C"
double *evolve_lenia_nccl(const unsigned int rows,
                          const unsigned int cols,
                          const unsigned int steps,
                          const double       dt,
                          const unsigned int kernel_size,
                          const struct orbium_coo *orbiums,
                          const unsigned int num_orbiums,
                          int generate_gif)
{
    /* --- Rank / size from MPI (used only for coordination) -------------- */
    int myid, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    /* --- Assign one GPU per rank --------------------------------------- */
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    CUDA_CHECK(cudaSetDevice(myid % num_gpus));

    /* --- Bootstrap NCCL communicator ----------------------------------- */
    ncclUniqueId nccl_id;
    if (myid == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t   comm;
    cudaStream_t stream;
    NCCL_CHECK(ncclCommInitRank(&comm, procs, nccl_id, myid));
    CUDA_CHECK(cudaStreamCreate(&stream));

    int halo = (int)kernel_size / 2;

    /* --- Row distribution (same as MPI version) ------------------------ */
    int *sendcounts = (int *)malloc(procs * sizeof(int));
    int *displs     = (int *)malloc(procs * sizeof(int));
    int rem = (int)rows % procs;
    displs[0] = 0;
    for (int i = 0; i < procs; i++) {
        sendcounts[i] = ((int)rows / procs + (i < rem ? 1 : 0)) * (int)cols;
        if (i > 0) displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
    int local_rows = sendcounts[myid] / (int)cols;
    int my_start   = displs[myid]     / (int)cols;

    /* --- Device buffers ------------------------------------------------ */
    int buf_rows = local_rows + 2 * halo;

    double *d_local_buf, *d_local_tmp, *d_kernel_w;
    CUDA_CHECK(cudaMalloc(&d_local_buf,
                          (size_t)buf_rows * cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_tmp,
                          (size_t)local_rows * cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_kernel_w,
                          (size_t)kernel_size * kernel_size * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_local_buf, 0,
                          (size_t)buf_rows * cols * sizeof(double)));

    /* --- Build and upload kernel --------------------------------------- */
    {
        double *h_kernel = (double *)calloc(kernel_size * kernel_size,
                                            sizeof(double));
        generate_kernel(h_kernel, kernel_size);   /* your existing host fn */
        CUDA_CHECK(cudaMemcpy(d_kernel_w, h_kernel,
                              kernel_size * kernel_size * sizeof(double),
                              cudaMemcpyHostToDevice));
        free(h_kernel);
    }

    /* --- Root: initialise full world on device and scatter ------------- */
    double *d_world = NULL;
    if (myid == 0) {
        double *h_world = (double *)calloc((size_t)rows * cols, sizeof(double));
        for (unsigned int o = 0; o < num_orbiums; o++)
            place_orbium(h_world, rows, cols,
                         (unsigned int)orbiums[o].row,
                         (unsigned int)orbiums[o].col,
                         (unsigned int)orbiums[o].angle);

        CUDA_CHECK(cudaMalloc(&d_world,
                              (size_t)rows * cols * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_world, h_world,
                              (size_t)rows * cols * sizeof(double),
                              cudaMemcpyHostToDevice));
        free(h_world);
    }

    /* Scatter initial rows to all GPUs */
    nccl_scatterv(d_world, d_local_buf,
                  sendcounts, displs,
                  local_rows, (int)cols, halo,
                  myid, procs, comm, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* --- Launch configuration for 2-D kernels ------------------------- */
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (local_rows + block.y - 1) / block.y);

    /* --- Optional GIF (host, rank 0 only) ----------------------------- */
    ge_GIF *gif = NULL;
    double  *h_frame = NULL;   /* pinned staging buffer for GIF frames    */
    uint8_t *h_pixels = NULL;
    if (generate_gif && myid == 0) {
        gif = ge_new_gif("lenia.gif",
                         (uint16_t)cols, (uint16_t)rows,
                         inferno_pallete, 8, -1, 0);
        CUDA_CHECK(cudaMallocHost(&h_frame,
                                  (size_t)rows * cols * sizeof(double)));
        h_pixels = (uint8_t *)malloc((size_t)rows * cols);
    }

    /* --- Main simulation loop ----------------------------------------- */
    for (unsigned int step = 0; step < steps; step++) {

        /* 1. Exchange halos via NCCL */
        exchange_halos_nccl(d_local_buf, local_rows, (int)cols, halo,
                            sendcounts, displs, my_start, (int)rows,
                            myid, procs, comm, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        /* 2. Convolve on GPU */
        convolve_kernel<<<grid, block, 0, stream>>>(
            d_local_tmp, d_local_buf,
            local_rows, (int)cols,
            d_kernel_w, (int)kernel_size);
        CUDA_CHECK(cudaGetLastError());

        /* 3. Apply growth and clip */
        apply_growth_kernel<<<grid, block, 0, stream>>>(
            d_local_buf, d_local_tmp,
            local_rows, (int)cols, halo, dt);
        CUDA_CHECK(cudaGetLastError());

        /* 4. Gather to rank 0 and write GIF frame */
        if (generate_gif) {
            nccl_gatherv(d_local_buf, d_world,
                         sendcounts, displs,
                         local_rows, (int)cols, halo,
                         myid, procs, comm, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (myid == 0 && gif != NULL) {
                CUDA_CHECK(cudaMemcpy(h_frame, d_world,
                                      (size_t)rows * cols * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                for (unsigned int px = 0; px < rows * cols; px++)
                    h_pixels[px] = (uint8_t)(h_frame[px] * 255.0);
                for (unsigned int px = 0; px < rows * cols; px++)
                    gif->frame[px] = h_pixels[px];
                ge_add_frame(gif, 5);
            }
        }
    }

    /* --- Final gather to rank 0 --------------------------------------- */
    if (!generate_gif) {
        nccl_gatherv(d_local_buf, d_world,
                     sendcounts, displs,
                     local_rows, (int)cols, halo,
                     myid, procs, comm, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (gif != NULL) ge_close_gif(gif);

    /* --- Copy final result to host (rank 0 only) ---------------------- */
    double *h_result = NULL;
    if (myid == 0) {
        h_result = (double *)malloc((size_t)rows * cols * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_result, d_world,
                              (size_t)rows * cols * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_world));
    }

    /* --- Cleanup ------------------------------------------------------- */
    CUDA_CHECK(cudaFree(d_local_buf));
    CUDA_CHECK(cudaFree(d_local_tmp));
    CUDA_CHECK(cudaFree(d_kernel_w));
    if (h_frame)  CUDA_CHECK(cudaFreeHost(h_frame));
    if (h_pixels) free(h_pixels);
    free(sendcounts);
    free(displs);

    NCCL_CHECK(ncclCommDestroy(comm));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return h_result;  /* non-NULL only on rank 0; caller must free() */
}