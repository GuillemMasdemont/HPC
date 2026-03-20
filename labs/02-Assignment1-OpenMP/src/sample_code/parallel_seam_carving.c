// Parallel Seam Carving with OpenMP
// Compile: gcc -O3 -fopenmp parallel_seam_carving.c -lm -o parallel_seam_carving
// Single run: ./parallel_seam_carving input.png output.png 128 8
// Benchmark: ./parallel_seam_carving --benchmark ../test_images 128 5 results.csv

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define COLOR_CHANNELS 0
#define ARRAY_SIZE(arr) ((int)(sizeof(arr) / sizeof((arr)[0])))

typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
    int stride;
} Image;

static const char *kBenchmarkImages[] = {
    "720x480.png",
    "1024x768.png",
    "1920x1200.png",
    "3840x2160.png",
    "7680x4320.png"
};

static const int kThreadCounts[] = {1, 2, 4, 8, 16, 32};

static int clamp_index(int value, int max_value) {
    if (value < 0) return 0;
    if (value >= max_value) return max_value - 1;
    return value;
}

static int min3(int a, int b, int c) {
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    return m;
}

static int saturating_add_nonnegative(int a, int b) {
    if (a >= INT_MAX - b) {
        return INT_MAX;
    }
    return a + b;
}

static inline size_t image_offset(int x, int y, int stride, int channels, int c) {
    return (((size_t)y * (size_t)stride) + (size_t)x) * (size_t)channels + (size_t)c;
}

static int parse_int_arg(const char *text, int min_value, const char *name, int *out_value) {
    char *end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < min_value || value > INT_MAX) {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
        return 0;
    }

    *out_value = (int)value;
    return 1;
}

static int load_image_file(const char *file_path, Image *image) {
    image->data = stbi_load(file_path, &image->width, &image->height, &image->channels, COLOR_CHANNELS);
    if (image->data == NULL) {
        fprintf(stderr, "Failed to load %s: %s\n", file_path, stbi_failure_reason());
        return 0;
    }

    image->stride = image->width;
    return 1;
}

static void release_image(Image *image) {
    if (image->data != NULL) {
        stbi_image_free(image->data);
        image->data = NULL;
    }
}

static unsigned char *copy_image_data(const Image *image) {
    size_t bytes = (size_t)image->stride * (size_t)image->height * (size_t)image->channels;
    unsigned char *copy = malloc(bytes);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, image->data, bytes);
    return copy;
}

static void compute_energy_sequential(
    const unsigned char *image,
    int stride,
    int width,
    int height,
    int channels,
    int *energy
) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int xm1 = clamp_index(x - 1, width);
            int xp1 = clamp_index(x + 1, width);
            int ym1 = clamp_index(y - 1, height);
            int yp1 = clamp_index(y + 1, height);

            double gx_total = 0.0;
            double gy_total = 0.0;

            for (int c = 0; c < channels; ++c) {
                int p00 = image[image_offset(xm1, ym1, stride, channels, c)];
                int p01 = image[image_offset(x, ym1, stride, channels, c)];
                int p02 = image[image_offset(xp1, ym1, stride, channels, c)];
                int p10 = image[image_offset(xm1, y, stride, channels, c)];
                int p12 = image[image_offset(xp1, y, stride, channels, c)];
                int p20 = image[image_offset(xm1, yp1, stride, channels, c)];
                int p21 = image[image_offset(x, yp1, stride, channels, c)];
                int p22 = image[image_offset(xp1, yp1, stride, channels, c)];

                int gx = -p00 + p02 - (2 * p10) + (2 * p12) - p20 + p22;
                int gy = p00 + (2 * p01) + p02 - p20 - (2 * p21) - p22;

                gx_total += (double)gx;
                gy_total += (double)gy;
            }

            energy[y * width + x] = (int)(sqrt(gx_total * gx_total + gy_total * gy_total) / (double)channels);
        }
    }
}

static void compute_energy_parallel(
    const unsigned char *image,
    int stride,
    int width,
    int height,
    int channels,
    int *energy
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int xm1 = clamp_index(x - 1, width);
            int xp1 = clamp_index(x + 1, width);
            int ym1 = clamp_index(y - 1, height);
            int yp1 = clamp_index(y + 1, height);

            double gx_total = 0.0;
            double gy_total = 0.0;

            for (int c = 0; c < channels; ++c) {
                int p00 = image[image_offset(xm1, ym1, stride, channels, c)];
                int p01 = image[image_offset(x, ym1, stride, channels, c)];
                int p02 = image[image_offset(xp1, ym1, stride, channels, c)];
                int p10 = image[image_offset(xm1, y, stride, channels, c)];
                int p12 = image[image_offset(xp1, y, stride, channels, c)];
                int p20 = image[image_offset(xm1, yp1, stride, channels, c)];
                int p21 = image[image_offset(x, yp1, stride, channels, c)];
                int p22 = image[image_offset(xp1, yp1, stride, channels, c)];

                int gx = -p00 + p02 - (2 * p10) + (2 * p12) - p20 + p22;
                int gy = p00 + (2 * p01) + p02 - p20 - (2 * p21) - p22;

                gx_total += (double)gx;
                gy_total += (double)gy;
            }

            energy[y * width + x] = (int)(sqrt(gx_total * gx_total + gy_total * gy_total) / (double)channels);
        }
    }
}

static void compute_cumulative_sequential(const int *energy, int *cumulative, int width, int height) {
    for (int x = 0; x < width; ++x) {
        cumulative[x] = energy[x];
    }

    for (int y = 1; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int left = cumulative[(y - 1) * width + clamp_index(x - 1, width)];
            int mid = cumulative[(y - 1) * width + x];
            int right = cumulative[(y - 1) * width + clamp_index(x + 1, width)];
            cumulative[y * width + x] = saturating_add_nonnegative(energy[y * width + x], min3(left, mid, right));
        }
    }
}

static void compute_cumulative_basic_parallel(const int *energy, int *cumulative, int width, int height) {
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int x = 0; x < width; ++x) {
            cumulative[x] = energy[x];
        }

        for (int y = 1; y < height; ++y) {
            #pragma omp for schedule(static)
            for (int x = 0; x < width; ++x) {
                int left = cumulative[(y - 1) * width + clamp_index(x - 1, width)];
                int mid = cumulative[(y - 1) * width + x];
                int right = cumulative[(y - 1) * width + clamp_index(x + 1, width)];
                cumulative[y * width + x] = saturating_add_nonnegative(energy[y * width + x], min3(left, mid, right));
            }
        }
    }
}

static int argmin_bottom_row_sequential(const int *cumulative, int width, int height) {
    int row = (height - 1) * width;
    int best_x = 0;
    int best_value = INT_MAX;

    for (int x = 0; x < width; ++x) {
        int value = cumulative[row + x];
        if (value < best_value) {
            best_value = value;
            best_x = x;
        }
    }

    return best_x;
}

static int argmin_bottom_row_parallel(const int *cumulative, int width, int height) {
    int row = (height - 1) * width;
    int best_x = 0;
    int best_value = INT_MAX;

    #pragma omp parallel
    {
        int local_best_x = 0;
        int local_best_value = INT_MAX;

        #pragma omp for nowait
        for (int x = 0; x < width; ++x) {
            int value = cumulative[row + x];
            if (value < local_best_value || (value == local_best_value && x < local_best_x)) {
                local_best_value = value;
                local_best_x = x;
            }
        }

        #pragma omp critical
        {
            if (local_best_value < best_value || (local_best_value == best_value && local_best_x < best_x)) {
                best_value = local_best_value;
                best_x = local_best_x;
            }
        }
    }

    return best_x;
}

static void backtrack_seam(const int *cumulative, int width, int height, int start_x, int *seam_path) {
    seam_path[height - 1] = start_x;

    for (int y = height - 2; y >= 0; --y) {
        int prev_x = seam_path[y + 1];
        int left = clamp_index(prev_x - 1, width);
        int mid = prev_x;
        int right = clamp_index(prev_x + 1, width);

        int best_x = left;
        int best_value = cumulative[y * width + left];

        int mid_value = cumulative[y * width + mid];
        if (mid_value < best_value) {
            best_value = mid_value;
            best_x = mid;
        }

        int right_value = cumulative[y * width + right];
        if (right_value < best_value) {
            best_x = right;
        }

        seam_path[y] = best_x;
    }
}

static void trace_seam_sequential(const int *cumulative, int width, int height, int *seam_path) {
    int start_x = argmin_bottom_row_sequential(cumulative, width, height);
    backtrack_seam(cumulative, width, height, start_x, seam_path);
}

static void trace_seam_parallel(const int *cumulative, int width, int height, int *seam_path) {
    int start_x = argmin_bottom_row_parallel(cumulative, width, height);
    backtrack_seam(cumulative, width, height, start_x, seam_path);
}

static void remove_seam_copy_sequential(
    unsigned char *image,
    int stride,
    int width,
    int height,
    int channels,
    const int *seam_path
) {
    for (int y = 0; y < height; ++y) {
        int seam_x = seam_path[y];
        unsigned char *row = image + (size_t)y * (size_t)stride * (size_t)channels;
        size_t bytes_to_move = (size_t)(width - seam_x - 1) * (size_t)channels;

        if (bytes_to_move > 0) {
            memmove(
                row + (size_t)seam_x * (size_t)channels,
                row + (size_t)(seam_x + 1) * (size_t)channels,
                bytes_to_move
            );
        }
    }
}

static void remove_seam_copy_parallel(
    unsigned char *image,
    int stride,
    int width,
    int height,
    int channels,
    const int *seam_path
) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        int seam_x = seam_path[y];
        unsigned char *row = image + (size_t)y * (size_t)stride * (size_t)channels;
        size_t bytes_to_move = (size_t)(width - seam_x - 1) * (size_t)channels;

        if (bytes_to_move > 0) {
            memmove(
                row + (size_t)seam_x * (size_t)channels,
                row + (size_t)(seam_x + 1) * (size_t)channels,
                bytes_to_move
            );
        }
    }
}

static int seam_overlaps_mask(const int *seam_path, const unsigned char *removed_mask, int width, int height) {
    for (int y = 0; y < height; ++y) {
        if (removed_mask[(size_t)y * (size_t)width + (size_t)seam_path[y]] != 0) {
            return 1;
        }
    }

    return 0;
}

static void mark_seam_mask(unsigned char *removed_mask, const int *seam_path, int width, int height) {
    for (int y = 0; y < height; ++y) {
        removed_mask[(size_t)y * (size_t)width + (size_t)seam_path[y]] = 1;
    }
}

static void block_seam_energy(int *energy, const int *seam_path, int width, int height) {
    const int blocked_cost = INT_MAX / 4;

    for (int y = 0; y < height; ++y) {
        energy[(size_t)y * (size_t)width + (size_t)seam_path[y]] = blocked_cost;
    }
}

static void compact_rows_parallel(
    unsigned char *image,
    int stride,
    int width,
    int height,
    int channels,
    const unsigned char *removed_mask
) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        unsigned char *row = image + (size_t)y * (size_t)stride * (size_t)channels;
        const unsigned char *mask_row = removed_mask + (size_t)y * (size_t)width;
        int write_x = 0;

        for (int x = 0; x < width; ++x) {
            if (mask_row[x] == 0) {
                if (write_x != x) {
                    memcpy(
                        row + (size_t)write_x * (size_t)channels,
                        row + (size_t)x * (size_t)channels,
                        (size_t)channels
                    );
                }
                ++write_x;
            }
        }
    }
}

static int remove_seams_batch_greedy(
    unsigned char *image,
    int stride,
    int width,
    int height,
    int channels,
    int batch_limit,
    int *energy,
    int *cumulative,
    int *seam_path,
    unsigned char *removed_mask
) {
    int accepted = 0;

    if (batch_limit <= 0 || width <= 1) {
        return 0;
    }

    memset(removed_mask, 0, (size_t)width * (size_t)height);

    while (accepted < batch_limit && width - accepted > 1) {
        compute_cumulative_sequential(energy, cumulative, width, height);
        trace_seam_sequential(cumulative, width, height, seam_path);

        if (seam_overlaps_mask(seam_path, removed_mask, width, height)) {
            break;
        }

        mark_seam_mask(removed_mask, seam_path, width, height);
        block_seam_energy(energy, seam_path, width, height);
        ++accepted;
    }

    if (accepted > 0) {
        compact_rows_parallel(image, stride, width, height, channels, removed_mask);
    }

    return accepted;
}

static double carve_vertical_seams(
    unsigned char *image,
    int stride,
    int height,
    int channels,
    int seams_to_remove,
    int num_threads,
    int use_parallel,
    int *final_width,
    int use_greedy_seam_removal
) {
    int width = stride;
    size_t max_pixels = (size_t)stride * (size_t)height;
    int *energy = NULL;
    int *cumulative = NULL;
    int *seam_path = NULL;
    unsigned char *removed_mask = NULL;

    if (seams_to_remove < 0) {
        seams_to_remove = 0;
    }
    if (seams_to_remove > width - 1) {
        seams_to_remove = width - 1;
    }

    energy = malloc(max_pixels * sizeof(int));
    cumulative = malloc(max_pixels * sizeof(int));
    seam_path = malloc((size_t)height * sizeof(int));
    removed_mask = malloc(max_pixels * sizeof(unsigned char));

    if (energy == NULL || cumulative == NULL || seam_path == NULL || removed_mask == NULL) {
        fprintf(stderr, "Allocation failed while preparing seam carving buffers\n");
        free(energy);
        free(cumulative);
        free(seam_path);
        free(removed_mask);
        *final_width = width;
        return -1.0;
    }

    if (use_parallel) {
        omp_set_num_threads(num_threads);
    }

    double t_start = omp_get_wtime();

    if (use_greedy_seam_removal) {
        const int batch_size = 16;

        while (seams_to_remove > 0 && width > 1) {
            int batch_limit = batch_size;
            int removed;

            if (batch_limit > seams_to_remove) {
                batch_limit = seams_to_remove;
            }
            if (batch_limit > width - 1) {
                batch_limit = width - 1;
            }

            if (use_parallel) {
                compute_energy_parallel(image, stride, width, height, channels, energy);
            } else {
                compute_energy_sequential(image, stride, width, height, channels, energy);
            }

            removed = remove_seams_batch_greedy(
                image,
                stride,
                width,
                height,
                channels,
                batch_limit,
                energy,
                cumulative,
                seam_path,
                removed_mask
            );

            if (removed <= 0) {
                if (use_parallel) {
                    compute_cumulative_basic_parallel(energy, cumulative, width, height);
                } else {
                    compute_cumulative_sequential(energy, cumulative, width, height);
                }
                trace_seam_sequential(cumulative, width, height, seam_path);
                remove_seam_copy_sequential(image, stride, width, height, channels, seam_path);
                removed = 1;
            }

            width -= removed;
            seams_to_remove -= removed;
        }
    } else {
        for (int seam = 0; seam < seams_to_remove && width > 1; ++seam) {
            if (use_parallel) {
                compute_energy_parallel(image, stride, width, height, channels, energy);
                compute_cumulative_basic_parallel(energy, cumulative, width, height);
                trace_seam_parallel(cumulative, width, height, seam_path);
                remove_seam_copy_parallel(image, stride, width, height, channels, seam_path);
            } else {
                compute_energy_sequential(image, stride, width, height, channels, energy);
                compute_cumulative_sequential(energy, cumulative, width, height);
                trace_seam_sequential(cumulative, width, height, seam_path);
                remove_seam_copy_sequential(image, stride, width, height, channels, seam_path);
            }

            width--;
        }
    }

    double elapsed = omp_get_wtime() - t_start;

    free(energy);
    free(cumulative);
    free(seam_path);
    free(removed_mask);

    *final_width = width;
    return elapsed;
}

static int run_benchmark_mode(const char *image_dir, int seams_to_remove, int runs, const char *csv_path) {
    FILE *csv = NULL;
    int failed = 0;

    if (csv_path != NULL) {
        csv = fopen(csv_path, "w");
        if (csv == NULL) {
            fprintf(stderr, "Failed to open CSV output file %s\n", csv_path);
            return 1;
        }
        fprintf(csv, "image,width,height,seams,runs,threads,t_seq_avg_sec,t_par_avg_sec,speedup\n");
    }

    printf("Benchmark settings: seams=%d, runs=%d\n", seams_to_remove, runs);
    printf("Use cluster flags such as --hint=nomultithread and OMP_PLACES=cores for stable measurements.\n");

    for (int i = 0; i < ARRAY_SIZE(kBenchmarkImages); ++i) {
        char image_path[1024];
        Image image = {0};
        int seams_for_this_image;

        snprintf(image_path, sizeof(image_path), "%s/%s", image_dir, kBenchmarkImages[i]);

        if (!load_image_file(image_path, &image)) {
            failed = 1;
            continue;
        }

        seams_for_this_image = seams_to_remove;
        if (seams_for_this_image > image.width - 1) {
            seams_for_this_image = image.width - 1;
        }

        double seq_sum = 0.0;
        for (int run = 0; run < runs; ++run) {
            unsigned char *work = copy_image_data(&image);
            int final_width = image.width;
            double elapsed;

            if (work == NULL) {
                fprintf(stderr, "Allocation failed while copying image for sequential benchmark\n");
                failed = 1;
                break;
            }

            elapsed = carve_vertical_seams(
                work,
                image.stride,
                image.height,
                image.channels,
                seams_for_this_image,
                1,
                0,
                &final_width,
                0
            );
            free(work);

            if (elapsed < 0.0) {
                failed = 1;
                break;
            }

            seq_sum += elapsed;
        }

        if (failed) {
            release_image(&image);
            continue;
        }

        double t_seq = seq_sum / (double)runs;

        printf("\nImage: %s (%dx%d, channels=%d), seams=%d\n",
               kBenchmarkImages[i], image.width, image.height, image.channels, seams_for_this_image);
        printf("Sequential average: %.6f s\n", t_seq);

        for (int t = 0; t < ARRAY_SIZE(kThreadCounts); ++t) {
            int threads = kThreadCounts[t];
            double par_sum = 0.0;

            for (int run = 0; run < runs; ++run) {
                unsigned char *work = copy_image_data(&image);
                int final_width = image.width;
                double elapsed;

                if (work == NULL) {
                    fprintf(stderr, "Allocation failed while copying image for parallel benchmark\n");
                    failed = 1;
                    break;
                }

                elapsed = carve_vertical_seams(
                    work,
                    image.stride,
                    image.height,
                    image.channels,
                    seams_for_this_image,
                    threads,
                    1,
                    &final_width,
                    0
                );
                free(work);

                if (elapsed < 0.0) {
                    failed = 1;
                    break;
                }

                par_sum += elapsed;
            }

            if (failed) {
                break;
            }

            double t_par = par_sum / (double)runs;
            double speedup = (t_par > 0.0) ? (t_seq / t_par) : 0.0;

            printf("  threads=%2d  parallel_avg=%.6f s  speedup=%.3f\n", threads, t_par, speedup);

            if (csv != NULL) {
                fprintf(csv,
                        "%s,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f\n",
                        kBenchmarkImages[i],
                        image.width,
                        image.height,
                        seams_for_this_image,
                        runs,
                        threads,
                        t_seq,
                        t_par,
                        speedup);
            }
        }

        release_image(&image);
    }

    if (csv != NULL) {
        fclose(csv);
    }

    return failed ? 1 : 0;
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s <input.png> <output.png> <seams_to_remove> [threads]\n"
            "  %s --benchmark <test_images_dir> [seams=128] [runs=5] [csv_output=benchmark_results.csv]\n"
            "\n"
            "Compile with: gcc -O3 -fopenmp parallel_seam_carving.c -lm -o parallel_seam_carving\n",
            program,
            program);
}

int main(int argc, char *argv[]) {
    if (argc >= 2 && strcmp(argv[1], "--benchmark") == 0) {
        int seams_to_remove = 128;
        int runs = 5;
        const char *csv_output = "benchmark_results.csv";

        if (argc < 3 || argc > 6) {
            print_usage(argv[0]);
            return 1;
        }

        if (argc >= 4 && !parse_int_arg(argv[3], 0, "seams_to_remove", &seams_to_remove)) {
            return 1;
        }
        if (argc >= 5 && !parse_int_arg(argv[4], 1, "runs", &runs)) {
            return 1;
        }
        if (argc >= 6) {
            csv_output = argv[5];
        }

        return run_benchmark_mode(argv[2], seams_to_remove, runs, csv_output);
    }

    if (argc < 4 || argc > 5) {
        print_usage(argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    int seams_to_remove = 0;
    int num_threads = omp_get_max_threads();

    if (!parse_int_arg(argv[3], 0, "seams_to_remove", &seams_to_remove)) {
        return 1;
    }

    if (argc == 5 && !parse_int_arg(argv[4], 1, "threads", &num_threads)) {
        return 1;
    }

    Image image = {0};
    if (!load_image_file(input_path, &image)) {
        return 1;
    }

    int seams_effective = seams_to_remove;
    if (seams_effective > image.width - 1) {
        seams_effective = image.width - 1;
    }

    int final_width = image.width;
    double elapsed = carve_vertical_seams(
        image.data,
        image.stride,
        image.height,
        image.channels,
        seams_effective,
        num_threads,
        0,
        &final_width,
        0
    );

    if (elapsed < 0.0) {
        release_image(&image);
        return 1;
    }

    if (!stbi_write_png(output_path, final_width, image.height, image.channels, image.data, image.stride * image.channels)) {
        fprintf(stderr, "Failed to write output image %s\n", output_path);
        release_image(&image);
        return 1;
    }

    printf("Input: %s (%dx%d, channels=%d)\n", input_path, image.width, image.height, image.channels);
    printf("Output: %s (%dx%d)\n", output_path, final_width, image.height);
    printf("Seams removed: %d\n", seams_effective);
    printf("Threads: %d\n", num_threads);
    printf("Elapsed: %.6f seconds\n", elapsed);

    release_image(&image);
    return 0;
}
