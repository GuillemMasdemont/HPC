#include <stdio.h> 
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

static int clamp(int val, int max) {
    if (val < 0) return 0;
    if (val >= max) return max - 1;
    return val;
}

static int min3(int a, int b, int c) {
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    return m;
}

static int pixel_index(int x, int y, int width, int cpp, int c) {
    return (y * width * cpp) + (x * cpp) + c;
}

static unsigned char *load_image_or_null(const char *image_name, int *width, int *height, int *cpp) {
    unsigned char *image = stbi_load(image_name, width, height, cpp, COLOR_CHANNELS);
    if (image == NULL) {
        printf("Loading failed: %s\n", stbi_failure_reason());
    }
    return image;
}

static unsigned char compute_energy_at(
    const unsigned char *image,
    int width,
    int height,
    int cpp,
    int x,
    int y
) {
    float total_Gx = 0.0f;
    float total_Gy = 0.0f;

    int y_prev = clamp(y - 1, height);
    int y_next = clamp(y + 1, height);
    int x_prev = clamp(x - 1, width);
    int x_next = clamp(x + 1, width);

    for (int c = 0; c < cpp; c++) {
        float Gx =
            -1 * image[pixel_index(x_prev, y_prev, width, cpp, c)] +
             1 * image[pixel_index(x_next, y_prev, width, cpp, c)] +
            -2 * image[pixel_index(x_prev, y,      width, cpp, c)] +
             2 * image[pixel_index(x_next, y,      width, cpp, c)] +
            -1 * image[pixel_index(x_prev, y_next, width, cpp, c)] +
             1 * image[pixel_index(x_next, y_next, width, cpp, c)];

        float Gy =
             1 * image[pixel_index(x_prev, y_prev, width, cpp, c)] +
             2 * image[pixel_index(x,      y_prev, width, cpp, c)] +
             1 * image[pixel_index(x_next, y_prev, width, cpp, c)] +
            -1 * image[pixel_index(x_prev, y_next, width, cpp, c)] +
            -2 * image[pixel_index(x,      y_next, width, cpp, c)] +
            -1 * image[pixel_index(x_next, y_next, width, cpp, c)];

        total_Gx += Gx;
        total_Gy += Gy;
    }

    float energy_x_y = sqrtf((total_Gx * total_Gx) + (total_Gy * total_Gy));
    int final_energy = (int)(energy_x_y / cpp);
    if (final_energy > 255) final_energy = 255;
    return (unsigned char)final_energy;
}

static unsigned char *compute_energy_image(const unsigned char *image, int width, int height, int cpp) {
    unsigned char *energy_image = malloc((size_t)width * height * sizeof(unsigned char));
    if (energy_image == NULL) {
        printf("Allocation failed for energy image\n");
        return NULL;
    }

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            energy_image[width * y + x] = compute_energy_at(image, width, height, cpp, x, y);
        }
    }

    return energy_image;
}

static int *create_cumulative_energy(const unsigned char *energy_image, int width, int height) {
    int total = width * height;
    int *cumulative_energy = malloc((size_t)total * sizeof(int));
    if (cumulative_energy == NULL) {
        printf("Allocation failed for cumulative energy\n");
        return NULL;
    }

    for (int i = 0; i < total; i++) {
        cumulative_energy[i] = (int)energy_image[i];
    }

    return cumulative_energy;
}

static void update_cumulative_energy(int *cumulative_energy, const unsigned char *energy_image, int width, int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 1; y < height; y++) {
            int x_left = clamp(x - 1, width);
            int x_right = clamp(x + 1, width);

            cumulative_energy[y * width + x] +=
                min3(
                    energy_image[(y - 1) * width + x_left],
                    energy_image[(y - 1) * width + x],
                    energy_image[(y - 1) * width + x_right]
                );
        }
    }
}

static void find_seam_path(const int *cumulative_energy, int width, int height, int *seam_path) {
    int current_x = 0;
    int min_energy = INT_MAX;

    for (int x = 0; x < width; x++) {
        int candidate = cumulative_energy[(height - 1) * width + x];
        if (candidate < min_energy) {
            min_energy = candidate;
            current_x = x;
        }
    }

    seam_path[height - 1] = current_x;
    for (int y = height - 1; y > 0; y--) {
        int left = clamp(current_x - 1, width);
        int mid = current_x;
        int right = clamp(current_x + 1, width);

        int val_l = cumulative_energy[(y - 1) * width + left];
        int val_m = cumulative_energy[(y - 1) * width + mid];
        int val_r = cumulative_energy[(y - 1) * width + right];

        if (val_l <= val_m && val_l <= val_r) current_x = left;
        else if (val_m <= val_l && val_m <= val_r) current_x = mid;
        else current_x = right;

        seam_path[y - 1] = current_x;
    }
}

static void remove_seam(unsigned char *image, int width, int height, int original_width, int cpp, const int *seam_path) {
    for (int y = 0; y < height; y++) {
        int target_x = seam_path[y];
        unsigned char *row = image + y * original_width * cpp;
        memmove(
            row + target_x * cpp,
            row + (target_x + 1) * cpp,
            (size_t)(width - target_x - 1) * cpp
        );
    }
}

static int carve_vertical_seams(
    unsigned char *image,
    const unsigned char *energy_image,
    int width,
    int height,
    int original_width,
    int cpp,
    int n_seams
) {
    int *cumulative_energy = create_cumulative_energy(energy_image, width, height);
    if (cumulative_energy == NULL) {
        return width;
    }

    int *seam_path = malloc((size_t)height * sizeof(int));
    if (seam_path == NULL) {
        printf("Allocation failed for seam path\n");
        free(cumulative_energy);
        return width;
    }

    for (int i = 0; i < n_seams && width > 1; i++) {
        update_cumulative_energy(cumulative_energy, energy_image, width, height);
        find_seam_path(cumulative_energy, width, height, seam_path);
        remove_seam(image, width, height, original_width, cpp, seam_path);
        width--;
    }

    free(seam_path);
    free(cumulative_energy);
    return width;
}

int main() {

    int width, height, cpp; 

    char image_in_name[MAX_FILENAME] = "valve.png";

    // use unsigned char as it occupies 1 Byte of memory -> 256 different values. 
    unsigned char *image_in = load_image_or_null(image_in_name, &width, &height, &cpp);


    if (image_in == NULL) {
        return 1;
    }

    printf("Loaded: %d x %d with %d channels\n", width, height, cpp);
    
    int original_width = width;

    // The image is stored row by row from left to right, and for each pixel we have R, G, B. 
    // Accesing a pixel: 

    unsigned char *image_out = compute_energy_image(image_in, width, height, cpp);
    if (image_out == NULL) {
        stbi_image_free(image_in);
        return 1;
    }

    stbi_write_png("energyvalve.png", width, height, 1, image_out, width * 1); 
    
    //SEAM carving  

    // perform 128 seams. 

    int n_seams = 128; 
    width = carve_vertical_seams(image_in, image_out, width, height, original_width, cpp, n_seams);

    // I output a image of dimension width x height, although 
    // the stride to jump to the next row is: original_width * cpp.

    stbi_write_png("valvecarved.png", width, height, cpp, image_in, original_width * cpp); 

    free(image_out);
    
    stbi_image_free(image_in);
    
    // normally return 0 means success 
    return 0; 
}