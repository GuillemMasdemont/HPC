#include <stdio.h> 
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define COLOR_CHANNELS 0
#define MAX_FILENAME 255

int clamp(int val, int max) {
    if (val < 0) return 0;
    if (val >= max) return max - 1;
    return val;
}

int min3(int a, int b, int c) {
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    return m;
}

int main() {

    int width, height, cpp; 

    char image_in_name[MAX_FILENAME] = "valve.png";

    // use unsigned char as it occupies 1 Byte of memory -> 256 different values. 
    unsigned char *image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);


    if (image_in == NULL) {
        printf("Loading failed :(", stbi_failure_reason());
        // " " for strings and ' ' for characters. 
        return 1;
    }

    printf("Loaded: %d x %d with %d channels\n", width, height, cpp);
    
    int original_width = width;
    int original_height = height;
    int original_cpp = cpp; 

    // The image is stored row by row from left to right, and for each pixel we have R, G, B. 
    // Accesing a pixel: 

    unsigned char *image_out = malloc(width * height * sizeof(unsigned char));

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            
            float total_Gx = 0;
            float total_Gy = 0;

            for (int c = 0; c < cpp; c++) {

                // we present from leaving the borders. 
                int y_prev = clamp(y - 1, height);
                int y_next = clamp(y + 1, height);
                int x_prev = clamp(x - 1, width);
                int x_next = clamp(x + 1, width);

                float Gx = 
                -1 * image_in[(y_prev * width * cpp) + (x_prev * cpp) + c] +
                 1 * image_in[(y_prev * width * cpp) + (x_next * cpp) + c] +
                -2 * image_in[(y      * width * cpp) + (x_prev * cpp) + c] +
                 2 * image_in[(y      * width * cpp) + (x_next * cpp) + c] +
                -1 * image_in[(y_next * width * cpp) + (x_prev * cpp) + c] +
                 1 * image_in[(y_next * width * cpp) + (x_next * cpp) + c];
                
                float Gy = 
                 1 * image_in[(y_prev * width * cpp) + (x_prev * cpp) + c] +
                 2 * image_in[(y_prev * width * cpp) + (x      * cpp) + c] +
                 1 * image_in[(y_prev * width * cpp) + (x_next * cpp) + c] +
                -1 * image_in[(y_next * width * cpp) + (x_prev * cpp) + c] +
                -2 * image_in[(y_next * width * cpp) + (x      * cpp) + c] +
                -1 * image_in[(y_next * width * cpp) + (x_next * cpp) + c];


                total_Gx += Gx;
                total_Gy += Gy;
            }
            float energy_x_y = sqrtf((total_Gx*total_Gx) + (total_Gy*total_Gy));
            int final_energy = (int)(energy_x_y / cpp);
            if (final_energy > 255) final_energy = 255;    
        
            image_out[width * y + x] = final_energy;
            //equivalent *image_out + width * y + x 
        }
    }

    stbi_write_png("energyvalve.png", width, height, 1, image_out, width * 1); 
    
    //SEAM carving  

    // create a new vector with a 4 Byte capacity instead of 1 because path overflow for energy paths + copy the initial values. 
    int *cumulative_energy = malloc(width * height * sizeof(int));  
    for (int i = 0; i < width * height; i++) {
        cumulative_energy[i] = (int)image_out[i];
    }

    // perform 128 seams. 

    int n_seams = 128; 

    for (int i = 0; i < n_seams; i ++) { 

        //compute the energy of the paths 
        for (int x = 0; x < width; x++) {
            for (int y = 1; y < height; y++) {
                
                int x_left = clamp(x - 1 , width);
                int x_right = clamp(x + 1, width);

                cumulative_energy[y * width + x] += 
                    min3(
                        image_out[(y-1) * width + x_left], 
                        image_out[(y-1) * width + x], 
                        image_out[(y-1) * width + x_right] 
                    );
            }
        }

        //perform seam carving

        // find minimum path energy
        int seam_path[height]; //vector to store the path (for each height the corresponding pixel index)
        int current_x = 0;
        int min_energy = INT_MAX;

        //find starting point  
        for (int x = 0; x < width; x++) {
            if (cumulative_energy[(height - 1) * width + x] < min_energy) {
                min_energy = cumulative_energy[(height - 1) * width + x];
                current_x = x;
            }
        }

        //perform the algorithm 
        seam_path[height - 1] = current_x;
        for (int y = height - 1; y > 0; y--) {

            int left  = clamp(current_x - 1, width);
            int mid   = current_x;
            int right = clamp(current_x + 1, width);

            int val_l = cumulative_energy[(y - 1) * width + left];
            int val_m = cumulative_energy[(y - 1) * width + mid];
            int val_r = cumulative_energy[(y - 1) * width + right];

            if (val_l <= val_m && val_l <= val_r) current_x = left;
            else if (val_m <= val_l && val_m <= val_r) current_x = mid;
            else current_x = right;

            seam_path[y - 1] = current_x;
        }

        //once we have a path, copy adjacent row
        for (int y = 0; y < height; y++) {
            int target_x = seam_path[y];
            for (int x = target_x; x < width - 1; x++) {
                for (int c = 0; c < cpp; c++) {
                    image_in[y * original_width * cpp + x * cpp + c] = 
                    image_in[y * original_width * cpp + (x + 1) * cpp + c];
                }
            }
        }

        width--;
    }

    // I output a image of dimension width x height, although 
    // the stride to jump to the next row is: original_width * cpp.

    stbi_write_png("valvecarved.png", width, height, cpp, image_in, original_width * cpp); 

    
    stbi_image_free(image_in);
    
    // normally return 0 means success 
    return 0; 
}