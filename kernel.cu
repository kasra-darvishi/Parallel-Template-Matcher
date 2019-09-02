
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bitmap_image.hpp"
#include <chrono>
#include <windows.h>
#include <string.h>
#include <iostream>
#include <tchar.h>
#include <filesystem>

using namespace std;;

#define BLOCK_SIZE 1024
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \ ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else 
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

void fill_array(long *a, size_t n);
void prefix_sum(long *a, size_t n);
void print_array(long *a, size_t n);

void findeMatches_serial(unsigned char *img, unsigned char *tmp, unsigned char *tmp_rotated, int img_height, int img_width,
	int tmp_height, int tmp_width, int tmp_rotated_height, int tmp_rotated_width) {


	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	unsigned int * errors = (unsigned int *)malloc((img_height - tmp_height + 1) * (img_width - tmp_width + 1) * sizeof(unsigned int));

	for (int j = 0; j < img_height - tmp_height + 1; j++) {
		for (int i = 0; i < img_width - tmp_width + 1; i++) {
			unsigned int err = 0;
			for (int n = 0; n < tmp_height; n++) {
				for (int m = 0; m < tmp_width; m++) {

					for (int clr = 0; clr < 3; clr++)
						err += abs(img[((n + j)*img_width + (i + m)) * 3 + clr] - tmp[(n*tmp_width + m) * 3 + clr]) / 3;

				}
			}
			errors[j*(img_width - tmp_width + 1) + i] = err;
		}
	}

	int maximumPosibleNumberOfMatches = (img_height / tmp_height + 1)*(img_width / tmp_width + 1);
	int * pos_x = (int *)malloc(maximumPosibleNumberOfMatches * sizeof(int));
	int * pos_y = (int *)malloc(maximumPosibleNumberOfMatches * sizeof(int));
	unsigned int * errorOfPos = (unsigned int *)malloc(maximumPosibleNumberOfMatches * sizeof(unsigned int));
	int pointer = 0;
	int curVal;

	for (int y = 0; y < img_height - tmp_height + 1; ++y) {
		for (int x = 0; x < img_width - tmp_width + 1; ++x) {
			curVal = errors[y*(img_width - tmp_width + 1) + x];
			if (curVal < errorOfPos[0] && curVal < 60) {
				pointer = 0;
				pos_x[pointer] = x;
				pos_y[pointer] = y;
				errorOfPos[pointer] = curVal;
				pointer++;
			}
			else if (curVal == errorOfPos[0]) {
				pos_x[pointer] = x;
				pos_y[pointer] = y;
				errorOfPos[pointer] = curVal;
				pointer++;
				if (pointer >= maximumPosibleNumberOfMatches)
					printf("pointer exeeded limit: %d", pointer--);
			}
		}
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

	printf("\n\n Matched positions in serial: \n\n");
	for (size_t i = 0; i < pointer; i++)
		printf("#%d x: %d y: %d error: %d\n", (i + 1), pos_x[i], pos_y[i], errorOfPos[i]);

	free(errors);
}

__global__ void computeError(unsigned char *img, unsigned char *tmp, int img_height, int img_width, int tmp_height,
							int tmp_width, unsigned int *errors) {

	unsigned char local_img[3];
	__shared__ unsigned char shm_tmp[3 * BLOCK_SIZE];
	//__shared__ unsigned char shm_tmp_rotated[3 * BLOCK_SIZE];
	__shared__ int errors_shMem[4];
	int stride_x = gridDim.x*blockDim.x;
	int stride_y = gridDim.y*blockDim.y;

	int errorMatrix_height = img_height - tmp_height + 1;
	int errorMatrix_width = img_width - tmp_width + 1;
	
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int initial_x = x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int initial_y = y;

	int indexInBlock = threadIdx.y*blockDim.x + threadIdx.x;
	int indexInTmp = y*tmp_width + x;

	bool validThread = (x < tmp_width && y < tmp_height);
	
	for (int clr = 0; clr < 3; clr++) {
		if(validThread)
			shm_tmp[indexInBlock*3 + clr] = tmp[indexInTmp*3 + clr];
	}

	for (int count_vert_move = 0; y < img_height; count_vert_move++) {

		if (!validThread /*&& !validThread_rotated*/)
			break;

		x = initial_x;

		for (int count_horiz_move = 0; x < img_width; count_horiz_move++) {

			for (int clr = 0; clr < 3; clr++)
				local_img[clr] = img[(y*img_width + x)*3 + clr];

			if (indexInBlock == 0){
				errors_shMem[0] = 0;
				errors_shMem[1] = 0;
				errors_shMem[2] = 0;
				errors_shMem[3] = 0;
			}
			__syncthreads();

			for (size_t i = 0; i < tmp_width; i++) {
				for (size_t j = 0; j < tmp_height; j++) {

					unsigned int localErr = 0;

					int x0 = (i + (count_horiz_move - 1)*tmp_width);
					int y0 = (j + (count_vert_move - 1)*tmp_height);

					int x1 = (i + count_horiz_move*tmp_width);
					int y1 = (j + (count_vert_move - 1)*tmp_height);

					int x2 = (i + (count_horiz_move - 1)*tmp_width);
					int y2 = (j + count_vert_move*tmp_height);

					int x3 = (i + count_horiz_move*tmp_width);
					int y3 = (j + count_vert_move*tmp_height);

					bool loopIsValid[4];
					loopIsValid[0] = (x0 > 0 && y0 > 0);
					loopIsValid[1] = (x1 < errorMatrix_width && y1 > 0);
					loopIsValid[2] = (x2 > 0 && y2 < errorMatrix_height);
					loopIsValid[3] = (x3 < errorMatrix_width && y3 < errorMatrix_height);
					
					int indexIn_errors_shMem;
					
					if (loopIsValid[0] && (initial_x < i && initial_y < j)) {
						indexInTmp = (initial_y + (tmp_height - j))*tmp_width + (initial_x + (tmp_width - i));
						for (int clr = 0; clr < 3; clr++)
							localErr += abs(local_img[clr] - tmp[indexInTmp * 3 + clr]);
						errors_shMem[0] += localErr / 3;
					}
					else if (loopIsValid[1] && (initial_x >= i && initial_y < j)) {
						indexInTmp = (initial_y + (tmp_height - j))*tmp_width + (initial_x - i);
						for (int clr = 0; clr < 3; clr++)
							localErr += abs(local_img[clr] - tmp[indexInTmp * 3 + clr]);
						errors_shMem[1] += localErr / 3;
					}
					else if (loopIsValid[2] && (initial_x < i && initial_y >= j)) {
						indexInTmp = (initial_y - j)*tmp_width + (initial_x + (tmp_width - i));
						for (int clr = 0; clr < 3; clr++)
							localErr += abs(local_img[clr] - tmp[indexInTmp * 3 + clr]);
						errors_shMem[2] += localErr / 3;
					}
					else if (loopIsValid[3] && (initial_x >= i && initial_y >= j)) {
						indexInTmp = (initial_y - j)*tmp_width + (initial_x - i);
						for (int clr = 0; clr < 3; clr++)
							localErr += abs(local_img[clr] - tmp[indexInTmp * 3 + clr]);
						errors_shMem[3] += localErr / 3;
					}

					__syncthreads();
					if (indexInBlock == 0) {
						if (loopIsValid[0]) {
							errors[y0 * errorMatrix_width + x0] += errors_shMem[0];
							//errors[y0 * errorMatrix_width + x0] += 10;
							errors_shMem[0] = 0;
						}
						if (loopIsValid[1]) {
							errors[y1 * errorMatrix_width + x1] += errors_shMem[1];
							//errors[y1 * errorMatrix_width + x1] += 10;
							errors_shMem[1] = 0;
						}
						if (loopIsValid[2]) {
							errors[y2 * errorMatrix_width + x2] += errors_shMem[2];
							//errors[y2 * errorMatrix_width + x2] += 10;
							errors_shMem[2] = 0;
						}
						if (loopIsValid[3]) {
							errors[y3 * errorMatrix_width + x3] += errors_shMem[3];
							//errors[y3 * errorMatrix_width + x3] += 10;
							errors_shMem[3] = 0;
						}
					}
					__syncthreads();

				}
			}

			x += tmp_width;
		}

		y += tmp_height;
		
	}
	
}

int findMatches(unsigned char *img, unsigned char *tmp, int img_height, int img_width, 
				int tmp_height, int tmp_width, unsigned char *img_inHostMem) {
	
	cudaError_t cudaStatus;
	cudaError_t error;

	if (tmp_height > img_height || tmp_width > img_width)
		return 0;

	int errorsMatrixSize = (img_height - tmp_height + 1) * (img_width - tmp_width + 1) * 3;

	unsigned int * errors_host = (unsigned int *)malloc(errorsMatrixSize * sizeof(unsigned int));
	for (size_t i = 0; i < errorsMatrixSize; i++){
		errors_host[i] = 0;
	}
	unsigned int * errors_device;
	cudaStatus = cudaMalloc((void**)&errors_device, errorsMatrixSize * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed 300!");
	}
	error = cudaMemcpy(errors_device, errors_host, errorsMatrixSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
	double a, b;

	float blockDim_y = (tmp_height > 32) ? 32 : tmp_height;
	a = ceil(tmp_width / 32.0);
	b = ceil(tmp_height / blockDim_y);
	//printf("\n\n%f %f %d %d \n\n", a, b, 32, blockDim_y);

	cudaEvent_t start;
	error = cudaEventCreate(&start);
	if (error != cudaSuccess){
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	cudaEvent_t stop;
	error = cudaEventCreate(&stop);
	if (error != cudaSuccess){
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	// Record the start event
	error = cudaEventRecord(start, NULL);
	if (error != cudaSuccess){
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	dim3 gridDim(ceil(tmp_width/32), ceil(tmp_height/blockDim_y), 1);
	dim3 blockDim(32, blockDim_y, 1);

	computeError << <gridDim, blockDim >> > (img, tmp, img_height, img_width, tmp_height, tmp_width, errors_device);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch 111 failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	
	error = cudaMemcpy(errors_host, errors_device, errorsMatrixSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	int maximumPosibleNumberOfMatches = (img_height / tmp_height + 1)*(img_width / tmp_width + 1);
	int * pos_x = (int *)malloc(maximumPosibleNumberOfMatches * sizeof(int));
	int * pos_y = (int *)malloc(maximumPosibleNumberOfMatches * sizeof(int));
	unsigned int * errorOfPos = (unsigned int *)malloc(maximumPosibleNumberOfMatches * sizeof(unsigned int));
	errorOfPos[0] = 99999;
	int pointer = 0;
	int curVal;

	for (int y = 0; y < img_height - tmp_height + 1; ++y) {
		for (int x = 0; x < img_width - tmp_width + 1; ++x) {
			curVal = errors_host[y*(img_width - tmp_width + 1) + x];
			if (curVal < errorOfPos[0] && curVal < 60) {
				pointer = 0;
				pos_x[pointer] = x;
				pos_y[pointer] = y;
				errorOfPos[pointer] = curVal;
				pointer++;
			}
			else if (curVal == errorOfPos[0]) {
				pos_x[pointer] = x;
				pos_y[pointer] = y;
				errorOfPos[pointer] = curVal;
				pointer++;
				if (pointer >= maximumPosibleNumberOfMatches)
					printf("pointer exeeded limit: %d", pointer--);
			}
		}
	}

	// Record the stop event
	error = cudaEventRecord(stop, NULL);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);
	if (error != cudaSuccess) {
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	//printf("\n\nElapsed time in msec = %f\n", msecTotal);
	
	//printf("\n\n Matched positions: %d\n\n", pointer);
	//for (size_t i = 0; i < pointer; i++)
		//printf("#%d x: %d y: %d error: %d\n", (i+1), pos_x[i], pos_y[i], errorOfPos[i]);
	
	free(errors_host);
	cudaFree(errors_device);
	return pointer;
}

int main(int argc, char *argv[]) {

	for (int i = 0; i<(argc - 1)/2.0; i++){

		//printf("\n\n--------- Pair #%d ---------\n\n", i);

		//Load image
		char str_main[MAX_PATH];
		char str_template[MAX_PATH];

		GetFullPathName(_T(argv[2 * i + 1]), MAX_PATH, str_main, NULL);
		GetFullPathName(_T(argv[2 * i + 2]), MAX_PATH, str_template, NULL);
		
		/*strcpy(str_main, "C:/Users/15/Desktop/");
		strcpy(str_template, "C:/Users/15/Desktop/");
		strcat(str_main, argv[2 * i + 1]);
		strcat(str_template, argv[2 * i + 2]);*/

		bitmap_image image(str_main);
		if (!image) {
			printf("\n\nError - Failed to open main image: %s",str_main);
			return 1;
		}
		const unsigned int height = image.height();
		const unsigned int width = image.width();

		bitmap_image temp_image(str_template);
		if (!temp_image) {
			printf("\nError - Failed to open temp image: %s\n", str_template);
			return 1;
		}
		int temp_height = temp_image.height();
		int temp_width = temp_image.width();

		//printf("height %d, width %d\n tempHeight %d, tempWidtch %d\n\n", height, width, temp_height, temp_width);
		
		unsigned char * imageArr = (unsigned char *)malloc(height * width * 3 * sizeof(unsigned char));
		unsigned char * temp_imageArr = (unsigned char *)malloc(temp_height * temp_width * 3 * sizeof(unsigned char));
		unsigned char * temp_imageArr_rotated = (unsigned char *)malloc(temp_height * temp_width * 3 * sizeof(unsigned char));
		
		for (std::size_t y = 0; y < height; ++y) {
			for (std::size_t x = 0; x < width; ++x) {
				rgb_t colour;
				image.get_pixel(x, y, colour);

				imageArr[(y*width + x) * 3] = colour.red;
				imageArr[(y*width + x) * 3 + 1] = colour.green;
				imageArr[(y*width + x) * 3 + 2] = colour.blue;
			}
		}
		
		for (std::size_t y = 0; y < temp_height; ++y) {
			for (std::size_t x = 0; x < temp_width; ++x) {
				
				rgb_t colour;
				temp_image.get_pixel(x, y, colour);

				temp_imageArr[(y*temp_width + x) * 3] = colour.red;
				temp_imageArr[(y*temp_width + x) * 3 + 1] = colour.green;
				temp_imageArr[(y*temp_width + x) * 3 + 2] = colour.blue;
				
				temp_imageArr_rotated[(x*temp_height + temp_height - y - 1) * 3] = colour.red;
				temp_imageArr_rotated[(x*temp_height + temp_height - y - 1) * 3 + 1] = colour.green;
				temp_imageArr_rotated[(x*temp_height + temp_height - y - 1) * 3 + 2] = colour.blue;
			}
		}
		
		cudaError_t cudaStatus;
		cudaError_t error;
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		}

		unsigned char * imageArr_dev;
		unsigned char * temp_imageArr_dev;
		unsigned char * temp_imageArr_dev_rotated;
		cudaStatus = cudaMalloc((void**)&imageArr_dev, height * width * 3 * sizeof(unsigned char));
		if (cudaStatus != cudaSuccess) {
			printf("cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)&temp_imageArr_dev, temp_height * temp_width * 3 * sizeof(unsigned char));
		if (cudaStatus != cudaSuccess) {
			printf("cudaMalloc failed!");
		}
		cudaStatus = cudaMalloc((void**)&temp_imageArr_dev_rotated, temp_height * temp_width * 3 * sizeof(unsigned char));
		if (cudaStatus != cudaSuccess) {
			printf("cudaMalloc failed!");
		}

		error = cudaMemcpy(imageArr_dev, imageArr, height * width * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
		error = cudaMemcpy(temp_imageArr_dev, temp_imageArr, temp_height * temp_width * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
		error = cudaMemcpy(temp_imageArr_dev_rotated, temp_imageArr_rotated, temp_height * temp_width * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
		
		int numberOfMatches = 0;
		//printf("--- >normal template:");
		numberOfMatches += findMatches(imageArr_dev, temp_imageArr_dev, height, width, temp_height, temp_width, imageArr);
		//printf("--- >rotated template:");
		numberOfMatches += findMatches(imageArr_dev, temp_imageArr_dev_rotated, height, width, temp_width, temp_height, imageArr);
		printf("%d\n", numberOfMatches);

		cudaFree(imageArr_dev);
		cudaFree(temp_imageArr_dev);
		cudaFree(temp_imageArr_dev_rotated);
		free(imageArr);
		free(temp_imageArr);
		free(temp_imageArr_rotated);
		//findeMatches_serial(imageArr, temp_imageArr, temp_imageArr_rotated, height, width, temp_height, temp_width, temp_width, temp_height);
	}

	return EXIT_SUCCESS;
}