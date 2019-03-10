#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include "device_atomic_functions.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>


using namespace std;


//Service functions
__global__ void evaluate_weights();
int absolute(int x);
int** split_coordinate(unsigned int* vec);


//Field operations
__device__ unsigned int rem(unsigned int x);
__device__ unsigned int mult(unsigned int x, unsigned int y);
__device__ unsigned int pow(unsigned int x, unsigned int y);


__device__ __managed__ unsigned int m = 16;
__device__ __managed__ unsigned int generator_polynomial = 0b10001000000001011;
__managed__ unsigned int weights[1 << 16] = { 0 };

//Kernals
__global__ void evaluate_function(unsigned int* res, unsigned int deg);
__global__ void fft_device(const int* vec, int* res, int var_index);
__global__ void fft_device(const long long* vec, long long* res, int var_index);
__global__ void fmt_device(const int* vec, int* res, int var_index);
__global__ void sqr_device(const int* vec, long long* res);
__global__ void div_device(const int* vec, float* res, int d);
__global__ void differencial_probability(const int* vec, int* res, int a);
__global__ void plus_minus_analog(const int* vec, int* res);


//Fast transforms
void fft(int** vec, int** res);
void fwt(int** vec, int** res);
void fmt(int** vec, int** res);
void fft(int* vec, int* res);
void fmt(int* vec, int* res);
void fft_with_normalization(int** vec, float** res);
void fft_with_normalization(int* vec, float* res);


//Cryptographic properies
int disbalance(const int* W);
int degree(const int* A);
int nonlinearity(const int* W);
int correlation_immunity(const int* W);
int coefficient_of_error_propagation(const int* T, int var_index);
int coefficient_of_error_propagation_in_mean(const int* T, int var_index);
int sac(const int* W);
bool zero_level_sac(int** T);
bool zero_level_sac(int* T);
bool mean_sac(const int* T);
float mdp(const int* T);

bool is_bent(int** W);


int main(int argc, char* argv[]) {
	//Fields pre-computing
	evaluate_weights <<< (1 << 16), 1 >>> ();
	cudaDeviceSynchronize();

	cout << "Enter degree: ";
	unsigned int N;
	cin >> N;

	unsigned int* T = new unsigned int[1 << 16];

	unsigned int* device_T;
	cudaMalloc(&device_T, (1 << 16) * sizeof(unsigned int));
	
	evaluate_function <<< (1 << 16), 1 >>> (device_T, N);
	cudaDeviceSynchronize();

	cudaMemcpy(T, device_T, (1 << 16) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	int** coordinate_T = split_coordinate(T);

	int** W = new int*[16];
	for (int i = 0; i < 16; i++)
		W[i] = new int[1 << 16];

	fwt(coordinate_T, W);

	int** A = new int*[16];
	for (int i = 0; i < 16; i++)
		A[i] = new int[1 << 16];

	fmt(coordinate_T, A);

	cout << "Disbalance of coordinate functions:" << endl;
	for (int i = 0; i < 16; i++)
		cout << "[" << i << "]: " << disbalance(W[i]) << endl;

	cout << "Algebraic degree of coordinate functions:" << endl;
	for (int i = 0; i < 16; i++)
		cout << "[" << i << "]: " << degree(A[i]) << endl;

	cout << "Nonlinearity of coordinate functions: (max_nonlinearity = " << ((1 << 15) - (1 << 7)) << ")" << endl;
	for (int i = 0; i < 16; i++)
		cout << "[" << i << "]: " << nonlinearity(W[i]) << endl;

	cout << "Correlation immunity of coordinate functions:" << endl;
	for (int i = 0; i < 16; i++)
		cout << "[" << i << "]: " << correlation_immunity(W[i]) << endl;

	cout << "Coefficient of error propagation for each function and varianble:" << endl;
	for (int i = 0; i < 16; i++) {
		for (int j = 15; j > -1; j--) {
			int k = coefficient_of_error_propagation(coordinate_T[i], j);
			float bias = ((float)(absolute(k - (1 << 15)))) / (1 << 15);
			cout << "[Function = " << i << "][Variable = " << 15 - j << "]: " << k << " (bias = " << bias << ")" << endl;
		}
	}

	cout << "Strict avalenche criteria for coordinate functions:" << endl;
	for (int i = 0; i < 16; i++)
		cout << "[" << i << "]: " << sac(W[i]) << endl;

	int* signed_T = new int[1 << 16];
	int* full_A = new int[1 << 16];
	for (int i = 0; i < (1 << 16); i++)
		signed_T[i] = (int)T[i];

	fmt(signed_T, full_A);

	cout << "Algebraic degree of (n, n) - function: " << degree(full_A) << endl;

	cout << "Coefficient of error propagation for (n, n) - function:" << endl;
	for (int i = 15; i > -1; i--) {
		int k = coefficient_of_error_propagation_in_mean(signed_T, i);
		float bias = ((float)(absolute(k - 16 * (1 << 15)))) / (16 * (1 << 15));
		cout << "[Variable = " << 15 - i << "]: " << k << " (bias = " << bias << ")" << endl;
	}

	string res;
	if (zero_level_sac(coordinate_T))
		res = "True";
	else
		res = "False";

	cout << "Strict SAC (zero level): " << res << endl;

	if (mean_sac(signed_T))
		res = "True";
	else
		res = "False";

	cout << "SAC in mean: " << res << endl;

	cout << "Processing MDP... (Wait a 1-2 minutes)" << endl;

	cout << "MDP: " << mdp(signed_T) << endl;

	for (int i = 0; i < 16; i++) {
		delete[] coordinate_T[i];
		delete[] W[i];
		delete[] A[i];
	}
	delete[] A;
	delete[] W;
	delete[] coordinate_T;
	delete[] T;
	delete[] signed_T;
	delete[] full_A;
	cudaFree(device_T);

	system("pause");

	return 0;
}


bool is_bent(int** W) {
	for (int i = 0; i < (1 << 16); i++) {
		for (int j = 0; j < 16; j++) {
			if (absolute(W[j][i]) != (1 << 8)) {
				return false;
			}
		}
	}
	return true;
}


__global__ void evaluate_weights() {
	weights[blockIdx.x] = __popc(blockIdx.x);
}


__device__ unsigned int rem(unsigned int x) {
	unsigned int res = x;
	for (int i = 31; i > 15; i--) {
		if (((res >> i) & 0x1) == 1)
			res = res ^ (generator_polynomial << (i - 16));
	}
	return res;
}


inline __device__ unsigned int mult(unsigned int x, unsigned int y) {
	unsigned int res = 0;
	for (int i = 0; i < 16; i++)
		if (((y >> i) & 0x1) == 1)
			res = res ^ (x << i); 
	return rem(res); 
}

inline __device__ unsigned int pow(unsigned int x, unsigned int y) {
	unsigned int b = 1;
	unsigned int c = x;
	for (int i = 0; i < 32; i++) {
		if (((y >> i) & 0x1) == 1)
			b = mult(b, c);
		c = mult(c, c);
	}
	return b;
}


__global__ void evaluate_function(unsigned int* res, unsigned int deg) {
	res[blockIdx.x] = pow(blockIdx.x, deg);
}


//Without division (normalization)
__global__ void fft_device(const int* vec, int* res, int var_index) {
	if (((blockIdx.x >> var_index) & 0x1) == 0)
		res[blockIdx.x] = vec[blockIdx.x] + vec[blockIdx.x ^ (1 << var_index)];
	else
		res[blockIdx.x] = vec[blockIdx.x] - vec[blockIdx.x ^ (1 << var_index)];
}


//Without division (normalization)
__global__ void fft_device(const long long* vec, long long* res, int var_index) {
	if (((blockIdx.x >> var_index) & 0x1) == 0)
		res[blockIdx.x] = vec[blockIdx.x] + vec[blockIdx.x ^ (1 << var_index)];
	else
		res[blockIdx.x] = vec[blockIdx.x] - vec[blockIdx.x ^ (1 << var_index)];
}


__global__ void fmt_device(const int* vec, int* res, int var_index) {
	if (((blockIdx.x >> var_index) & 0x1) == 0)
		res[blockIdx.x] = vec[blockIdx.x];
	else
		res[blockIdx.x] = vec[blockIdx.x] ^ vec[blockIdx.x ^ (1 << var_index)];
}


__global__ void sqr_device(const int* vec, long long* res) {
	long long x = vec[blockIdx.x];
	res[blockIdx.x] = x * x;
}


__global__ void div_device(const int* vec, float* res, int d) {
	res[blockIdx.x] = ((float)(vec[blockIdx.x])) / d;
}


__global__ void differencial_probability(const int* vec, int* res, int a) {
	atomicAdd(&res[vec[blockIdx.x] ^ vec[blockIdx.x ^ a]], 1);
}


__global__ void plus_minus_analog(const int* vec, int* res) {
	if (vec[blockIdx.x] == 0)
		res[blockIdx.x] = 1;
	else
		res[blockIdx.x] = -1;
}


int disbalance(const int* W) {
	if (W[0] >= 0)
		return W[0];
	else
		return -W[0];
}


int degree(const int* A) {
	unsigned int deg = 0;
	for (int i = 1; i < (1 << 16); i++)
		if ((A[i] != 0) && (deg < weights[i]))
			deg = weights[i];
	return (int)deg;
}


int nonlinearity(const int* W) {
	int max_w = 0;
	for (int i = 0; i < (1 << 16); i++) {
		int abs_w = absolute(W[i]);
		if (max_w < abs_w)
			max_w = abs_w;
	}
	return (int)((1 << 15) - max_w / 2);
}


int correlation_immunity(const int* W) {
	int cor = 0;
	bool flag = true;
	for (int w = 1; w <= 16; w++) {
		for (int i = 1; i < (1 << 16); i++) {
			if ((weights[i] == w) && (W[i] != 0))
				flag = false;
		}
		if (flag)
			cor++;
		else
			break;
	}
	return cor;
}


int coefficient_of_error_propagation(const int* T, int var_index) {
	int k = 0;
	for (int i = 0; i < (1 << 16); i++)
		k += (T[i] ^ T[i ^ (1 << var_index)]);
	return k;
}


int coefficient_of_error_propagation_in_mean(const int* T, int var_index) {
	int k = 0;
	for (int i = 0; i < (1 << 16); i++)
		k += weights[(T[i] ^ T[i ^ (1 << var_index)])];
	return k;
}


int sac(const int* W) {
	long long* auto_cor_func = new long long[1 << 16];
	long long* temp_1;
	long long* temp_2;
	int* temp_3;

	cudaMalloc(&temp_1, (1 << 16) * sizeof(long long));
	cudaMalloc(&temp_2, (1 << 16) * sizeof(long long));
	cudaMalloc(&temp_3, (1 << 16) * sizeof(int));

	cudaMemcpy(temp_3, W, (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);

	sqr_device <<< (1 << 16), 1 >>> (temp_3, temp_1);
	cudaDeviceSynchronize();

	for (int j = 0; j < 16; j++) {
		fft_device <<< (1 << 16), 1 >>> (temp_1, temp_2, j);

		cudaDeviceSynchronize();
		cudaMemcpy(temp_1, temp_2, (1 << 16) * sizeof(long long), cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(auto_cor_func, temp_1, (1 << 16) * sizeof(long long), cudaMemcpyDeviceToHost);

	cudaFree(temp_1);
	cudaFree(temp_2);
	cudaFree(temp_3);

	int s = 0;
	bool flag = true;
	for (int w = 1; w <= 16; w++) {
		for (int i = 1; i < (1 << 16); i++) {
			if ((weights[i] == w) && (auto_cor_func[i] != 0))
				flag = false;
		}
		if (flag)
			s++;
		else
			break;
	}

	delete[] auto_cor_func;
	return s;
}


bool zero_level_sac(int** T) {
	for (int i = 0; i < 16; i++)
		if (!zero_level_sac(T[i]))
			return false;
	return true;
}


bool zero_level_sac(int* T) {
	for (int i = 0; i < 16; i++)
		if (coefficient_of_error_propagation(T, i) != (1 << 15))
			return false;
	return true;
}


bool mean_sac(const int* T) {
	bool flag = true;

	for (int i = 0; i < 16; i++) {
		int k = coefficient_of_error_propagation_in_mean(T, i);
		if (k != 16 * (1 << 15)) {
			flag = false;
			break;
		}
	}

	return flag;
}


float mdp(const int* T) {
	int max = 0;
	int* host_d = new int[1 << 16];
	int* d;
	int* temp;

	cudaMalloc(&temp, (1 << 16) * sizeof(int));
	cudaMalloc(&d, (1 << 16) * sizeof(int));
	cudaMemcpy(temp, T, (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);

	for (int i = 0; i < (1 << 16); i++)
		host_d[i] = 0;

	for (int a = 1; a < (1 << 16); a++) {
		cudaMemcpy(d, host_d, (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);

		differencial_probability <<< (1 << 16), 1 >>> (temp, d, a);

		cudaDeviceSynchronize();
		cudaMemcpy(host_d, d, (1 << 16) * sizeof(int), cudaMemcpyDeviceToHost);

		for (int i = 0; i < (1 << 16); i++) {
			if (max < host_d[i]) {
				max = host_d[i];
			}
			host_d[i] = 0;
		}
	}

	delete[] host_d;
	cudaFree(d);
	cudaFree(temp);

	return ((float)max) / (1 << 16);
}


//Without division (normalization)
void fft(int** vec, int** res) {
	int* temp_1;
	int* temp_2;

	cudaMalloc(&temp_1, (1 << 16) * sizeof(int));
	cudaMalloc(&temp_2, (1 << 16) * sizeof(int));

	for (int i = 0; i < 16; i++) {
		cudaMemcpy(temp_1, vec[i], (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);

		for (int j = 0; j < 16; j++) {
			fft_device <<< (1 << 16), 1 >>> (temp_1, temp_2, j);

			cudaDeviceSynchronize();
			cudaMemcpy(temp_1, temp_2, (1 << 16) * sizeof(int), cudaMemcpyDeviceToDevice);
		}

		cudaMemcpy(res[i], temp_1, (1 << 16) * sizeof(int), cudaMemcpyDeviceToHost);
	}

	cudaFree(temp_1);
	cudaFree(temp_2);
}


//Without division (normalization)
void fft(int* vec, int* res) {
	int* temp_1;
	int* temp_2;

	cudaMalloc(&temp_1, (1 << 16) * sizeof(int));
	cudaMalloc(&temp_2, (1 << 16) * sizeof(int));

	cudaMemcpy(temp_1, vec, (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);
	for (int j = 0; j < 16; j++) {
		fft_device <<< (1 << 16), 1 >>> (temp_1, temp_2, j);

		cudaDeviceSynchronize();
		cudaMemcpy(temp_1, temp_2, (1 << 16) * sizeof(int), cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(res, temp_1, (1 << 16) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(temp_1);
	cudaFree(temp_2);
}


void fwt(int** vec, int** res) {
	int* temp_1;
	int* temp_2;

	cudaMalloc(&temp_1, (1 << 16) * sizeof(int));
	cudaMalloc(&temp_2, (1 << 16) * sizeof(int));

	for (int i = 0; i < 16; i++) {
		cudaMemcpy(temp_2, vec[i], (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);
		plus_minus_analog <<< (1 << 16), 1 >>> (temp_2, temp_1);
		cudaDeviceSynchronize();

		for (int j = 0; j < 16; j++) {
			fft_device <<< (1 << 16), 1 >>> (temp_1, temp_2, j);

			cudaDeviceSynchronize();
			cudaMemcpy(temp_1, temp_2, (1 << 16) * sizeof(int), cudaMemcpyDeviceToDevice);
		}

		cudaMemcpy(res[i], temp_1, (1 << 16) * sizeof(int), cudaMemcpyDeviceToHost);
	}

	cudaFree(temp_1);
	cudaFree(temp_2);
}


void fwt(int* vec, int* res) {
	int* temp_1;
	int* temp_2;

	cudaMalloc(&temp_1, (1 << 16) * sizeof(int));
	cudaMalloc(&temp_2, (1 << 16) * sizeof(int));

	cudaMemcpy(temp_2, vec, (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);
	plus_minus_analog <<< (1 << 16), 1 >>> (temp_2, temp_1);
	cudaDeviceSynchronize();

	for (int j = 0; j < 16; j++) {
		fft_device <<< (1 << 16), 1 >>> (temp_1, temp_2, j);
		cudaDeviceSynchronize();
		cudaMemcpy(temp_1, temp_2, (1 << 16) * sizeof(int), cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(res, temp_1, (1 << 16) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(temp_1);
	cudaFree(temp_2);
}


void fmt(int** vec, int** res) {
	int* temp_1;
	int* temp_2;

	cudaMalloc(&temp_1, (1 << 16) * sizeof(int));
	cudaMalloc(&temp_2, (1 << 16) * sizeof(int));

	for (int i = 0; i < 16; i++) {
		cudaMemcpy(temp_1, vec[i], (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);
		for (int j = 0; j < 16; j++) {
			fmt_device <<< (1 << 16), 1 >>> (temp_1, temp_2, j);

			cudaDeviceSynchronize();
			cudaMemcpy(temp_1, temp_2, (1 << 16) * sizeof(int), cudaMemcpyDeviceToDevice);
		}
		cudaMemcpy(res[i], temp_1, (1 << 16) * sizeof(int), cudaMemcpyDeviceToHost);
	}

	cudaFree(temp_1);
	cudaFree(temp_2);
}


void fmt(int* vec, int* res) {
	int* temp_1;
	int* temp_2;

	cudaMalloc(&temp_1, (1 << 16) * sizeof(int));
	cudaMalloc(&temp_2, (1 << 16) * sizeof(int));

	cudaMemcpy(temp_1, vec, (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);

	for (int j = 0; j < 16; j++) {
		fmt_device <<< (1 << 16), 1 >>> (temp_1, temp_2, j);

		cudaDeviceSynchronize();
		cudaMemcpy(temp_1, temp_2, (1 << 16) * sizeof(int), cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(res, temp_1, (1 << 16) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(temp_1);
	cudaFree(temp_2);
}


void fft_with_normalization(int** vec, float** res) {
	int* temp_1;
	int* temp_2;
	float* temp_3;

	cudaMalloc(&temp_1, (1 << 16) * sizeof(int));
	cudaMalloc(&temp_2, (1 << 16) * sizeof(int));
	cudaMalloc(&temp_3, (1 << 16) * sizeof(float));

	for (int i = 0; i < 16; i++) {
		cudaMemcpy(temp_1, vec[i], (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);
		for (int j = 0; j < 16; j++) {
			fft_device <<< (1 << 16), 1 >>> (temp_1, temp_2, j);

			cudaDeviceSynchronize();
			cudaMemcpy(temp_1, temp_2, (1 << 16) * sizeof(int), cudaMemcpyDeviceToDevice);
		}
		div_device <<< (1 << 16), 1 >>> (temp_1, temp_3, (1 << 16));

		cudaDeviceSynchronize();
		cudaMemcpy(res[i], temp_3, (1 << 16) * sizeof(float), cudaMemcpyDeviceToHost);
	}

	cudaFree(temp_1);
	cudaFree(temp_2);
	cudaFree(temp_3);
}


void fft_with_normalization(int* vec, float* res) {
	int* temp_1;
	int* temp_2;

	cudaMalloc(&temp_1, (1 << 16) * sizeof(int));
	cudaMalloc(&temp_2, (1 << 16) * sizeof(int));

	cudaMemcpy(temp_1, vec, (1 << 16) * sizeof(int), cudaMemcpyHostToDevice);

	for (int j = 0; j < 16; j++) {
		fft_device <<< (1 << 16), 1 >>> (temp_1, temp_2, j);

		cudaDeviceSynchronize();
		cudaMemcpy(temp_1, temp_2, (1 << 16) * sizeof(int), cudaMemcpyDeviceToDevice);
	}

	float* temp_3;
	cudaMalloc(&temp_3, (1 << 16) * sizeof(float));

	div_device <<< (1 << 16), 1 >>> (temp_1, temp_3, (1 << 16));

	cudaDeviceSynchronize();
	cudaMemcpy(res, temp_3, (1 << 16) * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(temp_1);
	cudaFree(temp_2);
	cudaFree(temp_3);
}


int absolute(int x) {
	if (x >= 0) {
		return x;
	}
	else {
		return -x;
	}
}


int** split_coordinate(unsigned int* vec) {
	int** res = new int*[16];
	for (int i = 0; i < 16; i++)
		res[i] = new int[1 << 16];
	for (int i = 0; i < (1 << 16); i++) {
		for (int j = 0; j < 16; j++) {
			res[15 - j][i] = (int)((vec[i] >> j) & 0x1);
		}
	}
	return res;
}