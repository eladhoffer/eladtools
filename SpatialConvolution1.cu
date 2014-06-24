extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include <stdio.h>
#include <assert.h>
#include "cublas_v2.h"

#define TB 128

/* operations */
struct opPlus {
	static const float base_value = 0.0;
	__device__ float operator()(float x, float y)
	{
		return x + y;
	}
};

struct opMinus {
	static const float base_value = 0.0;
	__device__ float operator()(float x, float y)
	{
		return x - y;
	}
};

struct opMult {
	static const float base_value = 1.0;
	__device__ float operator()(float x, float y)
	{
		return x * y;
	}
};

struct opDiv {
	static const float base_value = 1.0;
	__device__ float operator()(float x, float y)
	{
		return x / y;
	}
};

struct opMask {
	__device__ float operator()(float x, float y)
	{
		return y == 0 ? 0 : x;
	}
};

struct opSMul {
	float alpha;
	opSMul(float alpha_) : alpha(alpha_) {};
	__device__ float operator()(float x)
	{
		return alpha * x;
	}
};

struct opMax {
	static const float base_value = -2e38;
	__device__ float operator()(float x, float y)
	{
		return fmaxf(x, y);
	}
};

struct opClip {
	float low, high;
	opClip(float low_, float high_) : low(low_), high(high_) {};
	__device__ float operator()(float x)
	{
		return min(high, max(low, x));
	}
};

struct opExp {
	__device__ float operator()(float x)
	{
		return exp(x);
	}
};

struct opSigmoid {
	__device__ float operator()(float x)
	{
		return 1 / (1 + exp(-x));
	}
};

struct opSigmoidDeriv {
	__device__ float operator()(float x, float y)
	{
		return x * y * (1 - y);
	}
};

struct opTanh {
	__device__ float operator()(float x)
	{
		return tanh(x);
	}
};

struct opTanhDeriv {
	__device__ float operator()(float x, float y)
	{
		return x * (1 - y * y);
	}
};

struct opRelu {
	__device__ float operator()(float x)
	{
		return max(x, 0.f);
	}
};

struct opReluDeriv {
	__device__ float operator()(float x, float y)
	{
		return y > 0 ? x : 0;
	}
};

struct opHuber {
	float threshold;
	opHuber(float threshold_) : threshold(threshold_) {};
	__device__ float operator()(float x, float y) {
		float d = x - y;
		if (-threshold < d && d < threshold) {
			return 0.5 * d * d;
		} else {
			return threshold * (abs(d) - 0.5 * threshold);
		}
	}
};

struct opHuberDeriv {
	float threshold;
	opHuberDeriv(float threshold_) : threshold(threshold_) {};
	__device__ float operator()(float x, float y) {
		float d = x - y;
		if (-threshold < d && d < threshold) {
			return d;
		} else {
			return threshold * signbit(d);
		}
	}
};

struct opShrink {
	float threshold;
	opShrink(float threshold_) : threshold(threshold_) {};
	
	__device__ float operator()(float x) { 
		if (x - threshold > 0) {
			return x - threshold;
		} else if (x + threshold < 0) {
			return x + threshold;
		} else {
			return 0.0;
		}
	}
};

struct distL1 {
	__device__ float forward(float x, float y) {
		return fabsf(x - y);
	}

	__device__ float backward(float x, float y) {
		if (x > y) {
			return 1;
		} else if (x < y) {
			return -1;
		} else {
			return 0;
		}
	}
};

struct distL2Square {
	__device__ float forward(float x, float y) {
		float d = x - y;
		return d * d;
	}

	__device__ float backward(float x, float y) {
		return 2 * (x - y);
	}
};

struct distCos {
	__device__ float forward(float x, float y) {
		return -x * y;
	}

	__device__ float backward(float x, float y) {
		return -y;
	}
};

/* Is A in row major format? */
int is_rm(THCudaTensor *A)
{
	for (int i = 0; i < 4; i++) {
		if (A->nDimension == i + 1) return 1;
		if (A->stride[i] < A->stride[i + 1]) return 0;
	}
	assert(0);
	return 0;
}

void checkCudaError(lua_State *L) {
	cudaError_t status = cudaPeekAtLastError();
	if (status != cudaSuccess) {
		luaL_error(L, cudaGetErrorString(status));
	}
}

/* res[i] = A[inds[i]] */
__global__ void get_cols(float *A, int A_stride, float *inds, float *res, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		res[i] = A[i * A_stride + (int)inds[i] - 1];
	}
}

int get_cols(lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *inds = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *res = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	int len = THCudaTensor_nElement(inds);
	get_cols<<<(len - 1)  / TB + 1, TB>>>(THCudaTensor_data(A), A->stride[0], THCudaTensor_data(inds), THCudaTensor_data(res), len);
	return 0;
}

__global__ void get_spatial_kernel(float *A, int A_stride, float *inds, float *res, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		int j = inds[i] - 1;
		res[i] = j == -1 ? 0 : A[j * A_stride + i];
	}
}

int get_spatial(lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *inds = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *res = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	assert(A->nDimension == 4);
	int len = THCudaTensor_nElement(inds);
	get_spatial_kernel<<<(len - 1)  / TB + 1, TB>>>(THCudaTensor_data(A), A->stride[1], THCudaTensor_data(inds), THCudaTensor_data(res), len);
	return 0;
}


/* A[inds[i]] = val */
__global__ void set_cols(float *A, int A_stride, float *inds, float val, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		A[i * A_stride + (int)inds[i] - 1] = val;
	}
}

int set_cols(lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *inds = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	float val = luaL_checknumber(L, 3);
	int len = THCudaTensor_nElement(inds);
	set_cols<<<(len - 1)  / TB + 1, TB>>>(THCudaTensor_data(A), A->stride[0], THCudaTensor_data(inds), val, len);
	return 0;
}

__global__ void set_spatial_kernel(float *A, int A_stride, float *inds, float val, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		int j = inds[i] - 1;
		if (j >= 0) {
			A[j * A_stride + i] = val;
		}
	}
}

int set_spatial(lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *inds = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	float val = luaL_checknumber(L, 3);
	int len = THCudaTensor_nElement(inds);
	set_spatial_kernel<<<(len - 1)  / TB + 1, TB>>>(THCudaTensor_data(A), A->stride[1], THCudaTensor_data(inds), val, len);
	return 0;
}

template<class Op>
int transform1(Op op, lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int lenA = THCudaTensor_nElement(A);
	int lenB = THCudaTensor_nElement(B);

	if (!is_rm(A) || !is_rm(B)) {
		luaL_error(L, "Matrices not in row major order");
	}

	if (lenA != lenB) {
		luaL_error(L, "Size mismatch");
	}

	thrust::device_ptr<float> pA(THCudaTensor_data(A));
	thrust::device_ptr<float> pB(THCudaTensor_data(B));
	thrust::transform(pA, pA + lenA, pB, op);
	return 0;
}

template<class Op>
int transform2(Op op, lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *C = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	int lenA = THCudaTensor_nElement(A);
	int lenB = THCudaTensor_nElement(B);
	int lenC = THCudaTensor_nElement(C);

	if (!is_rm(A) || !is_rm(B) || !is_rm(C)) {
		luaL_error(L, "Matrices not in roj major order");
	}

	if (lenA != lenB || lenA != lenC) {
		luaL_error(L, "Size mismatch");
	}

	thrust::device_ptr<float> pA(THCudaTensor_data(A));
	thrust::device_ptr<float> pB(THCudaTensor_data(B));
	thrust::device_ptr<float> pC(THCudaTensor_data(C));
	thrust::transform(pA, pA + lenA, pB, pC, op);
	return 0;
}

int huber(lua_State *L)
{
	float threshold = luaL_checknumber(L, 4);
	return transform2(opHuber(threshold), L);
}

int huber_deriv(lua_State *L)
{
	float threshold = luaL_checknumber(L, 4);
	return transform2(opHuberDeriv(threshold), L);
}

int mask(lua_State *L)
{
	return transform2(opMask(), L);
}

int shrink(lua_State *L)
{
	float threshold = luaL_checknumber(L, 3);
	return transform1(opShrink(threshold), L);
}

int sigmoid(lua_State *L)
{
	return transform1(opSigmoid(), L);
}

int mult_by_sigmoid_deriv(lua_State *L)
{
	return transform2(opSigmoidDeriv(), L);
}

int tanh(lua_State *L)
{
	return transform1(opTanh(), L);
}

int mult_by_tanh_deriv(lua_State *L)
{
	return transform2(opTanhDeriv(), L);
}

int relu(lua_State *L)
{
	return transform1(opRelu(), L);
}

int mult_by_relu_deriv(lua_State *L)
{
	return transform2(opReluDeriv(), L);
}

int clip(lua_State *L)
{
	float low = luaL_checknumber(L, 3);
	float high = luaL_checknumber(L, 4);
	return transform1(opClip(low, high), L);
}

int _exp(lua_State *L)
{
	return transform1(opExp(), L);
}

int smul(lua_State *L)
{
	float alpha = luaL_checknumber(L, 3);
	return transform1(opSMul(alpha), L);
}

/* What a crazy bug!
 *
 *
 *
 *
 *
 */
template <class Op, int axis>
__global__ void kMatVect(Op op, float *A, float *x, float *B, int len, int size0)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		if (axis == 0) B[i] = op(A[i], x[i % size0]);
		if (axis == 1) B[i] = op(A[i], x[i / size0]);
	}
}

template <class Op>
int mat_vect(Op op, lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *x = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	int axis = luaL_checkint(L, 4) - 1;

	assert(axis == 0 || axis == 1);

	if (!is_rm(A) || !is_rm(B)) {
		luaL_error(L, "Matrix not in row major order");
	}

	if (THCudaTensor_nElement(A) != THCudaTensor_nElement(B)) {
		luaL_error(L, "Size mismatch");
	}

	int len = THCudaTensor_nElement(A);
	if (axis == 0) {
		if (A->size[1] != THCudaTensor_nElement(x)) {
			luaL_error(L, "Size mismatch");
		}
		kMatVect<Op, 0><<<(len - 1) / TB + 1, TB>>>(op, THCudaTensor_data(A), THCudaTensor_data(x), THCudaTensor_data(B), len, A->size[1]);
	} else if (axis == 1) {
		if (A->size[0] != THCudaTensor_nElement(x)) {
			luaL_error(L, "Size mismatch");
		}
		kMatVect<Op, 1><<<(len - 1) / TB + 1, TB>>>(op, THCudaTensor_data(A), THCudaTensor_data(x), THCudaTensor_data(B), len, A->size[1]);
	}

	checkCudaError(L);
	return 0;
}

int add_mat_vect(lua_State *L)
{
	return mat_vect(opPlus(), L);
}

int sub_mat_vect(lua_State *L)
{
	return mat_vect(opMinus(), L);
}

int mult_mat_vect(lua_State *L)
{
	return mat_vect(opMult(), L);
}

int div_mat_vect(lua_State *L)
{
	return mat_vect(opDiv(), L);
}

__global__ void kAdd(float *A, float *B, float *C, float alpha, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) C[i] = A[i] + alpha * B[i];
}

/* C = A + alpha * B */
int add(lua_State *L)
{
	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *C = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	float alpha = luaL_optnumber(L, 4, 1.0);

	if (!is_rm(A) || !is_rm(B) || !is_rm(C)) {
		luaL_error(L, "Matrices not in row major order");
	}

	if (!(A->size[0] == B->size[0] && A->size[1] == B->size[1] && A->size[0] == C->size[0] && A->size[1] == C->size[1])) {
		luaL_error(L, "Size mismatch");
	}

	int len = THCudaTensor_nElement(A);
	kAdd<<<(len - 1) / TB + 1, TB>>>(THCudaTensor_data(A), THCudaTensor_data(B), THCudaTensor_data(C), alpha, len);
	checkCudaError(L);
	return 0;
}

/* What a crazy bug!
 *
 *
 *
 *
 *
 */
template <class Op>
__global__ void kReduce(Op op, float *A, float *x, int n, int axis)
{
	extern __shared__ float sdata[];

	int i = threadIdx.x;

	sdata[i] = op.base_value;
	if (i < n) {
		if (axis == 0) {
			sdata[i] = A[gridDim.x * threadIdx.x + blockIdx.x];
		} else if (axis == 1) {
			sdata[i] = A[threadIdx.x + n * blockIdx.x];
		}
	}
	__syncthreads();

	for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (i < s) {
			sdata[i] = op(sdata[i], sdata[i + s]);
		}
		__syncthreads();
	}

	if (i == 0) {
		x[blockIdx.x] = sdata[0];
	}
}

template <class Op>
int reduce(Op op, lua_State *L)
{
	int reduce_dim, other_dim;

	THCudaTensor *A = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *x = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int axis = luaL_checkint(L, 3) - 1;

	if (!is_rm(A)) {
		luaL_error(L, "Matrix not in row major order");
	}

	if (axis != 0 && axis != 1) {
		luaL_error(L, "axis not in {0, 1}");
	}

	if (axis == 0) {
		reduce_dim = A->size[0];
		other_dim = A->size[1];
	} else if (axis == 1) {
		reduce_dim = A->size[1];
		other_dim = A->size[0];
	}

	assert(reduce_dim <= 1024);
	if (other_dim != THCudaTensor_nElement(x)) {
		luaL_error(L, "Size mismatch"); 
	}

	int threads = 1;
	while(threads < reduce_dim) {
		threads = threads << 1;
	}

	kReduce<Op><<<other_dim, threads, threads * sizeof(float)>>>(op, THCudaTensor_data(A), THCudaTensor_data(x), reduce_dim, axis);
	checkCudaError(L);
	return 0;
}

int sum(lua_State *L)
{
	return reduce(opPlus(), L);
}

int _max(lua_State *L)
{
	return reduce(opMax(), L);
}

__global__ void kShrink2(float *x1, float *x2, float l, float g, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
		float u, v;
		// This must be painfully slow because of branching
		u = -l -g +x1[i]; v = -l +g +x2[i]; if (u > v && v > 0) goto end;
		u = -l -g +x1[i]; v = +l +g +x2[i]; if (u > 0 && 0 > v) goto end;
		u = -l +g +x1[i]; v = -l -g +x2[i]; if (v > u && u > 0) goto end;
		u = +l +g +x1[i]; v = -l -g +x2[i]; if (v > 0 && 0 > u) goto end;
		u = +l -g +x1[i]; v = +l +g +x2[i]; if (0 > u && u > v) goto end;
		u = +l +g +x1[i]; v = +l -g +x2[i]; if (0 > v && v > u) goto end;
end:
		x1[i] = u;
		x2[i] = v;
    }
}

int shrink2(lua_State *L)
{
	THCudaTensor *x1 = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *x2 = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	float lambda = luaL_checknumber(L, 3);
	float gamma = luaL_checknumber(L, 4);

	int x1_size = THCudaTensor_nElement(x1);
	int x2_size = THCudaTensor_nElement(x2);
	
	if (!is_rm(x1) && !is_rm(x2)) {
		luaL_error(L, "Matrix not in row major order");
	}

	if (x1_size != x2_size) {
		luaL_error(L, "Size mismatch");
	}

	kShrink2<<<(x1_size - 1)  / TB + 1, TB>>>(THCudaTensor_data(x1), THCudaTensor_data(x2), lambda, gamma, x1_size);
	checkCudaError(L);
	return 0;
}

__global__ void spatial_argmax_kernel(float *input, float *output, int size, int size1, int size23)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dim23 = id % size23;
		int dim0 = id / size23;

		int argmax = 0;
		float max = -2e38;
		for (int i = 0; i < size1; i++) {
			float val = input[(dim0 * size1 + i) * size23 + dim23];
			if (val > max) {
				max = val;
				argmax = i;
			}
		}
		output[id] = argmax + 1;
	}
}

int spatial_argmax(lua_State *L)
{
	THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

	if (!is_rm(input) && !is_rm(output)) {
		luaL_error(L, "Matrix not in row major order");
	}

	if (input->nDimension != 4 || output->nDimension != 4) {
		luaL_error(L, "Number of dimensions has to be 4");
	}

	if (THCudaTensor_size(input, 0) != THCudaTensor_size(output, 0) ||
	  THCudaTensor_size(output, 1) != 1 ||
	  THCudaTensor_size(input, 2) != THCudaTensor_size(output, 2) ||
	  THCudaTensor_size(input, 3) != THCudaTensor_size(output, 3)) {
		luaL_error(L, "Size mismatch");
	}

	int size = THCudaTensor_nElement(output);
	spatial_argmax_kernel<<<(size - 1) / TB + 1, TB>>>(
		THCudaTensor_data(input), 
		THCudaTensor_data(output), 
		size,
		THCudaTensor_size(input, 1),
		THCudaTensor_size(input, 2) * THCudaTensor_size(output, 3));
	checkCudaError(L);
	return 0;
}

__global__ void spatial_argmin_kernel(float *input, float *output, int size, int size1, int size23)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dim23 = id % size23;
		int dim0 = id / size23;

		int argmin = 0;
		float min = 2e38;
		for (int i = 0; i < size1; i++) {
			float val = input[(dim0 * size1 + i) * size23 + dim23];
			if (val < min) {
				min = val;
				argmin = i;
			}
		}
		output[id] = argmin + 1;
	}
}

int spatial_argmin(lua_State *L)
{
	THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

	if (!is_rm(input) && !is_rm(output)) {
		luaL_error(L, "Matrix not in row major order");
	}

	if (input->nDimension != 4 || output->nDimension != 4) {
		luaL_error(L, "Number of dimensions has to be 4");
	}

	if (THCudaTensor_size(input, 0) != THCudaTensor_size(output, 0) ||
	  THCudaTensor_size(output, 1) != 1 ||
	  THCudaTensor_size(input, 2) != THCudaTensor_size(output, 2) ||
	  THCudaTensor_size(input, 3) != THCudaTensor_size(output, 3)) {
		luaL_error(L, "Size mismatch");
	}

	int size = THCudaTensor_nElement(output);
	spatial_argmin_kernel<<<(size - 1) / TB + 1, TB>>>(
		THCudaTensor_data(input), 
		THCudaTensor_data(output), 
		size,
		THCudaTensor_size(input, 1),
		THCudaTensor_size(input, 2) * THCudaTensor_size(output, 3));
	checkCudaError(L);
	return 0;
}

__global__ void cunn_SpatialConvolution1_updateOutput_kernel(float *input, float *weight, int transpose_weight, float *output, int img_size, int num_input, int num_output)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.y;
	float input_reg[32];

	__shared__ float weight_s[32 * 32];
	for (int i = threadIdx.x; i < num_input * num_output; i += blockDim.x) {
		if (transpose_weight) {
			weight_s[(i % num_output) * num_input + (i / num_output)] = weight[i];
		} else {
			weight_s[i] = weight[i];
		}
	}
	__syncthreads();

	if (id < img_size) { 
		for (int j = 0; j < num_input; j++) {
			input_reg[j] = input[(batch * num_input + j) * img_size + id];
		}

		for (int i = 0; i < num_output; i++) {
			float s = 0;
			for (int j = 0; j < num_input; j++) {
				s += input_reg[j] * weight_s[i * num_input + j];
			}
			output[(batch * num_output + i) * img_size + id] = s;
		}
	}
}

int cunn_SpatialConvolution1_updateOutput(lua_State *L)
{
	THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *weight = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int transpose_weight = luaL_checkinteger(L, 3);
	THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");

	int batch_size = THCudaTensor_size(input, 0);
	int img_size = THCudaTensor_size(input, 2) * THCudaTensor_size(input, 3);

	int num_input, num_output;
	if (transpose_weight) {
		num_input = THCudaTensor_size(weight, 0);
		num_output = THCudaTensor_size(weight, 1);
	} else {
		num_input = THCudaTensor_size(weight, 1);
		num_output = THCudaTensor_size(weight, 0);
	}

	if (!is_rm(input) || !is_rm(weight) || !is_rm(output)) {
		luaL_error(L, "Matrix not in row major order");
	}

	assert(num_input <= 32 && num_input * num_output <= 32 * 32);

	dim3 grid((img_size - 1) / TB + 1, batch_size);
	cunn_SpatialConvolution1_updateOutput_kernel<<<grid, TB>>>(
		THCudaTensor_data(input), 
		THCudaTensor_data(weight), 
		transpose_weight,
		THCudaTensor_data(output), 
		img_size, num_input, num_output);

	checkCudaError(L);
	return 0;
}

__global__ void cunn_SpatialConvolution1_accGradParameters_kernel(float *input, float *grad_output, float *grad, int batch_size, int img_size, int num_input, int num_output)
{
	__shared__ float input_s[32 * 32];
	__shared__ float grad_output_s[32 * 32];

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < num_input * batch_size; i += blockDim.x * blockDim.y) {
		input_s[i] = input[i * img_size + blockIdx.x];
	}

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < num_output * batch_size; i += blockDim.x * blockDim.y) {
		grad_output_s[i] = grad_output[i * img_size + blockIdx.x];
	}

	__syncthreads();
	
	float s = 0;
	for (int k = 0; k < batch_size; k++) {
		s += grad_output_s[k * num_output + threadIdx.x] * input_s[k * num_input + threadIdx.y];
	}
	
	atomicAdd(grad + threadIdx.x * num_input + threadIdx.y, s);
}

int cunn_SpatialConvolution1_accGradParameters(lua_State *L)
{
	THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *grad_output = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *grad = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

	int batch_size = THCudaTensor_size(input, 0);
	int img_size = THCudaTensor_size(input, 2) * THCudaTensor_size(input, 3);
	int num_input = THCudaTensor_size(input, 1);
	int num_output = THCudaTensor_size(grad_output, 1);

	if (!is_rm(input) || !is_rm(grad_output) || !is_rm(grad)) {
		luaL_error(L, "Matrix not in row major order");
	}

	assert(num_input <= 32 && batch_size <= 32 && num_input * num_output <= 32 * 32);
	dim3 block(num_output, num_input);
	cunn_SpatialConvolution1_accGradParameters_kernel<<<img_size, block>>>(THCudaTensor_data(input), THCudaTensor_data(grad_output), THCudaTensor_data(grad), batch_size, img_size, num_input, num_output);

	checkCudaError(L);
	return 0;
}

__global__ void add_bias4_kernel(float *input, float *bias, int input_size, int bias_size, int img_size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < input_size) {
		input[id] += bias[(id / img_size) % bias_size];
	}
}

int add_bias4(lua_State *L)
{
	THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *bias = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

	assert(THCudaTensor_size(input, 1) == THCudaTensor_nElement(bias));
	assert(THCudaTensor_size(input, 1) <= 32);

	if (!is_rm(input) || !is_rm(bias)) {
		luaL_error(L, "Matrix not in row major order");
	}

	add_bias4_kernel<<<(THCudaTensor_nElement(input) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(input), 
		THCudaTensor_data(bias), 
		THCudaTensor_nElement(input),
		THCudaTensor_size(input, 1),
		THCudaTensor_size(input, 2) * THCudaTensor_size(input, 3));
	checkCudaError(L);
	return 0;
}


static const struct luaL_Reg cunn_SpatialConvolution1__ [] = {
  {"cunn_SpatialConvolution1_updateOutput", cunn_SpatialConvolution1_updateOutput},
  {NULL, NULL}
};

void cunn_SpatialConvolution1_init(lua_State *L)
{
luaL_openlib(L, "nn", cunn_SpatialConvolution1__, 0);
}
