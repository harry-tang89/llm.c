#include <float.h>


#include <iostream>
#include <functional>

#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "llmc/dataloader.h"
#include "llmc/_tokenizer.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void error_usage() {
    fprintf(stderr, "Usage:   ./train_gptfp32cu [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -o <string> output log file (default = NULL)\n");
    fprintf(stderr, "  -b <int>    batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    exit(EXIT_FAILURE);
}

typedef struct {
    FILE *logfile;
    int flush_every;
} Logger;
void logger_init(Logger *logger, const char *filename) {
    logger->flush_every = 20;
    logger->logfile = NULL;
    if (filename != NULL) { logger->logfile = fopenCheck(filename, "w"); }
}
void logger_log_val(Logger *logger, int step, float val_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d tel:%.4f\n", step, val_loss);
    }
}
void logger_log_train(Logger *logger, int step, float train_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d trl:%.4f\n", step, train_loss);
        if (step % 10 == 0) { fflush(logger->logfile); }
    }
}
void logger_free(Logger *logger) {
    if (logger->logfile != NULL) { fclose(logger->logfile); }
}

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;


typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
} GPTConfig;

typedef struct {
    float* bt4c; // (B, T, 4*C)
    float* preatt; // (B, NH, T, T)
    float* residual3; // (B, T, C)
} GradActTensors;

#define NUM_ACTIVATION_TENSORS 21
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* atty; // (L, B, T, C)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)

    float* losses;
    float* qkvr;
    float* output;
} ActivationPointerTensors;

#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float *weightsTokenEmbedding;
    float *positionEmbeddingWeights;
    float *layerNorm1Weights;
    float *layerNorm1Biases;
    float *attentionQKVWeights;
    float *attentionQKVBiases;
    float *attentionOutputWeights;
    float *attentionOutputBiases;
    float *layerNorm2Weights;
    float *ln2b;
    float *fcw;
    float *fcb;
    float *fcprojw;
    float *fcprojb;
    float *finalLayerNormWeights;
    float *finalLayerNormBiases;
} GPTParameterPointerTensors;


typedef struct {
    GPTConfig configuration;
    GPTParameterPointerTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;

    GPTParameterPointerTensors grads;
    float *gradients_memory;
    float *m_memory;
    float *v_memory;

    ActivationPointerTensors acts;
    size_t activation_sizes[NUM_ACTIVATION_TENSORS];
    float *activations_memory;
    size_t num_activations;
    GradActTensors grads_acts;
    size_t num_grad_acts;
    float *grads_acts_memory;

    int batch_size;
    int seq_len;
    int* inputs;
    int* targets;
    float mean_loss;
    float *cpu_losses;
} GPT;

// extern inline FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
//     FILE *fp = fopen(path, mode);
//     if (fp == NULL) {
//         fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
//         fprintf(stderr, "Error details:\n");
//         fprintf(stderr, "  File: %s\n", file);
//         fprintf(stderr, "  Line: %d\n", line);
//         fprintf(stderr, "  Path: %s\n", path);
//         fprintf(stderr, "  Mode: %s\n", mode);
//         fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
//         fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt.py`\n");
//         exit(EXIT_FAILURE);
//     }
//     return fp;
// }
// #define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

// extern inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
//     size_t result = fread(ptr, size, nmemb, stream);
//     if (result != nmemb) {
//         if (feof(stream)) {
//             fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
//         } else if (ferror(stream)) {
//             fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
//         } else {
//             fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
//                     file, line, nmemb, result);
//         }
//         fprintf(stderr, "Error details:\n");
//         fprintf(stderr, "  File: %s\n", file);
//         fprintf(stderr, "  Line: %d\n", line);
//         fprintf(stderr, "  Expected elements: %zu\n", nmemb);
//         fprintf(stderr, "  Read elements: %zu\n", result);
//         exit(EXIT_FAILURE);
//     }
// }
// #define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// extern inline void *malloc_check(size_t size, const char *file, int line) {
//     void *ptr = malloc(size);
//     if (ptr == NULL) {
//         fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
//         fprintf(stderr, "Error details:\n");
//         fprintf(stderr, "  File: %s\n", file);
//         fprintf(stderr, "  Line: %d\n", line);
//         fprintf(stderr, "  Size: %zu bytes\n", size);
//         exit(EXIT_FAILURE);
//     }
//     return ptr;
// }
// #define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

void fill_in_parameter_sizes(size_t *param_sizes, GPTConfig config) {
    int paddedVocabSize = config.padded_vocab_size;
    int C = config.channels;
    int maxSequenceLength = config.max_seq_len;
    int numLayers = config.num_layers;
    param_sizes[0] = paddedVocabSize * C;
    param_sizes[1] = maxSequenceLength * C;
    param_sizes[2] = numLayers * C;
    param_sizes[3] = numLayers * C;
    param_sizes[4] = numLayers * (3 * C) * C;
    param_sizes[5] = numLayers * (3 * C);
    param_sizes[6] = numLayers * C * C;
    param_sizes[7] = numLayers * C;
    param_sizes[8] = numLayers * C;
    param_sizes[9] = numLayers * C;
    param_sizes[10] = numLayers * (4 * C) * C;
    param_sizes[11] = numLayers * (4 * C);
    param_sizes[12] = numLayers * C * (4 * C);
    param_sizes[13] = numLayers * C;
    param_sizes[14] = C;
    param_sizes[15] = C;
}

float* malloc_and_point_parameters(GPTParameterPointerTensors* params, size_t* param_sizes, int on_device) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }

    float* parametersMemory;

    if (on_device) {
        cudaCheck(cudaMalloc((void**)&parametersMemory, num_parameters * sizeof(float)));
    } else {
        parametersMemory = (float*)mallocCheck(num_parameters * sizeof(float));
    }

    // Array of pointers to update the tensor pointers in 'tensors' struct to point to the correct locations within the allocated memory
    float** pointersToUpdate[] = {
        &params->weightsTokenEmbedding, &params->positionEmbeddingWeights, &params->layerNorm1Weights, &params->layerNorm1Biases, &params->attentionQKVWeights, &params->attentionQKVBiases,
        &params->attentionOutputWeights, &params->attentionOutputBiases, &params->layerNorm2Weights, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->finalLayerNormWeights, &params->finalLayerNormBiases
    };
    // Iterate through the memory and assign each tensor pointer a location in the allocated block
    float* currentMemoryPosition = parametersMemory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(pointersToUpdate[i]) = currentMemoryPosition;
        currentMemoryPosition += param_sizes[i];
    }
    return parametersMemory;
}

__global__ void initialize_weight(float* weights, float stddev, float mean, int N, curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {
        curand_init(seed, id, 0, &state[id]);
        float normal_val = curand_normal(&state[id]) * stddev + mean;
        // printf("normal_val: %f\n", normal_val);
        weights[id] = normal_val;
    }
}

void gpt_initialize(GPT* model) {
    int C = 2048;
    int maxSequenceLength = 2048;
    int numLayers = 6;
    int numHeads = 24;

    int vocabSize = 50257;
    int paddedVocabSize = 50304;

    model->configuration.max_seq_len = maxSequenceLength;
    model->configuration.vocab_size = paddedVocabSize;
    model->configuration.num_layers = numLayers;
    model->configuration.num_heads = numHeads;
    model->configuration.channels = C;
    model->configuration.padded_vocab_size = paddedVocabSize;

    model->param_sizes[0] = paddedVocabSize * C;
    model->param_sizes[1] = maxSequenceLength * C;
    model->param_sizes[2] = numLayers * C;
    model->param_sizes[3] = numLayers * C;
    model->param_sizes[4] = numLayers * (3 * C) * C;
    model->param_sizes[5] = numLayers * (3 * C);
    model->param_sizes[6] = numLayers * C * C;
    model->param_sizes[7] = numLayers * C;
    model->param_sizes[8] = numLayers * C;
    model->param_sizes[9] = numLayers * C;
    model->param_sizes[10] = numLayers * (4 * C) * C;
    model->param_sizes[11] = numLayers * (4 * C);
    model->param_sizes[12] = numLayers * C * (4 * C);
    model->param_sizes[13] = numLayers * C;
    model->param_sizes[14] = C;
    model->param_sizes[15] = C;
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;
    float* parametersMemoryPointer;
    cudaMalloc(&parametersMemoryPointer, num_parameters * sizeof(float));

    GPTParameterPointerTensors* params = &model->params;
    float** pointersToUpdate[] = {
        &params->weightsTokenEmbedding, &params->positionEmbeddingWeights, &params->layerNorm1Weights, &params->layerNorm1Biases, &params->attentionQKVWeights, &params->attentionQKVBiases,
        &params->attentionOutputWeights, &params->attentionOutputBiases, &params->layerNorm2Weights, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->finalLayerNormWeights, &params->finalLayerNormBiases
    };
    curandState *d_states;
    int thread_size = 1024;
    model->params_memory = parametersMemoryPointer;
    float* currentMemoryPosition = parametersMemoryPointer;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
         *(pointersToUpdate[i]) = currentMemoryPosition;
        cudaCheck(cudaMalloc(&d_states, model->param_sizes[i] * sizeof(curandState)));

        initialize_weight<<<CEIL_DIV(model->param_sizes[i], thread_size), thread_size>>>(currentMemoryPosition, 
        0.2f, 0.0f, model->param_sizes[i], d_states, time(NULL));
        cudaDeviceSynchronize();

        cudaCheck(cudaFree(d_states));
        currentMemoryPosition += model->param_sizes[i];
    }

    // Initialize other memory pointers in the model to NULL as they are not yet allocated or used
    model->activations_memory = NULL;
    model->gradients_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;

    // Initialize scalar values related to model operation
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
}

void gpt_build_from_checkpoint(GPT *model, const char *checkpoint_path) {
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { fprintf(stderr, "Bad magic model file\n"); exit(EXIT_FAILURE); }
    printf("model_header[0]: %i", model_header[0]);
    if (model_header[1] != 3) {
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt.py`\n");
        exit(EXIT_FAILURE);
    }

    model->configuration.max_seq_len = model_header[2];
    model->configuration.vocab_size = model_header[3];
    model->configuration.num_layers = model_header[4];
    model->configuration.num_heads = model_header[5];
    model->configuration.channels = model_header[6];
    model->configuration.padded_vocab_size = model_header[7];

    fill_in_parameter_sizes(model->param_sizes, model->configuration);

    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;
    printf("num_parameter: %i\n", num_parameters);

    // Allocate memory for model parameters on GPU and set pointers in the model structure
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);

    // Allocate temporary memory on CPU to read parameters from a file
    float* params_memory_cpu = (float*)mallocCheck(num_parameters * sizeof(float));
    freadCheck(params_memory_cpu, sizeof(float), num_parameters, model_file);

    // Copy parameters from CPU memory to GPU memory
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free the temporary CPU memory and close the model file
    free(params_memory_cpu);
    fcloseCheck(model_file);

    // Initialize other memory pointers in the model to NULL as they are not yet allocated or used
    model->activations_memory = NULL;
    model->gradients_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;

    // Initialize scalar values related to model operation
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
}

void fill_in_activation_sizes(size_t* act_sizes, int B, int T, GPTConfig config) {
    size_t paddedVocabSize = config.padded_vocab_size;
    size_t numLayers = config.num_layers;
    size_t numHeads = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = numLayers * B * T * C; // ln1
    act_sizes[2] = numLayers * B * T; // ln1_mean
    act_sizes[3] = numLayers * B * T; // ln1_rstd
    act_sizes[4] = numLayers * B * T * C; // atty
    act_sizes[5] = numLayers* B * numHeads * T * T; // att
    act_sizes[6] = numLayers * B * T * C; // attproj
    act_sizes[7] = numLayers * B * T * C; // residual2
    act_sizes[8] = numLayers * B * T * C; // ln2
    act_sizes[9] = numLayers * B * T; // ln2_mean
    act_sizes[10] = numLayers * B * T; // ln2_rstd
    act_sizes[11] = numLayers * B * T * 4*C; // fch
    act_sizes[12] = numLayers * B * T * 4*C; // fch_gelu
    act_sizes[13] = numLayers * B * T * C; // fcproj
    act_sizes[14] = numLayers * B * T * C; // residual3
    act_sizes[15] = B * T * C; // lnf
    act_sizes[16] = B * T; // lnf_mean
    act_sizes[17] = B * T; // lnf_rstd
    act_sizes[18] = B * T; // losses
    act_sizes[19] = numLayers * B * T * 3*C; // qkvr
    act_sizes[20] = B * T * max(3*C, max(numHeads*T, paddedVocabSize)); // output / scratch
}

float* malloc_and_point(float** targets[], const size_t* act_sizes, int n) {
    size_t num_activations = 0;
    for (size_t i = 0; i < n; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, num_activations * sizeof(float)));
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < n; i++) {
        *(targets[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

float* malloc_and_point_activations(ActivationPointerTensors* acts, const size_t* act_sizes) {
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->atty,
        &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->losses, &acts->qkvr, &acts->output
    };
    return malloc_and_point(ptrs, act_sizes, NUM_ACTIVATION_TENSORS);
}

__device__ inline float4 add_float4(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void encoder_forward_kernel(float* outputEmbeddings,
                                        const int* inputTokens, const float* tokenEmbeddingWeights, const float* positionEmbeddingWeights,
                            int B, int T, int embeddingSize, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int batchIndex = idx / (T * embeddingSize);
        int timeIndex = (idx / embeddingSize) % T;
        int embeddingIndex = idx % embeddingSize;

        int tokenIndx = inputTokens[batchIndex * T + timeIndex];

        outputEmbeddings[batchIndex * T * embeddingSize + timeIndex * embeddingSize + embeddingIndex] = 
            tokenEmbeddingWeights[tokenIndx * embeddingSize + embeddingIndex] + positionEmbeddingWeights[timeIndex * embeddingSize + embeddingIndex];
    }
}
void encoder_forward_simple(float* outputEmbeddings,
                            const int* inputTokens, const float* tokenEmbeddingWeights, const float* positionEmbeddingWeights,
                            int B, int T, int embeddingSize){
    int block_size = 1024;
    int N = B * T * embeddingSize;
    int grid_size = CEIL_DIV(N, block_size);
    encoder_forward_kernel<<<grid_size, block_size>>>(outputEmbeddings, inputTokens, tokenEmbeddingWeights, positionEmbeddingWeights, B, T, embeddingSize, N);
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
}

__global__ void encoder_forward_kernel3(float4* outputEmbeddings,
                                        const int* inputTokens, const float4* tokenEmbeddingWeights, const float4* positionEmbeddingWeights,
                                        int B, int T, int embeddingSize) {
    // Calculate the number of vectorized elements (as we're working with float4)
    int C4 = embeddingSize / 4;
    // Compute the global index for the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the total number of vectorized elements to process
    int N = B * T * C4;
    // Only perform operations if the current index is within the valid range
    if (idx < N) {
        // Calculate batch and time indices from the flattened index
        int batchTimeIndex = idx / C4;
        int batchIndex = batchTimeIndex / T;
        int timeIndex = batchTimeIndex % T;

        int c4 = idx % C4;

        int ix = inputTokens[batchIndex * T + timeIndex];

        outputEmbeddings[batchIndex * T * C4 + timeIndex * C4 + c4] = add_float4(tokenEmbeddingWeights[ix * C4 + c4], positionEmbeddingWeights[timeIndex * C4 + c4]);
    }
}
void encoder_forward(float* outputEmbeddings,
                    const int* inputTokens, const float* tokenEmbeddingWeights, const float* positionEmbeddingWeights,
                    int B, int T, int embeddingSize) {
    assert(embeddingSize % 4 == 0);
    const int block_size = 512;
    const int N = B * T * embeddingSize;
    const int grid_size = CEIL_DIV(N / 4, block_size);
    encoder_forward_kernel3<<<grid_size, block_size>>>((float4*) outputEmbeddings, inputTokens, (float4*) tokenEmbeddingWeights, (float4*) positionEmbeddingWeights, B, T, embeddingSize);
    cudaCheck(cudaGetLastError());
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void compareSpeedFunction(float* d_output, int N, std::function<void()> functionOne, std::function<void()> functionTwo) {
    float* h_temp;
    cudaMallocHost(&h_temp, N * sizeof(float));  // Allocate pinned memory for fast transfer
    cudaMemcpy(h_temp, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    double start_time, end_time, time_one, time_two;
    
    start_time = get_time();
    functionOne();
    cudaDeviceSynchronize();
    end_time = get_time();
    time_one = end_time - start_time;

    cudaMemcpy(d_output, h_temp, N * sizeof(float), cudaMemcpyHostToDevice);

    start_time = get_time();
    functionTwo();
    cudaDeviceSynchronize();
    end_time = get_time();
    time_two = end_time - start_time;

    printf("First function time: %.6f ms, Second function time: %.6f ms\n", time_one, time_two);
    cudaFreeHost(h_temp);
}

__global__ void printFirstFive(float* array) {
    int idx = threadIdx.x;
    printf("%f ", array[idx]);
    __syncthreads();
    if (idx == 0){
        printf("\n");
    }
}

__global__ void layernorm_forward_kernel(float* out, float* mean, float* rstd, 
                                        float* inp, float* weight, float* bias,
                        int B, int T, int C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int batchIndex = idx / (T * C);
        int timeIndex = (idx / C) % T;
        int cIndex = idx % C;

        float i_mean = 0.0f;
        for (int i = 0; i < C; ++i) {
            i_mean += inp[batchIndex * T * C + timeIndex * C + i];
        }
        i_mean /= C;

        float i_var = 0.0f;
        for (int i = 0; i < C; ++i) {
            i_var += (inp[batchIndex * T * C + timeIndex * C + i] - i_mean)*
                    (inp[batchIndex * T * C + timeIndex * C + i] - i_mean);
        }
        i_var /= C;

        float i_rstd = rsqrt(i_var + 1e-5);
        float normalized = (inp[batchIndex * T * C + timeIndex * C + cIndex] - i_mean) * i_rstd;

        out[batchIndex * T * C + timeIndex * C + cIndex] = weight[cIndex] * normalized + bias[cIndex];

        if (idx % C == 0) {
            rstd[batchIndex * T + timeIndex] = i_rstd;
            mean[batchIndex * T + timeIndex] = i_mean;
        }
    }
}
void layernorm_forward_simple(float* out, float* mean, float* rstd,
                        float* inp, float* weight, float* bias,
                        int B, int T, int C) {
    // Define the block size for CUDA kernel execution; a typical value for optimal performance
    const int block_size = 1024;
    const int N = B * T * C;
    const int grid_size = CEIL_DIV(N, block_size);
    // Launch the layer normalization kernel with the calculated grid and block sizes
    layernorm_forward_kernel<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, B, T, C, N);
    // Check for any errors during the execution of the kernel
    cudaCheck(cudaGetLastError());
}

__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                        const float* __restrict__ inp, const float* __restrict__ weight,
                                        const float* __restrict__ bias, int N, int C) {
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N) {
        return;
    }

    const float* x = inp + idx * C;

    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    sum = cooperative_groups::reduce(warp, sum, cooperative_groups::plus<float>{});
    float m = sum / C;
    if (warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cooperative_groups::reduce(warp, sum, cooperative_groups::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if (warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}
void layernorm_forward(float* out, float* mean, float* rstd,
                        float* inp, float* weight, float* bias,
                        int B, int T, int C) {
    // Define the block size for CUDA kernel execution; a typical value for optimal performance
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * 32, block_size);
    // Launch the layer normalization kernel with the calculated grid and block sizes
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    // Check for any errors during the execution of the kernel
    cudaCheck(cudaGetLastError());
}

__device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}

__device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}

__global__ void __launch_bounds__(16*16, 2) matmul_forward_kernel4(float* out,
                                                                    const float* inp, const float* weight, const float* bias,
                                                                    int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // each thread handles 8x8 elements; each block 128 by 128 elements.
    int oc = 8*(blockIdx.y * blockDim.y + threadIdx.y);

    __shared__ float lhs_s[128][32];
    __shared__ float rhs_s[128][32];

    inp += 128 * blockIdx.x * C;
    weight += 128 * blockIdx.y * C;
    out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    float vals[8][8] = {};
    if(bias != NULL) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                float4 b = ld_vec(bias + oc + j);
                vals[i][j+0] = b.x;
                vals[i][j+1] = b.y;
                vals[i][j+2] = b.z;
                vals[i][j+3] = b.w;
            }
        }
    }

    int si_start = 4*(16 * threadIdx.y + threadIdx.x);
    for (int so = 0; so < C; so +=32) {
        __syncthreads();
        int xmod8 = threadIdx.x % 8;
        int xby8 = threadIdx.x / 8;
        int xo = 4 * xmod8;
        for(int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        __syncthreads();

        for (int si = si_start; si < si_start + 32; si += 4) {
            float4 rhs[8];
            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
            }

            for (int ii = 0; ii < 8; ++ii) {
                float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            st_vec(out + (8*threadIdx.x+i) * OC + 8*threadIdx.y + j, result);
        }
    }
}

__global__ void matmul_forward_kernel(float* out,
        const float* inp, const float* weight, const float* bias,
        int B, int T, int inputChannels, int outputChannels){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < B * T && col < outputChannels){
        float sum = 0.0f;
        for (int i = 0; i < inputChannels; ++i) {
            sum += inp[row * inputChannels + i] * weight[inputChannels * col + i];
        }
        if (bias != NULL) sum += bias[col];
        out[row * outputChannels + col] = sum;
    }
}

void matmul_forward_simple(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int inputChannels, int outputChannels) {
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(B * T, blockDim.x), CEIL_DIV(outputChannels, blockDim.y));
    matmul_forward_kernel<<<gridDim, blockDim>>>(out, inp, weight, bias, B, T, inputChannels, outputChannels);
    cudaCheck(cudaGetLastError());
}

void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int inputChannels, int outputChannels) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // Define the size of the square block for the kernel's thread block dimensions
    int sqrt_block_size = 16;

    // Calculate grid dimensions for launching the CUDA kernel
    // Divide the total elements (batchSize * sequenceLength) and outputChannels by a factor to optimize performance
    dim3 gridDim(CEIL_DIV(B * T, 8*sqrt_block_size), CEIL_DIV(outputChannels, 8*sqrt_block_size));
    // Define block dimensions as a square of size sqrtBlockSize x sqrtBlockSize
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    // Launch the matrix multiplication kernel with calculated dimensions
    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, inputChannels, outputChannels);
    cudaCheck(cudaGetLastError());
}

__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}



__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx = (gridDim.x - blockIdx.x - 1) * warp.meta_group_size() + warp.meta_group_rank(); // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const float* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cooperative_groups::reduce(warp, maxval, cooperative_groups::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = cooperative_groups::reduce(warp, sumval, cooperative_groups::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}

__global__ void unpermute_kernel(float* input, float* output, int batch_size, int sequence_length, int num_heads, int head_dim) {
    // Kernel Description:
    // This kernel rearranges a tensor from shape (B, NH, N, d) back to (B, N, NH, d).
    // Where:
    //   - B: batch size
    //   - NH: number of heads
    //   - N: sequence length
    //   - d: head dimension
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    // Thread Index:
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Total Number of Elements:
    int total_elements = batch_size * num_heads * sequence_length * head_dim;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (global_thread_id < total_elements) {
        int batch_index = global_thread_id / (num_heads * sequence_length * head_dim);
        int remaining_elements_input = global_thread_id % (num_heads * sequence_length * head_dim);
        int head_index = remaining_elements_input / (sequence_length * head_dim);
        remaining_elements_input %= (sequence_length * head_dim);
        int sequence_index = remaining_elements_input / head_dim;
        int dim_index = remaining_elements_input % head_dim;

        int other_idx = (batch_index * sequence_length * num_heads * head_dim) +
                           (sequence_index * num_heads * head_dim) +
                           (head_index * head_dim) +
                           dim_index;
        output[other_idx] = __ldcs(&input[global_thread_id]);
    }
}

__global__ void unpermute_kernel_simple(float* input, float* output, int batch_size, int sequence_length, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * sequence_length * head_dim;

    if (idx < total_elements) {
        int batch_index = idx / (num_heads * sequence_length * head_dim);
        int remainElement = idx % (num_heads * sequence_length * head_dim);
        int head_indx = remainElement / (sequence_length * head_dim);
        remainElement = remainElement % (sequence_length * head_dim);
        int sequence_index = remainElement / head_dim;
        int dim_index = remainElement % head_dim;

        int new_index = batch_index * sequence_length * num_heads * head_dim +
                        sequence_index * num_heads * head_dim +
                        head_indx * head_dim +
                        dim_index;
        output[new_index] = input[idx];
    }
}

__global__ void softmax_forward_kernel(float* out, float inv_temperature, const float* inp, int B, int T, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * NH * T * T;
    if (idx < total_elements) {
        int batchIndex = total_elements / (NH * T * T);
        int leftElement = total_elements % (NH * T * T);
        int headIndex = leftElement / (T * T);
        leftElement = leftElement % (T * T);
        int sequenceIndex = leftElement / T;
        int inner_sequenceIndex = leftElement % T;

        int startIndex = (batchIndex * NH * T * T + headIndex * T * T + sequenceIndex * T);
        float maxval = -FLT_MAX;
        for (int i = 0; i < T; ++i){
            maxval = max(inp[startIndex + i], maxval);
        }
        float sum = 0.0f;
        for (int i = 0; i < T; ++i){
            sum += expf(inv_temperature * (inp[startIndex + i] - maxval));
        }
        out[startIndex + inner_sequenceIndex] = expf(inv_temperature * (inp[startIndex + inner_sequenceIndex] - maxval)) / sum;
    }
}



__global__ void permute_kernel(float* queries, float* keys, float* values,
                               const float* inp,
                               int batch_size, int sequence_length, int num_heads, int head_dim) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * sequence_length * head_dim;

    if (global_thread_id < total_elements) {
        int batch_index = global_thread_id / (num_heads * sequence_length * head_dim);
        int remaining_elements = global_thread_id % (num_heads * sequence_length * head_dim);
        int head_index = remaining_elements / (sequence_length * head_dim);
        remaining_elements = remaining_elements % (sequence_length * head_dim);
        int sequence_index = remaining_elements / head_dim;
        int dim_index = remaining_elements % head_dim;
        // Calculate the corresponding index in the input tensor (combined_qkv):
        // The input tensor is organized as [B, N, {Q,K,V}, NH, d]
        int input_index = (batch_index * sequence_length * 3 * num_heads * head_dim) +
                          (sequence_index * 3 * num_heads * head_dim) +
                          (0 * num_heads * head_dim) + // Offset for Q (0), K (1), V (2)
                          (head_index * head_dim) +
                          dim_index;
        queries[global_thread_id] = __ldcs(&inp[input_index]);
        keys[global_thread_id] = __ldcs(&inp[input_index + 1 * num_heads * head_dim]);  // Offset for K
        values[global_thread_id] = __ldcs(&inp[input_index + 2 * num_heads * head_dim]); // Offset for V
    }
}
__global__ void permute_kernel_simple(float* q, float* k, float* v, float* input, int batch_size, int sequence_length, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * sequence_length * num_heads * head_dim;
    if (idx < total_elements){
        int batchIndex = idx / (sequence_length * num_heads * head_dim);
        int leftElement = idx % (sequence_length * num_heads * head_dim);
        int timeIndex = leftElement / (num_heads * head_dim);
        leftElement = leftElement % (num_heads * head_dim);
        int headIndex = leftElement / head_dim;
        int dimIndex = leftElement % head_dim;
        int index = batchIndex * sequence_length * 3 * num_heads * head_dim
                    + timeIndex * 3 * num_heads * head_dim
                    + 0 * num_heads * head_dim
                    + headIndex * head_dim
                    + dimIndex;
        q[index] = input[index + 0 * num_heads * head_dim];
        k[index] = input[index + 1 * num_heads * head_dim];
        v[index] = input[index + 2 * num_heads * head_dim];
    }
}
void attention_forward_simple(float* out, float* qkvr, float*att,
                        float* inp,
                        int B, int T, int C, int NH) {
    const int block_size = 256;

    int HS = C / NH;

    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel_simple<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, C);
    cudaCheck(cudaGetLastError());

    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* preatt = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));

    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T, block_size);
    softmax_forward_kernel<<<grid_size, block_size>>>(att, scale, preatt, B, T, NH);
    cudaCheck(cudaGetLastError());

    float* vaccum = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_simple<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

void attention_forward(float* out, float* qkvr, float*att,
                        float* inp,
                        int B, int T, int C, int NH) {
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH;

    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* preatt = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));

    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    float* vaccum = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

__global__ void residual_forward_kernel(float* output, float* input1, float* input2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = __ldcs(&input1[idx]) + __ldcs(&input2[idx]);
    }
}
void residual_forward(float* output, float* input1, float* input2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel<<<grid_size, block_size>>>(output, input1, input2, N);
    cudaCheck(cudaGetLastError());
}

__global__ void residual_forward_kernel_simple(float* output, float* input1, float* input2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input1[idx] + input2[idx];
    }
}
void residual_forward_simple(float* output, float* input1, float* input2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel_simple<<<grid_size, block_size>>>(output, input1, input2, N);
    cudaCheck(cudaGetLastError());
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel(float* output, const float* input, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = input[i];
        float cube = 0.044715f * xi * xi * xi;
        output[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}
void gelu_forward(float* output, const float* input, int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_forward_kernel<<<grid_size, block_size>>>(output, input, N);
    cudaCheck(cudaGetLastError());
}

__global__ void gelu_forward_kernel_simple(float* output, const float* input, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = input[i];
        output[i] = 0.5f * xi * (1 + tanhf(GELU_SCALING_FACTOR * (xi + 0.044715f * xi * xi * xi)));
    }
}
void gelu_forward_simple(float* output, const float* input, int N) {
    int block_size = 128;
    int grid_size = CEIL_DIV(N, block_size);
    gelu_forward_kernel_simple<<<grid_size, block_size>>>(output, input, N);
    cudaCheck(cudaGetLastError());
}

struct SoftmaxParams {
    float Scale;
    float Offset;
};

__device__ SoftmaxParams prepare_softmax_blockwide_nofloat4(cooperative_groups::thread_block_tile<32>& warp,
                                                   int idx, const float* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const float* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = fmaxf(thread_maxval, v);
        thread_sumval *= expf((old_maxval - thread_maxval));
        thread_sumval += expf(v - thread_maxval);
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = cooperative_groups::reduce(warp, thread_maxval, cooperative_groups::greater<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = cooperative_groups::reduce(warp, warp_maxval, cooperative_groups::greater<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = cooperative_groups::reduce(warp, thread_sumval, cooperative_groups::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = cooperative_groups::reduce(warp, warp_sumval, cooperative_groups::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

__global__ void fused_classifier_kernel3(float* logits, float* losses, float* probs,
                                         const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float v = __ldcs(&logits_vec[i]);
        float prob = expf(v - sp.Offset) * sp.Scale;
        if (probs != NULL) {
            probs[idx * P + i] = prob;
        }
        float indicator = (i == ix) ? 1.0f : 0.0f;
        logits[idx * P + i] = (prob - indicator) * dloss;
    }
}
void fused_classifier3(float* logits, float* losses,
                        const float* dlosses, const int* targets,
                        int B, int T, int vocabSize, int paddedVocabSize) {
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel3<<<grid_size, block_size>>>(logits, losses, NULL, dlosses, targets, B, T, vocabSize, paddedVocabSize);
    cudaCheck(cudaGetLastError());
}

__global__ void cross_entropy_kernel(float* logits, float* losses,
                        const float* dlosses, const int* targets,
                        int B, int T, int vocabSize, int paddedVocabSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;
    if (idx < N) {
        int target_class = targets[idx];
        float loss = 0.0f;

        // Cross-entropy loss computation
        loss = -logf(logits[idx * paddedVocabSize + target_class]); // Negative log of the correct class
        // printf("loss: %f\n", loss);
        // printf("logits[idx * paddedVocabSize + target_class]: %f\n", logits[idx * paddedVocabSize + target_class]);
        losses[idx] = loss;
        float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    }
}
__global__ void softmax_forward_kernel_l(float* logits, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * T * C;
    if (idx < total_elements) {
        int batchIndex = idx / (T * C);
        int leftElement = idx % (T * C);
        int timeIndex = leftElement / C;
        int channelIndex = leftElement % C;

        int startIndex = (batchIndex * T * C + timeIndex * C);
        float maxval = -FLT_MAX;
        for (int i = 0; i < C; ++i){
            maxval = max(logits[startIndex + i], maxval);
        }
        float sum = 0.0f;
        for (int i = 0; i < T; ++i){
            sum += expf(logits[startIndex + i] - maxval);
        }
        logits[startIndex + channelIndex] = expf(logits[startIndex + channelIndex] - maxval) / sum;
        // if (logits[startIndex + channelIndex] > 1 || logits[startIndex + channelIndex] <= 0){
        //     printf("logits[startIndex + channelIndex]: %f\n", logits[startIndex + channelIndex]);
        // }
    }
}
void cross_entropy(float* logits, float* losses,
                        const float* dlosses, const int* targets,
                        int B, int T, int vocabSize, int paddedVocabSize) {
    int block_size = 1024;
    int N = B * T * paddedVocabSize;
    softmax_forward_kernel_l<<<CEIL_DIV(N, block_size), block_size>>>(logits, B, T, paddedVocabSize);
    cudaCheck(cudaGetLastError());
    N = B * T;
    cross_entropy_kernel<<<CEIL_DIV(N, block_size), block_size>>>(logits, losses, dlosses, targets, B, T, vocabSize, paddedVocabSize);
    cudaCheck(cudaGetLastError());
}

void gpt_forward(GPT* model, int* inputs, int* targets, int B, int T) {
    // Check if the model's parameter memory has been initialized
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // Extract configuration details from the model
    int vocabSize = model->configuration.vocab_size;
    int paddedVocabSize = model->configuration.padded_vocab_size;
    int numLayers = model->configuration.num_layers;
    int numHeads = model->configuration.num_heads;
    int C = model->configuration.channels;

    // Validate input and target indices to ensure they are within the defined vocabulary size
    for (int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < vocabSize);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < vocabSize);
        }
    }

    // Check if activation memory has not been initialized and set it up if necessary
    if (model->activations_memory == NULL) {
        model->batch_size = B;
        model->seq_len = T;

        model->activation_sizes[0] = B * T * (size_t)C; // encoded
        model->activation_sizes[1] = (size_t)numLayers * B * T * (size_t)C; // ln1
        model->activation_sizes[2] = (size_t)numLayers * B * T; // ln1_mean
        model->activation_sizes[3] = (size_t)numLayers * B * T; // ln1_rstd
        model->activation_sizes[4] = (size_t)numLayers * B * T * (size_t)C; // atty
        model->activation_sizes[5] = (size_t)numLayers* B * numHeads * T * T; // att
        model->activation_sizes[6] = (size_t)numLayers * B * T * (size_t)C; // attproj
        model->activation_sizes[7] = (size_t)numLayers * B * T * (size_t)C; // residual2
        model->activation_sizes[8] = (size_t)numLayers * B * T * (size_t)C; // ln2
        model->activation_sizes[9] = (size_t)numLayers * B * T; // ln2_mean
        model->activation_sizes[10] = (size_t)numLayers * B * T; // ln2_rstd
        model->activation_sizes[11] = (size_t)numLayers * B * T * 4*(size_t)C; // fch
        model->activation_sizes[12] = (size_t)numLayers * B * T * 4*(size_t)C; // fch_gelu
        model->activation_sizes[13] = (size_t)numLayers * B * T * (size_t)C; // fcproj
        model->activation_sizes[14] = (size_t)numLayers * B * T * (size_t)C; // residual3
        model->activation_sizes[15] = B * T * (size_t)C; // lnf
        model->activation_sizes[16] = B * T; // lnf_mean
        model->activation_sizes[17] = B * T; // lnf_rstd
        model->activation_sizes[18] = B * T; // losses
        model->activation_sizes[19] = (size_t)numLayers * B * T * 3*(size_t)C; // qkvr
        model->activation_sizes[20] = B * T * max(3*(size_t)C, max((size_t)numHeads*T, (size_t)paddedVocabSize)); // output / scratct
        
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->activation_sizes[i];
        }
        model->num_activations = num_activations;
        printf("num_activations: %i\n", num_activations);

        float* acts_memory;
        cudaCheck(cudaMalloc((void**)&acts_memory, num_activations * sizeof(float)));
        model->activations_memory = acts_memory;

        ActivationPointerTensors* acts = &model->acts;
        float** ptrs[] = {
            &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->atty,
            &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
            &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
            &acts->lnf_mean, &acts->lnf_rstd, &acts->losses, &acts->qkvr, &acts->output
        };
        float* acts_memory_iterator = acts_memory;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            *(ptrs[i]) = acts_memory_iterator;
            acts_memory_iterator += model->activation_sizes[i];
        }

        printf("allocated %zu MiB for activations\n", (num_activations * sizeof(float)) >> 20);

        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
    } else {
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }

    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    if (targets != NULL) {
        cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    }

    GPTParameterPointerTensors params = model->params;
    ActivationPointerTensors acts = model->acts;
    float* residual;
    encoder_forward_simple(acts.encoded, model->inputs, params.weightsTokenEmbedding, params.positionEmbeddingWeights, B, T, C);

    for (int l = 0; l < numLayers; l++) {
        // Determine the starting point for the residual connections
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // Layer-specific parameter pointers for layer normalization and attention mechanism
        float* l_layerNorm1Weights = params.layerNorm1Weights + l * C;
        float* l_layerNorm1Bias = params.layerNorm1Biases + l * C;
        float* l_queryKeyValueWeights = params.attentionQKVWeights + l * 3*C * C;
        float* l_queryKeyValueBias = params.attentionQKVBiases + l * 3*C;
        float* l_attentionProjWeights = params.attentionOutputWeights + l * C * C;
        float* l_attentionProjBias = params.attentionOutputBiases + l * C;
        float* l_layerNorm2Weights = params.layerNorm2Weights + l * C;
        float* l_layerNorm2Bias = params.ln2b + l * C;
        float* l_fullyConnectedWeights = params.fcw + l * 4*C * C;
        float* l_fullyConnectedBias = params.fcb + l * 4*C;
        float* l_fullyConnectedProjWeights = params.fcprojw + l * C * 4*C;
        float* l_fullyConnectedProjBias = params.fcprojb + l * C;

        // Layer-specific activations pointers for the computation of each layer
        float* l_layerNorm1Output = acts.ln1 + l * B * T * C;
        float* l_layerNorm1Mean = acts.ln1_mean + l * B * T;
        float* l_layerNorm1RStd = acts.ln1_rstd + l * B * T;
        float* l_queryKeyValueResult = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * numHeads * T * T;
        float* l_attentionProjection = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fullyConnectedGelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fullyConnectedProjection = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        float* scratch = acts.output;

        layernorm_forward_simple(l_layerNorm1Output, l_layerNorm1Mean, l_layerNorm1RStd, residual, l_layerNorm1Weights, l_layerNorm1Bias, B, T, C);
        // if (l == 0) printFirstFive<<<1, 5>>>(l_layerNorm1Output);
        matmul_forward_simple(scratch, l_layerNorm1Output, l_queryKeyValueWeights, l_queryKeyValueBias, B, T, C, 3*C);
        // if (l == 0) printFirstFive<<<1, 5>>>(scratch);
        attention_forward_simple(l_atty, l_queryKeyValueResult, l_att, scratch, B, T, C, numHeads);
        // if (l == 0) printFirstFive<<<1, 5>>>(l_atty);
        matmul_forward_simple(l_attentionProjection, l_atty, l_attentionProjWeights, l_attentionProjBias, B, T, C, C);
        // if (l == 0) printFirstFive<<<1, 5>>>(l_attentionProjection);
        residual_forward_simple(l_residual2, residual, l_attentionProjection, B*T*C);
        // if (l == 0) printFirstFive<<<1, 5>>>(l_residual2);

        layernorm_forward_simple(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_layerNorm2Weights, l_layerNorm2Bias, B, T, C);
        // if (l == 0) printFirstFive<<<1, 5>>>(l_ln2);
        matmul_forward_simple(l_fch, l_ln2, l_fullyConnectedWeights, l_fullyConnectedBias, B, T, C, 4*C);
        // if (l == 0) printFirstFive<<<1, 5>>>(l_fch);
        gelu_forward_simple(l_fullyConnectedGelu, l_fch, B*T*4*C);
        // if (l == 0) printFirstFive<<<1, 5>>>(l_fullyConnectedGelu);
        matmul_forward_simple(l_fullyConnectedProjection, l_fullyConnectedGelu, l_fullyConnectedProjWeights, l_fullyConnectedProjBias, B, T, 4*C, C);
        // if (l == 0) printFirstFive<<<1, 5>>>(l_fullyConnectedProjection);
        residual_forward_simple(l_residual3, l_residual2, l_fullyConnectedProjection, B*T*C);
        // if (l == 0) printFirstFive<<<1, 5>>>(l_residual3);
    }
    residual = acts.residual3 + (numLayers-1) * B * T * C;
    layernorm_forward_simple(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.finalLayerNormWeights, params.finalLayerNormBiases, B, T, C);
    matmul_forward_simple(acts.output, acts.lnf, params.weightsTokenEmbedding, NULL, B, T, C, paddedVocabSize);

    if (targets != NULL) {
        cross_entropy(acts.output, acts.losses, NULL, model->targets, B, T, vocabSize, paddedVocabSize);
        cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
        float mean_loss = 0.0f;
        for (int i=0; i < B * T; i++) { mean_loss += model->cpu_losses[i]; }
        mean_loss /= B * T;
        model->mean_loss = mean_loss;
    } else {
        model->mean_loss = -1.0f;
    }
}

void gpt_zero_grad(GPT* model) {
    if (model->grads_acts_memory != NULL) { cudaCheck(cudaMemset(model->grads_acts_memory, 0, model->num_grad_acts * sizeof(float))); }
    if (model->gradients_memory != NULL) { cudaCheck(cudaMemset(model->gradients_memory, 0, model->num_parameters * sizeof(float))); }
}

void fill_in_grad_act_sizes(size_t* act_sizes, int B, int T, GPTConfig config) {
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * 4 * C;
    act_sizes[1] = B * NH * T * T;
    act_sizes[2] = B * T * C;
}

#define NUM_BACKWARD_TENSORS 3
float* malloc_and_point_backward(GradActTensors* acts, const size_t* act_sizes) {
    float** ptrs[] = {
        &acts->bt4c, &acts->preatt, &acts->residual3
    };
    return malloc_and_point(ptrs, act_sizes, NUM_BACKWARD_TENSORS);
}

// This kernel performs a column-wise reduction of the matrix 'dout', equivalent to:
// dbias = dout.sum((0,1)) in PyTorch
// The kernel design uses one block to reduce multiple columns, with each block handling a width of 32 columns
// to ensure coalesced memory access. After local reductions by warps, results are accumulated in shared memory.
__global__ void matmul_backward_bias_kernel4(float* dbias, const float* dout, int B, int T, int OC) {
    // Launch this kernel with a 1D grid dimension of output_channels / 32
    // Example: Assume block_size is 128
    extern __shared__ float shared_memory[]; // Array size equals block_size (128)
    const int warp_index = threadIdx.x / warpSize; // Warp index within the block (0, 1, 2, or 3)
    const int thread_lane = threadIdx.x % warpSize; // Thread index within the warp (0 through 31)
    const int column_start = blockIdx.x * warpSize; // Starting column index for this block
    const int num_warps = blockDim.x / warpSize; // Number of warps per block, e.g., 4

    // Each thread processes a specific column, and threads with the same 'thread_lane' across different warps
    // reduce the same column. Hence, four threads, each from a different warp, cooperatively reduce a single column.
    const float* column_pointer = dout + column_start + thread_lane;

    // Sum up the 'output_gradients' column-wise. Each thread in a warp starts at an offset determined by its warp index
    // and skips 'num_warps' rows, ensuring all threads collectively cover all rows of the current column.
    float column_sum = 0.0f;
    for (int row = warp_index; row < B * T; row += num_warps) {
        column_sum += column_pointer[row * OC];
    }
    // Store the result of each thread's reduction in shared memory
    shared_memory[thread_lane + warp_index * warpSize] = column_sum;
    __syncthreads(); // Synchronize threads to ensure all reductions are complete

    // Reduce the partial sums stored in shared memory by the first warp
    if (warp_index == 0) {
        column_sum = 0.0f;
        for (int j = 0; j < num_warps; j++) {
            column_sum += shared_memory[thread_lane + j * warpSize];
        }
        // Update the bias gradient for each column processed by this block
        dbias[column_start + thread_lane] += column_sum;
    }
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                    float* dout, float* inp, float* weight,
                    int B, int T, int C, int OC) {
    float one = 1.0f;
    float zero = 0.0f;
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &one, weight, C, dout, OC, &zero, dinp, C));
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &one, inp, C, dout, OC, &one, dweight, C));
    if (dbias != NULL) {
        const int block_size = 1024;
        const int grid_size = OC / 32;
        matmul_backward_bias_kernel4<<<grid_size, block_size, block_size * sizeof(float)>>>(dbias, dout, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}

__global__ void layernorm_backward_kernel2(float* dinp, float* dweight, float* dbias,
                                           const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                                           int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int N = B * T;
    if(idx >= N) { return; } // thread guards

    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll
	for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
    dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias_shared[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight_shared[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
    __syncthreads();

    // write to global memory
	for(int i = threadIdx.x; i < C; i+= blockDim.x){
        atomicAdd(&dbias[i], dbias_shared[i]);
        atomicAdd(&dweight[i], dweight_shared[i]);
	}
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const  float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(32*N, block_size);
    size_t shared_mem_size = 2 * C * sizeof(float);
    layernorm_backward_kernel2<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}

__global__ void gelu_backward_kernel(float* dinp, const float* inp, const float* dout, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];
    }
}

void gelu_backward(float* dinp, const float* inp, const float* dout, const int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

__global__ void unpermute_kernel_backward(float* dinp, const float *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = dout[other_idx];
    }
}

__global__ void softmax_autoregressive_backward_kernel(float* dpreatt, const float* datt, const float* att,
                                                       int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int idx = blockIdx.y;
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx.x;

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    if (warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const float* att_bth = att + t * T;
        const float* datt_bth = datt + t * T;
        float* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        block_acc[warp.meta_group_rank()] = cooperative_groups::reduce(warp, local_sum, cooperative_groups::plus<float>{});
        block.sync();
        local_sum = cooperative_groups::reduce(warp, block_acc[warp.thread_rank()], cooperative_groups::plus<float>{});

        for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
            __stcs(dpreatt_bth + t3, scale * acc);
        }
    }
}

__global__ void permute_kernel_backward(float* dinp,
                                        const float* dq, const float* dk, const float* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + NH * d] = dk[idx];
        dinp[inp_idx + 2 * (NH * d)] = dv[idx];
    }
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* scratch,
                        const float* dout,
                        const float* qkvr, const float* att,
                        int B, int T, int C, int NH) {
    const int block_size = 256;
    int HS = C / NH; // head size
    const float one = 1.0f;
    const float zero = 0.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;
    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(scratch, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS, scratch, HS, T * HS, &zero, datt, T, T * T, B * NH));
    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, scratch, HS, T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));
    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    softmax_autoregressive_backward_kernel<<<dim3(T / 4, B * NH), 256>>>(dpreatt, datt, att, B, T, C, scale);
    cudaCheck(cudaGetLastError());
    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS, dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));
    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS, dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// really bad naive kernel with atomicAdd
__global__ void encoder_backward_kernel(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        atomicAdd(dwte_ix, *dout_btc);
        atomicAdd(dwpe_tc, *dout_btc);
    }
}

void encoder_backward(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_backward_kernel<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

void gpt_backward(GPT *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(EXIT_FAILURE);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->gradients_memory == NULL) {
        // allocate buffers for weight gradients
        // model->gradients_memory = malloc_and_point_parameters(&model->grads, model->param_sizes, 1);
        
        size_t num_parameters = 0;
        for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            num_parameters += model->param_sizes[i];
        }
        float* parametersMemory;
        cudaCheck(cudaMalloc((void**)&parametersMemory, num_parameters * sizeof(float)));

        GPTParameterPointerTensors* params = &model->grads;
        // Array of pointers to update the tensor pointers in 'tensors' struct to point to the correct locations within the allocated memory
        float** pointersToUpdate[] = {
            &params->weightsTokenEmbedding, &params->positionEmbeddingWeights, &params->layerNorm1Weights, &params->layerNorm1Biases, &params->attentionQKVWeights, &params->attentionQKVBiases,
            &params->attentionOutputWeights, &params->attentionOutputBiases, &params->layerNorm2Weights, &params->ln2b, &params->fcw, &params->fcb,
            &params->fcprojw, &params->fcprojb, &params->finalLayerNormWeights, &params->finalLayerNormBiases
        };
        // Iterate through the memory and assign each tensor pointer a location in the allocated block
        float* currentMemoryPosition = parametersMemory;
        for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            *(pointersToUpdate[i]) = currentMemoryPosition;
            currentMemoryPosition += model->param_sizes[i];
        }
        model->gradients_memory = parametersMemory;
        
        printf("allocated %zu MiB for parameter gradients\n", (model->num_parameters * sizeof(float)) >> 20);
        // we're going to be clever for the activations backward pass. we don't need to exactly
        // mirror the forward pass acrtivations and we will save memory.
        size_t bw_act_sizes[NUM_ACTIVATION_TENSORS];
        GPTConfig cfg = model->configuration;
        cfg.num_layers = 1; // copy the configuration but override number of layers to 1
        fill_in_grad_act_sizes(bw_act_sizes, model->batch_size, model->seq_len, cfg);
        // count up and allocate the space
        model->grads_acts_memory = malloc_and_point_backward(&model->grads_acts, bw_act_sizes);
        model->num_grad_acts = 0;
        for (int i = 0; i < NUM_BACKWARD_TENSORS; i++) {
            model->num_grad_acts += bw_act_sizes[i];
        }
        printf("allocated %zu MiB for activation gradients\n", (model->num_grad_acts * sizeof(float)) >> 20);
        // init gradients of parameters and activations to zero
        gpt_zero_grad(model);
    }

    // convenience shortcuts
    int B = model->batch_size;
    int T = model->seq_len;
    int Vp = model->configuration.padded_vocab_size;
    int L = model->configuration.num_layers;
    int NH = model->configuration.num_heads;
    int C = model->configuration.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    GPTParameterPointerTensors params = model->params; // for brevity
    GPTParameterPointerTensors grads = model->grads;
    ActivationPointerTensors acts = model->acts;
    GradActTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    // next: backward the classifier matmul
    // printFirstFive<<<1, 5>>>(acts.output);
    // printFirstFive<<<1, 5>>>(acts.lnf);
    // printFirstFive<<<1, 5>>>(params.wte);
    matmul_backward(grads_acts.bt4c, grads.weightsTokenEmbedding, NULL, acts.output, acts.lnf, params.weightsTokenEmbedding, B, T, C, Vp);
    // backward the final layernorm
    float* residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    float* dresidual = grads_acts.residual3; // the main buffer holding the gradient in the backward pass
    layernorm_backward(dresidual, grads.finalLayerNormWeights, grads.finalLayerNormBiases, grads_acts.bt4c, residual, params.finalLayerNormWeights, acts.lnf_mean, acts.lnf_rstd, B, T, C);
    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.layerNorm1Weights + l * C;
        float* l_qkvw = params.attentionQKVWeights + l * 3*C * C;
        float* l_attprojw = params.attentionOutputWeights + l * C * C;
        float* l_ln2w = params.layerNorm2Weights + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.layerNorm1Weights + l * C;
        float* dl_ln1b = grads.layerNorm1Biases + l * C;
        float* dl_qkvw = grads.attentionQKVWeights + l * 3*C * C;
        float* dl_qkvb = grads.attentionQKVBiases + l * 3*C;
        float* dl_attprojw = grads.attentionOutputWeights + l * C * C;
        float* dl_attprojb = grads.attentionOutputBiases + l * C;
        float* dl_ln2w = grads.layerNorm2Weights + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass

        // we need a B x T x C buffer; thankfully, the forward activation for lnf isn't needed anymore,
        // so we can co-opt it here.
        float* dl_btc = acts.lnf;
        float* dl_bt4c = grads_acts.bt4c;
        float* dl_preatt = grads_acts.preatt;

        // re-use scratch buffer of the forward pass
        float* scratch = acts.output;

        // backprop this layer
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_bt4c, l_fch, dl_bt4c, B*T*4*C);
        // if (l==0) printFirstFive<<<1, 5>>>(dl_bt4c);
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
        // if (l==0) printFirstFive<<<1, 5>>>(dl_btc);
        // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);
        // we more B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
        float* buffer_a = l_atty;
        float* buffer_b = l_fch;        // this is B x T x 4C, so even larger than what we need

        attention_backward(dl_bt4c, buffer_b, dl_preatt, scratch, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH);
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    cudaDeviceSynchronize();
    encoder_backward(grads.weightsTokenEmbedding, grads.positionEmbeddingWeights, dresidual, model->inputs, B, T, C);
}

__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

__global__ void adamw_kernel2(float* params_memory, float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_parameters) return;
    float grad = grads_memory[i];
    float m = m_memory[i];
    float v = v_memory[i];
    m = lerp(grad, m, beta1);
    m_memory[i] = m;
    v = lerp(grad * grad, v, beta2);
    v_memory[i] = v;
    m /= beta1_correction;  // m_hat
    v /= beta2_correction;  // v_hat
    params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

void gpt_update(GPT *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        cudaCheck(cudaMalloc((void**)&model->m_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&model->v_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->m_memory, 0, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, model->num_parameters * sizeof(float)));
        printf("allocated %zu MiB for AdamW optimizer state m\n", (model->num_parameters * sizeof(float)) >> 20);
        printf("allocated %zu MiB for AdamW optimizer state v\n", (model->num_parameters * sizeof(float)) >> 20);
    }

    int block_size = 512;
    int num_blocks = CEIL_DIV(model->num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel2<<<num_blocks, block_size>>>(model->params_memory, model->gradients_memory, model->m_memory, model->v_memory,
                                              model->num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

void gpt_free(GPT* model) {
    cudaCheck(cudaFree(model->params_memory));
    cudaCheck(cudaFree(model->gradients_memory));
    cudaCheck(cudaFree(model->m_memory));
    cudaCheck(cudaFree(model->v_memory));
    cudaCheck(cudaFree(model->activations_memory));
    cudaCheck(cudaFree(model->grads_acts_memory));
    cudaCheck(cudaFree(model->inputs));
    cudaCheck(cudaFree(model->targets));
    cudaFreeHost(model->cpu_losses);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_softmax(const float* logits, int n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int main(int argc, char *argv[]) {
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";

    const char* output_log_file = NULL;
    int B = 4;
    int T = 2048;
    float learning_rate = 3e-4f;
    int val_loss_every = 20;
    int val_max_steps = 20;
    int sample_every = 20;
    int genT = 64;
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_file = argv[i+1]; }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); }
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else { error_usage(); }
    }
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| Parameter             | Value                                              |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| train data pattern    | %-50s |\n", train_data_pattern);
    printf("| val data pattern      | %-50s |\n", val_data_pattern);
    printf("| output log file       | %-50s |\n", output_log_file == NULL ? "NULL" : output_log_file);
    printf("| batch size B          | %-50d |\n", B);
    printf("| sequence length T     | %-50d |\n", T);
    printf("| learning rate         | %-50f |\n", learning_rate);
    printf("| val_loss_every        | %-50d |\n", val_loss_every);
    printf("| val_max_steps         | %-50d |\n", val_max_steps);
    printf("| sample_every          | %-50d |\n", sample_every);
    printf("| genT                  | %-50d |\n", genT);
    printf("+-----------------------+----------------------------------------------------+\n");

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    

    cublasCheck(cublasCreate(&cublas_handle));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    printf("| device                | %-50s |\n", deviceProp.name);
    printf("| TF32                  | %-50s |\n", enable_tf32 ? "enabled" : "disabled");
    printf("+-----------------------+----------------------------------------------------+\n");

    GPT model;
    gpt_initialize(&model);
    // gpt_build_from_checkpoint(&model, "gpt2_124M.bin");
    printf("| max_sequence_length T | %-50d |\n", model.configuration.max_seq_len);
    printf("| vocab_size V          | %-50d |\n", model.configuration.vocab_size);
    printf("| padded_vocab_size Vp  | %-50d |\n", model.configuration.padded_vocab_size);
    printf("| num_layers L          | %-50d |\n", model.configuration.num_layers);
    printf("| num_heads NH          | %-50d |\n", model.configuration.num_heads);
    printf("| channels C            | %-50d |\n", model.configuration.channels);
    printf("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf("+-----------------------+----------------------------------------------------+\n");

    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_data_pattern, B, T, 0, 1, 0);
    int train_num_batches = train_loader.num_tokens / (B*T);
    int val_num_batches = val_loader.num_tokens / (B*T);
    if (val_num_batches > val_max_steps) { val_num_batches = val_max_steps; }
    printf("| train_num_batches     | %-50d |\n", train_num_batches);
    printf("| val_num_batches       | %-50d |\n", val_num_batches);
    printf("+-----------------------+----------------------------------------------------+\n");

    printf("allocated %d MiB for model parameters\n", (int)round(model.num_parameters * sizeof(float) / (1024 * 1024)));

    Logger logger;
    logger_init(&logger, output_log_file);

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    float* cpu_logits = (float*)mallocCheck(model.configuration.vocab_size * sizeof(float));

    struct timespec start, end;
    double total_sum_iteration_time_s = 0.0;
    #define GPT_EOT 50256
    for(int i = 0; i < B * T; ++i) {
        gen_tokens[i] = GPT_EOT;
    }
    // for (int t = 1; t < genT; t++) {
    //         // note that inference is very wasteful here because for each token
    //         // we re-calculate the forward pass for all of (B,T) positions from scratch
    //         // but the inference here is just for sanity checking anyway
    //         // and we can maybe optimize a bit more later, with careful tests
    //         gpt_forward(&model, gen_tokens, NULL, B, T);
    //         // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
    //         // we're in principle running B "inference streams" in parallel here
    //         // only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
    //         // get the V-dimensional vector probs[0, t-1, :]
    //         float* logits = model.acts.output + (t - 1) * model.configuration.padded_vocab_size;
    //         // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
    //         cudaCheck(cudaMemcpy(cpu_logits, logits, model.configuration.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    //         float coin = random_f32(&rng_state);
    //         int next_token = sample_softmax(cpu_logits, model.configuration.vocab_size, coin);
    //         gen_tokens[t] = next_token;
    //         // print the generated token, either using the Tokenizer or a fallback
    //         if (tokenizer.init_ok) {
    //             const char* token_str = tokenizer_decode(&tokenizer, next_token);
    //             safe_printf(token_str);
    //         } else {
    //             // fall back to printing the token id
    //             printf("%d ", next_token);
    //         }
    //         fflush(stdout);
    // }
    // printf("\n---\n");

    for (int i; i < 1; ++i) {
        dataloader_reset(&train_loader);
        for (int step = 0; step <= train_num_batches; step++) {
            int last_step = step == train_num_batches;

            if (last_step) {break; }

            clock_gettime(CLOCK_MONOTONIC, &start);
            dataloader_next_batch(&train_loader);

            gpt_forward(&model, train_loader.inputs, train_loader.targets, B, T);
            gpt_zero_grad(&model);
            gpt_backward(&model);
            cudaCheck(cudaDeviceSynchronize());
            gpt_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
            cudaCheck(cudaDeviceSynchronize());
            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
            total_sum_iteration_time_s += time_elapsed_s;
            int tokens_per_second = (B * T) / time_elapsed_s;
            printf("step %4d/%d: train loss %f (%f ms, %d tok/s)\n", step + 1, train_num_batches, model.mean_loss, time_elapsed_s * 1000, tokens_per_second);
            logger_log_train(&logger, step, model.mean_loss);
        } 
    }
    printf("total average iteration time: %f ms\n", total_sum_iteration_time_s / train_num_batches * 1000);

    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    gpt_free(&model);
    free(cpu_logits);
    free(gen_tokens);
    cublasCheck(cublasDestroy(cublas_handle));
    logger_free(&logger);

    return 0;
}
