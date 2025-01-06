#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
// Define a macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/// first we need write this model architecture, it will have 
// parameters and parameter gridients, most functional, 
// and because transformer architecture, it will have Embedding and layer norm weight
///

// Model Hyperparameters (Settings for the architecture)
typedef struct {
    int maxSequenceLength;        // Maximum length of the input sequence (e.g., 512 or 1024)
    int vocabularySize;           // Total number of tokens (words or subwords) in the vocabulary
    int numLayers;                // Number of transformer layers in the model (e.g., 12 or 24)
    int hiddenChannels;           // Number of hidden units or channels in each layer (e.g., 768 or 1024)
    int numAttentionHeads;        // Number of attention heads used in multi-head self-attention
} Hyperparameters;

// Pointers to Model Parameters (Weights and Biases for each layer)
typedef struct {
    float* tokenEmbeddingWeights;   // Weights for the token embeddings (converts token IDs to vectors)
    float* positionalEmbeddingWeights; // Weights for the positional encodings (adds positional information to token embeddings)
    float* layerNorm1Weights;        // Weights for the first layer normalization
    float* layerNorm1Biases;         // Biases for the first layer normalization
    float* attentionQueryWeights;    // Weights for the query part of attention mechanism (Q)
    float* attentionKeyWeights;      // Weights for the key part of attention mechanism (K)
    float* attentionValueWeights;    // Weights for the value part of attention mechanism (V)
    float* attentionQueryBiases;     // Biases for the query part of attention mechanism
    float* attentionKeyBiases;       // Biases for the key part of attention mechanism
    float* attentionValueBiases;     // Biases for the value part of attention mechanism
    float* attentionOutputWeights;   // Weights for the output of the attention mechanism
    float* attentionOutputBiases;    // Biases for the output of the attention mechanism
    float* layerNorm2Weights;        // Weights for the second layer normalization (after attention)
    float* layerNorm2Biases;         // Biases for the second layer normalization (after attention)
    float* feedForwardWeights;       // Weights for the feed-forward network (fully connected layers)
    float* feedForwardBiases;        // Biases for the feed-forward network
    float* projectionWeights;        // Weights for the final linear projection layer (before output)
    float* projectionBiases;         // Biases for the final linear projection layer
    float* finalLayerNormWeights;    // Weights for the final layer normalization
    float* finalLayerNormBiases;     // Biases for the final layer normalization
} ModelParameters;
#define NUMBER_OF_PARAMETERS 20

// GPT-3 Model Structure, combining hyperparameters and model parameters
struct GPT3Model{
    Hyperparameters hyperparameters;  // Hyperparameters for the model (settings like max sequence length, etc.)
    size_t paramSizes[NUMBER_OF_PARAMETERS];
    ModelParameters modelParameters;           // Pointer to the model's parameter array (weights and biases)
    float* parameterGradients;        // Pointer to the gradient array (used in backpropagation for training)
    size_t allParametersSize;
    int tokenEmbeddingSize;           // Size of token embeddings (dimensionality of word vectors, e.g., 768)
    int positionalEncodingSize;       // Size of positional encoding (same as token embedding size, typically)
    int layerNorm1Size;               // Size of the first layer normalization weights (same as hidden size)
    int layerNorm1BiasSize;           // Size of the first layer normalization biases
};

// Function to calculate and fill in the sizes of model parameters
void fill_parameter_sizes(size_t *paramSizes, size_t& totalParameterSize, const Hyperparameters& hyperparameters) {
    size_t vocabSize = hyperparameters.vocabularySize; // Vocabulary size (number of tokens)
    size_t hiddenSize = hyperparameters.hiddenChannels; // Number of hidden channels (features per token)
    size_t maxSeqLength = hyperparameters.maxSequenceLength; // Maximum sequence length
    size_t numLayers = hyperparameters.numLayers; // Number of transformer layers
    
    // Fill the parameter sizes based on the model configuration
    paramSizes[0] = vocabSize * hiddenSize; // Token embedding weights
    paramSizes[1] = maxSeqLength * hiddenSize; // Positional embedding weights
    paramSizes[2] = numLayers * hiddenSize; // Layer norm weights 1
    paramSizes[3] = numLayers * hiddenSize; // Layer norm biases 1
    paramSizes[4] = numLayers * hiddenSize * hiddenSize; // Attention query weights
    paramSizes[5] = numLayers * hiddenSize * hiddenSize; // Attention key weights
    paramSizes[6] = numLayers * hiddenSize * hiddenSize; // Attention value weights
    paramSizes[7] = numLayers * hiddenSize; // Attention query biases
    paramSizes[8] = numLayers * hiddenSize; // Attention key biases
    paramSizes[9] = numLayers * hiddenSize; // Attention value biases
    paramSizes[10] = numLayers * hiddenSize * hiddenSize; // Attention output weights
    paramSizes[11] = numLayers * hiddenSize; // Attention output biases
    paramSizes[12] = numLayers * hiddenSize; // Layer norm weights 2
    paramSizes[13] = numLayers * hiddenSize; // Layer norm biases 2
    paramSizes[14] = numLayers * hiddenSize * (4 * hiddenSize); // Feedforward weights
    paramSizes[15] = numLayers * (4 * hiddenSize); // Feedforward biases
    paramSizes[16] = numLayers * (4 * hiddenSize) * hiddenSize; // Projection weights
    paramSizes[17] = numLayers * hiddenSize; // Projection biases
    paramSizes[18] = numLayers * hiddenSize; // Final layer norm weights
    paramSizes[19] = numLayers * hiddenSize; // Final layer norm biases

    // Calculate the total number of parameters by summing the individual parameter sizes
    size_t totalSize = 0;
    for (int i = 0; i < NUMBER_OF_PARAMETERS; ++i) {
        totalSize += paramSizes[i];
    }
    totalParameterSize = totalSize;
}

// Function to allocate memory for model parameters and initialize pointers to them
void allocate_param_pointers(GPT3Model* model) {
    float* memoryPointer;
    // Allocate GPU memory for all model parameters
    cudaMalloc(&memoryPointer, model->allParametersSize * sizeof(float));
    
    // Pointer to the model's parameter structure
    ModelParameters* params = &model->modelParameters; 
    // Array of pointers to model parameters
    float** paramPointers[] = {
        &params->tokenEmbeddingWeights, &params->positionalEmbeddingWeights,
        &params->layerNorm1Weights, &params->layerNorm1Biases,
        &params->attentionQueryWeights, &params->attentionKeyWeights, &params->attentionValueWeights,
        &params->attentionQueryBiases, &params->attentionKeyBiases, &params->attentionValueBiases,
        &params->attentionOutputWeights, &params->attentionOutputBiases,
        &params->layerNorm2Weights, &params->layerNorm2Biases, &params->feedForwardWeights, &params->feedForwardBiases,
        &params->projectionWeights, &params->projectionBiases, &params->finalLayerNormWeights, &params->finalLayerNormBiases
    };
    
    // Initialize pointers to point to their corresponding memory locations
    float* ptr = memoryPointer;
    for (int i = 0; i < NUMBER_OF_PARAMETERS; ++i) {
        paramPointers[i] = &ptr;
        ptr += model->paramSizes[i]; // Move the pointer by the size of the parameter
    }
}

__global__ void initialize_norm_kernel(float* weights, float mean, float stddev, float norm, int N, curandState* state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &state[idx]);
        weights[idx] = (curand_normal(&state[idx]) * stddev + mean) * norm;
    }
}

// Helper function to initialize weights using a normal distribution
void initialize_norm(float* weights, float mean, float stddev, int N, float norm = 1.0f){
    int thread_size = 32 * 32;
    curandState *d_states;
    CUDA_CHECK(cudaMalloc(&d_states, N * sizeof(curandState)));
    initialize_norm_kernel<<<CEIL_DIV(N, thread_size), thread_size>>>(weights, mean, stddev, norm, N, d_states, time(NULL));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_states));
}

// Function to initialize model weights
void initialize_weight(GPT3Model* model){
    ModelParameters* params = &model->modelParameters;
    size_t* paramSizes = model->paramSizes;
    int numLayers = model->hyperparameters.numLayers;
    float residualNormFactor = 1.0f / sqrt(2.0f * numLayers); // Normalization factor for residual connections
    printf("1\n%zu\n",paramSizes[0]);
    // Initialize embedding weights
    initialize_norm(params->tokenEmbeddingWeights, 0.0f, 0.02f, paramSizes[0]);
    printf("2\n");
    initialize_norm(params->positionalEmbeddingWeights, 0.0f, 0.01f, paramSizes[1]);
    
    // Initialize layer normalization weights and biases to constant values
    cudaMemset(params->layerNorm1Weights, 0x3f800000, paramSizes[2]); // Layer norm 1 weights (1.0 in IEEE 754 format)
    cudaMemset(params->layerNorm1Biases, 0, paramSizes[3]); // Layer norm 1 biases (zero)
    
    // Initialize attention weights and biases
    initialize_norm(params->attentionQueryWeights, 0.0f, 1.0f, paramSizes[4]);
    initialize_norm(params->attentionKeyWeights, 0.0f, 1.0f, paramSizes[5]);
    initialize_norm(params->attentionValueWeights, 0.0f, 1.0f, paramSizes[6]);
    // Initialize attention biases to zero
    cudaMemset(params->attentionQueryBiases, 0, paramSizes[7]);
    cudaMemset(params->attentionKeyBiases, 0, paramSizes[8]);
    cudaMemset(params->attentionValueBiases, 0, paramSizes[9]);
    
    // Initialize attention output weights with residual norm scaling
    initialize_norm(params->attentionOutputWeights, 0.0f, 1.0f, paramSizes[10], residualNormFactor);
    cudaMemset(params->attentionOutputBiases, 0, paramSizes[11]); // Attention output biases
    
    // Initialize second layer norm weights and biases
    cudaMemset(params->layerNorm2Weights, 0x3f800000, paramSizes[12]);
    cudaMemset(params->layerNorm2Biases, 0, paramSizes[13]);
    
    // Initialize feedforward network weights and biases
    initialize_norm(params->feedForwardWeights, 0.0f, 1.0f, paramSizes[14]);
    cudaMemset(params->feedForwardBiases, 0, paramSizes[15]);
    
    // Initialize projection weights and biases with residual norm scaling
    initialize_norm(params->projectionWeights, 0.0f, 1.0f, paramSizes[16], residualNormFactor);
    cudaMemset(params->projectionBiases, 0, paramSizes[17]);
    
    // Initialize final layer normalization weights and biases
    cudaMemset(params->finalLayerNormWeights, 0x3f800000, paramSizes[18]);
    cudaMemset(params->finalLayerNormBiases, 0, paramSizes[19]);
}

void initialModel(GPT3Model* model) {
    fill_parameter_sizes(model->paramSizes, model->allParametersSize, model->hyperparameters);
    allocate_param_pointers(model);
    initialize_weight(model);
}

void forward(GPT3Model* model, int* input, int* ouput){

}

int main(int argc, char *argv[]) {
    GPT3Model model;
    model.hyperparameters.numLayers = 6;
    model.hyperparameters.hiddenChannels = 2048;
    model.hyperparameters.numAttentionHeads = 24;
    model.hyperparameters.maxSequenceLength = 2048;
    model.hyperparameters.vocabularySize = 50304;
    for (int i = 1; i < argc; i+=2) {
        if (argv[i][0] == '-') {EXIT_FAILURE;}
        if (argv[i][1] == 'l') {model.hyperparameters.numLayers = atoi(argv[i+1]);}
        else if (argv[i][1] == 'd') {model.hyperparameters.hiddenChannels = atoi(argv[i+1]);}
        else if (argv[i][1] == 'h') {model.hyperparameters.numAttentionHeads = atoi(argv[i+1]);}
        else if (argv[i][1] == 's') {model.hyperparameters.maxSequenceLength = atoi(argv[i+1]);}
    // Just printing the arguments passed to the program
    }

    initialModel(&model);
    // forward();
    return 0;
}