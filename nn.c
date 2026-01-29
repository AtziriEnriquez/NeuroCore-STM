#include "nn.h"

#include <stdint.h>
#include <stdio.h>

#include <math.h>

// Neural Network architecture
#define NN_LAYER_0_NEURONS (2)
#define NN_LAYER_1_NEURONS (1)

// First (hidden) layer, 2 inputs, 2 neurons
const float l0_weights[NN_INPUTS][NN_LAYER_0_NEURONS] = {
    {  3.0583365f, -2.789102f },
    { -3.114127f,  2.696875f }
};
const float l0_biases[NN_LAYER_0_NEURONS] = {
    -0.8766896,
    -0.7031553
};
static float l0_neuron_values[NN_LAYER_0_NEURONS]; 

// Second (output) layer, 2 inputes, 1 neuron
const float l1_weights[NN_LAYER_0_NEURONS][NN_LAYER_1_NEURONS] = {
    {3.396049},
    {3.614842}
};
const float l1_biases[NN_LAYER_1_NEURONS] = {
    0.3589306
};
static float l1_neuron_values[NN_LAYER_1_NEURONS]; 

// Quantized NN Model Parameters
const int8_t l0_qweights[2][2] = {
    { 49, -45, },
    { -50, 43, },
};

const int8_t l0_qbiases[2] = { -14, -11, };

const int8_t l1_qweights[2][1] = {
    { 54, },
    { 58, },
};

const int8_t l1_qbiases[1] = { 6, };

// Quantized lookup table for qelu activation function
// Look up quantized result of input argument by casting the
// int8_t (qm.n) to uint8_t for the lookup table index, e.g.:
// int8_t qresult = qelu_lut[(uint8_t)qargument];

static const int8_t qelu_lut[256] = {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87,
    88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127,
    -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -15, -15, -15, -15, -15, -15, -15,
    -15, -15, -15, -15, -15, -15, -15, -15,
    -15, -15, -15, -14, -14, -14, -14, -14,
    -14, -14, -14, -13, -13, -13, -13, -13,
    -12, -12, -12, -12, -11, -11, -11, -10,
    -10, -10, -9, -9, -8, -8, -7, -7,
    -6, -6, -5, -4, -4, -3, -2, -1,
};

// Quantized lookup table for qsigmoid activation function
// Look up quantized result of input argument by casting the
// int8_t (qm.n) to uint8_t for the lookup table index, e.g.:
// int8_t qresult = qsigmoid_lut[(uint8_t)qargument];

static const int8_t qsigmoid_lut[256] = {
    8, 8, 8, 9, 9, 9, 9, 10,
    10, 10, 10, 11, 11, 11, 11, 11,
    12, 12, 12, 12, 12, 13, 13, 13,
    13, 13, 13, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 3, 3,
    3, 3, 3, 3, 4, 4, 4, 4,
    4, 5, 5, 5, 5, 5, 6, 6,
    6, 6, 7, 7, 7, 7, 8, 8,
};




// Implement the ELU activation function
float elu(float x) {
    float y;
    if( x >= 0.0 )
        y = x;
    else 
        y = expf(x)-1.0;
    return y;
}

// Implement the sigmoid activation function
float sigmoid(float x) {
    return 1.0 / (1 + expf(-x));
}

// run the NN, return output value
static void NN_compute_layer(
    const int neurons,              // Number of neurons in the layer
    float *neuron_values,           // storage for resulting neuron values after update
    const int inputs,               // Number of inputs to the layer (assumes dense NN)
    const float input_values[],     // Input values
    const float biases[],           // NN parameters: biases
    const float weights[][neurons],  // NN parameters: weights
    float(*af)(float)) {            // activation function

    int j, k; // iterators
    float accumulator;
    for( j = 0; j < neurons; j++ ) {
        accumulator = biases[j]; // initailize with bias
        // iterate over each input attached to the neuron,
        // scaling it and accumulating the result
        for( k = 0; k < inputs; k++ ) {
            float weight = weights[k][j];
            float input = input_values[k];
            accumulator += weight * input;
        }
        // call the activation function
        neuron_values[j] = af(accumulator);
    }  
}

float NN_predict(const float *input_values) {
    // Process the first (hidden) layer
    NN_compute_layer(NN_LAYER_0_NEURONS, l0_neuron_values,
                     NN_INPUTS, input_values,
                    l0_biases, l0_weights, elu);
    // Process the second (output) layer
    NN_compute_layer(NN_LAYER_1_NEURONS, l1_neuron_values,
                     NN_LAYER_0_NEURONS, l0_neuron_values,
                     l1_biases, l1_weights, sigmoid);
    return l1_neuron_values[0];
} 

//
// Quantized Model
//

// TODO: if you use your own implementation, replace my analysis with your own!
//
// Quantize the floating-point based NN architecture to work with int8_t,
// which is an 8-bit signed value ranging from -128 to 127.  Characterize the 
// maximum/minium value through your NN.  
//
// Hidden Layer (input range 0 to 1.1):
// 
// Both scale factors for the first neuron are ~ -3 so my accumulated sum of scaled
// inputs, worst case, is about -6.  But the bias is ~ +1.5 so
// this neuron's value into the activation function will range from about -4.5 to about 1.5
// its activation function is elu, which clamps negative values, 
// so after elu() this neuron's value ranges ~-1 to 1.5
//
// Both scale factors for the second neuron are ~2 so my accumulated sum of scaled
// inputs, worst case, is about 4, but the bias is ~ -2 so
// this neuron's value into the activation function will range from about -2 to about 2
// its activation function is elu, so again the neuron's final value ranges from ~-1 to ~2
//
// So, back of napkin, the worst-case intermediate value is ~-6 and the final neuron 
// values, which become inputs to the next layer, are within ~-1 to ~2
// 
// Output Layer (input range approx -1 to 2)
// 
// The scale factors are both negative, around -4 and -3 respectively.
// My worst-case accumulated sum of scaled inputs is -6, and the bias is 
// approximately zero, so this neuron's largest value into the activation
// function is ~-6.  
// its activation function is sigmoid, which ranges from 0 to 1
//
// So the worst, case value in my system has a magnitude of 6, which I can encode in 3 bits.
//
// This suggests I can get away with Q3.4 fixed-point representation, but I don't want any 
// trouble while I'm debugging so I'm going to hedge my bet and start out with Q4.3


// TODO: Implement this function using fixed-point numbers/arithmetic - no floating point!
// TODO: Comment the function as if I were sitting next to you and you are walking me through
// your implementation - what are your intentions, how do you implement them, what design 
// decisions did you think about, and then make, and why?
int8_t NN_qpredict(const int8_t *input_qvalues)
{
    // Big picture:
    // We are doing the same neural network math as NN_predict(), but we will
    // ONLY use integers (fixed-point), not floats
    //
    // Our fixed-point format is Q4.3:
    // - everything is scaled by 2^4 = 16
    // - so "real 1.0" is stored as 16, "real 0.5" is stored as 8, etc.
    //
    // The network is:
    //   Layer 0 (hidden): 2 inputs -> 2 neurons, activation = ELU
    //   Layer 1 (output): 2 inputs -> 1 neuron, activation = sigmoid
    //
    // We cannot compute ELU/sigmoid with expf(), so we use LUTs:
    //   qelu_lut[...] and qsigmoid_lut[...]

    // We'll store hidden layer outputs here (2 neurons)
    int8_t l0_qvalues[NN_LAYER_0_NEURONS];

    // LAYER 0 (HIDDEN): 2 inputs -> 2 neurons with ELU activation
    for (int j = 0; j < NN_LAYER_0_NEURONS; j++) {

        // qaccumulator will hold the running sum for neuron j
        // It is still "Q?.4" format, but we store it in int32_t so we don't overflow
        // Start with the bias (bias is already Q3.4)
        int32_t qaccumulator = (int32_t)l0_qbiases[j];  // Q3.4 stored in a bigger box

        // Now add each weight*input term
        // weights and inputs are Q3.4 (int8_t)
        // When we multiply Q3.4 * Q3.4, we get Q6.8 (needs bigger type)
        // Then we shift right by 4 to bring it back to Q6.4
        // After that, we can safely add to our accumulator (same fractional bits = 4)
        for (int k = 0; k < NN_INPUTS; k++) {

            // Step 1: cast int8 -> int16 so multiplication doesn't overflow
            int16_t qweight = (int16_t)l0_qweights[k][j];      // Q3.4 in int16 container
            int16_t qinput  = (int16_t)input_qvalues[k];       // Q3.4 in int16 container

            // Step 2: multiply -> Q6.8
            int16_t qmult = (int16_t)(qweight * qinput);       // Q6.8

            // Step 3: shift right by fractional bits to align back to ".4"
            // Q6.8 -> Q6.4 (because we removed 4 fractional bits)
            qmult = (int16_t)(qmult >> QNN_FRACTIONAL_BITS);   // Q6.4

            // Step 4: add to accumulator (both have 4 fractional bits now)
            qaccumulator += (int32_t)qmult;                    // still Q?.4
        }

        // At this point, qaccumulator is the neuron's "pre-activation" value in Q?.4
        // Our LUT expects an int8_t input value (Q3.4 range)
        // We will clamp to [-128, 127] so we don't wrap around
        if (qaccumulator > 127)  qaccumulator = 127;
        if (qaccumulator < -128) qaccumulator = -128;

        int8_t qargument = (int8_t)qaccumulator;  // Q3.4 (fits in int8 now)

        // Apply ELU activation using the lookup table
        // IMPORTANT:
        // We MUST cast qargument to uint8_t for the LUT index
        // This is because negative int8 values should map to indices 128..255
        // Example: qargument = -1 -> (uint8_t)255
        int8_t qresult = qelu_lut[(uint8_t)qargument];

        // Store hidden neuron output (still Q3.4)
        l0_qvalues[j] = qresult;
    }

    // LAYER 1 (OUTPUT): 2 inputs -> 1 neuron with sigmoid activation

    // Start output neuron accumulator at its bias (Q3.4)
    int32_t qaccumulator = (int32_t)l1_qbiases[0];

    // Add weight*input terms where "inputs" are the hidden layer outputs
    for (int k = 0; k < NN_LAYER_0_NEURONS; k++) {

        int16_t qweight = (int16_t)l1_qweights[k][0];     // Q3.4
        int16_t qinput  = (int16_t)l0_qvalues[k];         // Q3.4

        int16_t qmult = (int16_t)(qweight * qinput);      // Q6.8
        qmult = (int16_t)(qmult >> QNN_FRACTIONAL_BITS);  // Q6.4

        qaccumulator += (int32_t)qmult;                   // Q?.4
    }

    // Clamp output accumulator before LUT (so it fits in int8_t)
    if (qaccumulator > 127)  qaccumulator = 127;
    if (qaccumulator < -128) qaccumulator = -128;

    int8_t qargument = (int8_t)qaccumulator;

    // Result is Q3.4, so output is roughly 0..16
    int8_t qresult = qsigmoid_lut[(uint8_t)qargument];

    // Return quantized output. Caller can convert to float with NN_dequantize()
    return qresult;
}

// Convert a floating point value to its fixed-point equivalent representation
int8_t NN_quantize(const float value) {
    return (int8_t)(roundf(value*QNN_SCALE_FACTOR));
}

// Convert a fixed-point value to its floating point equivalent representation
float NN_dequantize(const int8_t value) {
    return (float)(value) / QNN_SCALE_FACTOR;
}

// Takes an activation function pointer and desired (output) lookup table name
// Iterates over all possible fixed-point inputs to the activation function and
// generates a lookup table with pre-computed results stored in fixed-point
// form.
void NN_generate_af_lut(float(*af)(float), char *name) {
    int8_t qinput, qoutput; // quantized activation function inputs/outputs
    float input, output;    // floating point activation function inputs/outputs

    // Generate output to console - usage comment
    printf("// Quantized lookup table for %s activation function\n", name);
    printf("// Look up quantized result of input argument by casting the\n");
    printf("// int8_t (qm.n) to uint8_t for the lookup table index, e.g.:\n");
    printf("// int8_t qresult = %s_lut[(uint8_t)qargument];\n", name);
    // Generate lookup table
    printf("\nstatic const int8_t %s_lut[256] = {\n\t", name);
    for( uint16_t index = 0; index < 256; index++ ) {
        // Convert actual array index into the int8_t range by interpreting
        // uint8_t type as int8_t, thus array index will map the lookup table
        // to holding entries in the following int8_t order
        // 0 -> 255 to 0 -> 127, -128 -> -1
        qinput = (int8_t)index; 
        // synthesize equivalent floating point value
        input = (float)qinput / QNN_SCALE_FACTOR;
        // call the activation function to compute a floating point result
        output = af(input);
        // quantize the result
        qoutput = NN_quantize(output);
        // generate table entry
        printf("%d, ", qoutput);
        if( ((index+1) % 8) == 0) {
            printf("\n\t");
        }
    }
    printf("};\n\n");
}

// Wrapper function to generate lookup tables for elu() and sigmoid() activation functions
// resulting lookup tables are printed to standard out in NN_generate_af_lut()
void NN_generate_qaf_luts(void) {
    NN_generate_af_lut(elu, "qelu");
    NN_generate_af_lut(sigmoid, "qsigmoid");
}

// Output 1 row of quantized data for a given input array (of float[] type) and length
void NN_quantize_array_1d(const uint8_t dim, const float *values) {
    for( uint8_t k = 0; k < dim; k++ ) {
        printf("%d, ", NN_quantize(values[k]));
    }
}

// Output 'x_dim' rows of quantized values for a given input array (of float[][] type) and dimensions
void NN_quantize_array_2d(const uint8_t x_dim, const uint8_t y_dim, const float values[][y_dim] ) {
    for( uint8_t x = 0; x < x_dim; x++ ) {
        printf("\t{ ");
        NN_quantize_array_1d(y_dim, values[x]);
        printf("},\n");
    }
}

// Wrapper function to generate quantized equivalents of the floating-point based 
// NN trained parameters
void NN_quantize_parameters(void) {
    // Banner
    printf("\n// Quantized NN Model Parameters\n");

    // L0 Weights
    printf("const int8_t l0_qweights[%d][%d] = {\n", NN_INPUTS, NN_LAYER_0_NEURONS);
    NN_quantize_array_2d(NN_INPUTS, NN_LAYER_0_NEURONS, l0_weights);
    printf("};\n\n");

    // L0 Biases
    printf("const int8_t l0_qbiases[%d] = { ", NN_LAYER_0_NEURONS);
    NN_quantize_array_1d(NN_LAYER_0_NEURONS, l0_biases);
    printf("};\n\n");

    // L1 Weights
    printf("const int8_t l1_qweights[%d][%d] = {\n", NN_LAYER_0_NEURONS, NN_LAYER_1_NEURONS);
    NN_quantize_array_2d(NN_LAYER_0_NEURONS, NN_LAYER_1_NEURONS, l1_weights);
    printf("};\n\n");

    // L1 Biases
    printf("const int8_t l1_qbiases[%d] = { ", NN_LAYER_1_NEURONS);
    NN_quantize_array_1d(NN_LAYER_1_NEURONS, l1_biases);
    printf("};\n\n");
}