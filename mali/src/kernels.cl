#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void vector_add(__global const float *a, __global const float *b, __global float *c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
}

// W is row-major [out_features, in_features] in the original FP16 checkpoint.
__kernel void linear_h(__global const float *x, __global const half *w,
                       __global const half *bias, __global float *y,
                       int rows, int in_features, int out_features, int has_bias) {
    int index = get_global_id(0);
    if (index >= rows * out_features) return;
    int row = index / out_features;
    int col = index - row * out_features;
    __global const float *a = x + row * in_features;
    __global const half *b = w + col * in_features;
    float acc = has_bias ? convert_float(bias[col]) : 0.0f;
    for (int k = 0; k < in_features; ++k) acc = fma(a[k], convert_float(b[k]), acc);
    y[index] = acc;
}

__kernel void linear_h_vec8(__global const float *x, __global const half *w,
                            __global const half *bias, __global float *y,
                            int rows, int in_features, int out_features, int has_bias) {
    int index = get_global_id(0);
    if (index >= rows * out_features) return;
    int row = index / out_features;
    int col = index - row * out_features;
    __global const float *a = x + row * in_features;
    __global const half *b = w + col * in_features;
    float8 partial = (float8)(0.0f);
    int blocks = in_features / 8;
    for (int block = 0; block < blocks; ++block) {
        float8 av = vload8(block, a);
        float8 bv = convert_float8(vload8(block, b));
        partial = fma(av, bv, partial);
    }
    float sum = partial.s0 + partial.s1 + partial.s2 + partial.s3
              + partial.s4 + partial.s5 + partial.s6 + partial.s7;
    for (int k = blocks * 8; k < in_features; ++k)
        sum = fma(a[k], convert_float(b[k]), sum);
    y[index] = sum + (has_bias ? convert_float(bias[col]) : 0.0f);
}

// Reuse each weight vector across four input rows without local-memory barriers.
__kernel void linear_h_rows4_vec8(__global const float *x, __global const half *w,
                                  __global const half *bias, __global float *y,
                                  int rows, int in_features, int out_features, int has_bias) {
    int index = get_global_id(0);
    int groups = (rows + 3) / 4;
    if (index >= groups * out_features) return;
    int row0 = (index / out_features) * 4;
    int col = index % out_features;
    __global const half *b = w + col * in_features;
    float8 p0 = (float8)(0.0f);
    float8 p1 = (float8)(0.0f);
    float8 p2 = (float8)(0.0f);
    float8 p3 = (float8)(0.0f);
    int blocks = in_features / 8;
    for (int block = 0; block < blocks; ++block) {
        float8 bv = convert_float8(vload8(block, b));
        p0 = fma(vload8(block, x + row0 * in_features), bv, p0);
        if (row0 + 1 < rows)
            p1 = fma(vload8(block, x + (row0 + 1) * in_features), bv, p1);
        if (row0 + 2 < rows)
            p2 = fma(vload8(block, x + (row0 + 2) * in_features), bv, p2);
        if (row0 + 3 < rows)
            p3 = fma(vload8(block, x + (row0 + 3) * in_features), bv, p3);
    }
    float sums[4] = {
        p0.s0 + p0.s1 + p0.s2 + p0.s3 + p0.s4 + p0.s5 + p0.s6 + p0.s7,
        p1.s0 + p1.s1 + p1.s2 + p1.s3 + p1.s4 + p1.s5 + p1.s6 + p1.s7,
        p2.s0 + p2.s1 + p2.s2 + p2.s3 + p2.s4 + p2.s5 + p2.s6 + p2.s7,
        p3.s0 + p3.s1 + p3.s2 + p3.s3 + p3.s4 + p3.s5 + p3.s6 + p3.s7
    };
    for (int k = blocks * 8; k < in_features; ++k) {
        float bv = convert_float(b[k]);
        for (int r = 0; r < 4 && row0 + r < rows; ++r)
            sums[r] = fma(x[(row0 + r) * in_features + k], bv, sums[r]);
    }
    float offset = has_bias ? convert_float(bias[col]) : 0.0f;
    for (int r = 0; r < 4 && row0 + r < rows; ++r)
        y[(row0 + r) * out_features + col] = sums[r] + offset;
}

// W is transposed to [in_features, out_features] so neighboring work-items
// fetch neighboring output columns on each reduction step.
__kernel void linear_h_transposed_rows4_cols4(__global const float *x,
                                               __global const half *w,
                                               __global const half *bias,
                                               __global float *y,
                                               int rows, int in_features,
                                               int out_features, int has_bias) {
    int groups = (rows + 3) / 4;
    int column_groups = out_features / 4;
    int index = get_global_id(0);
    if (index >= groups * column_groups) return;
    int row0 = (index / column_groups) * 4;
    int col0 = (index % column_groups) * 4;
    float4 a0=(float4)(0.0f), a1=(float4)(0.0f);
    float4 a2=(float4)(0.0f), a3=(float4)(0.0f);
    for (int k = 0; k < in_features; ++k) {
        float4 weights = convert_float4(vload4(0, w + k * out_features + col0));
        a0 = fma(x[row0 * in_features + k], weights, a0);
        if (row0 + 1 < rows) a1 = fma(x[(row0 + 1) * in_features + k], weights, a1);
        if (row0 + 2 < rows) a2 = fma(x[(row0 + 2) * in_features + k], weights, a2);
        if (row0 + 3 < rows) a3 = fma(x[(row0 + 3) * in_features + k], weights, a3);
    }
    float4 offset = has_bias ? convert_float4(vload4(0, bias + col0)) : (float4)(0.0f);
    vstore4(a0 + offset, 0, y + row0 * out_features + col0);
    if (row0 + 1 < rows) vstore4(a1 + offset, 0, y + (row0 + 1) * out_features + col0);
    if (row0 + 2 < rows) vstore4(a2 + offset, 0, y + (row0 + 2) * out_features + col0);
    if (row0 + 3 < rows) vstore4(a3 + offset, 0, y + (row0 + 3) * out_features + col0);
}

__kernel void linear_h_transposed_rows4_cols8(__global const float *x,
                                               __global const half *w,
                                               __global const half *bias,
                                               __global float *y,
                                               int rows, int in_features,
                                               int out_features, int has_bias) {
    int groups = (rows + 3) / 4;
    int column_groups = out_features / 8;
    int index = get_global_id(0);
    if (index >= groups * column_groups) return;
    int row0 = (index / column_groups) * 4;
    int col0 = (index % column_groups) * 8;
    float8 a0=(float8)(0.0f), a1=(float8)(0.0f);
    float8 a2=(float8)(0.0f), a3=(float8)(0.0f);
    for (int k = 0; k < in_features; ++k) {
        float8 weights = convert_float8(vload8(0, w + k * out_features + col0));
        a0 = fma(x[row0 * in_features + k], weights, a0);
        if (row0 + 1 < rows) a1 = fma(x[(row0 + 1) * in_features + k], weights, a1);
        if (row0 + 2 < rows) a2 = fma(x[(row0 + 2) * in_features + k], weights, a2);
        if (row0 + 3 < rows) a3 = fma(x[(row0 + 3) * in_features + k], weights, a3);
    }
    float8 offset = has_bias ? convert_float8(vload8(0, bias + col0)) : (float8)(0.0f);
    vstore8(a0 + offset, 0, y + row0 * out_features + col0);
    if (row0 + 1 < rows) vstore8(a1 + offset, 0, y + (row0 + 1) * out_features + col0);
    if (row0 + 2 < rows) vstore8(a2 + offset, 0, y + (row0 + 2) * out_features + col0);
    if (row0 + 3 < rows) vstore8(a3 + offset, 0, y + (row0 + 3) * out_features + col0);
}

__kernel void embedding_norm(__global const int *token_ids, __global const half *emb,
                             __global const half *gamma, __global float *out,
                             int rows, int dim) {
    int row = get_global_id(0);
    if (row >= rows) return;
    __global const half *source = emb + token_ids[row] * dim;
    float sum = 0.0f, sumsq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        float x = convert_float(source[j]);
        sum += x;
        sumsq = fma(x, x, sumsq);
    }
    float mean = sum / dim;
    float inv = rsqrt(fmax(sumsq / dim - mean * mean, 0.0f) + 1.0e-5f);
    for (int j = 0; j < dim; ++j)
        out[row * dim + j] = (convert_float(source[j]) - mean) * inv * convert_float(gamma[j]);
}

__kernel void norm_rows(__global const float *input, __global const half *gamma,
                        __global const half *bias, __global float *out,
                        int rows, int dim, int has_bias) {
    int row = get_global_id(0);
    if (row >= rows) return;
    __global const float *source = input + row * dim;
    float sum = 0.0f, sumsq = 0.0f;
    for (int j = 0; j < dim; ++j) {
        float x = source[j];
        sum += x;
        sumsq = fma(x, x, sumsq);
    }
    float mean = sum / dim;
    float inv = rsqrt(fmax(sumsq / dim - mean * mean, 0.0f) + 1.0e-5f);
    for (int j = 0; j < dim; ++j) {
        float value = (source[j] - mean) * inv * convert_float(gamma[j]);
        out[row * dim + j] = has_bias ? value + convert_float(bias[j]) : value;
    }
}

__kernel void norm_rows_vec8(__global const float *input, __global const half *gamma,
                             __global const half *bias, __global float *out,
                             int rows, int dim, int has_bias) {
    int row = get_global_id(0);
    if (row >= rows) return;
    __global const float *source = input + row * dim;
    float8 sum8 = (float8)(0.0f);
    float8 square8 = (float8)(0.0f);
    int blocks = dim / 8;
    for (int block = 0; block < blocks; ++block) {
        float8 values = vload8(block, source);
        sum8 += values;
        square8 = fma(values, values, square8);
    }
    float sum = sum8.s0+sum8.s1+sum8.s2+sum8.s3
              + sum8.s4+sum8.s5+sum8.s6+sum8.s7;
    float sumsq = square8.s0+square8.s1+square8.s2+square8.s3
                + square8.s4+square8.s5+square8.s6+square8.s7;
    for (int j = blocks * 8; j < dim; ++j) {
        float value = source[j];
        sum += value;
        sumsq = fma(value, value, sumsq);
    }
    float mean = sum / dim;
    float inv = rsqrt(fmax(sumsq / dim - mean * mean, 0.0f) + 1.0e-5f);
    for (int block = 0; block < blocks; ++block) {
        float8 value = (vload8(block, source) - mean) * inv
                     * convert_float8(vload8(block, gamma));
        if (has_bias) value += convert_float8(vload8(block, bias));
        vstore8(value, block, out + row * dim);
    }
    for (int j = blocks * 8; j < dim; ++j) {
        float value = (source[j] - mean) * inv * convert_float(gamma[j]);
        out[row * dim + j] = has_bias ? value + convert_float(bias[j]) : value;
    }
}

__kernel void add_type(__global const float *input, __global const half *embedding,
                       __global float *out, int n, int type_id, int dim) {
    int i = get_global_id(0);
    if (i < n) out[i] = input[i] + convert_float(embedding[type_id * dim + i % dim]);
}

__kernel void rotary_qkv(__global const float *input, __global float *out,
                         int seq, int dim, int heads, float theta) {
    int i = get_global_id(0);
    if (i >= seq * 3 * dim) return;
    int position = i / (3 * dim);
    int within = i % (3 * dim);
    int component = within / dim;
    int head_dim = dim / heads;
    int head_position = within % head_dim;
    if (component == 2) { out[i] = input[i]; return; }
    int half_width = head_dim / 2;
    int half_dim = head_position % half_width;
    float exponent = -(float)(half_dim * 2) / (float)head_dim;
    float angle = (float)position * pow(theta, exponent);
    int partner = head_position < half_width ? i + half_width : i - half_width;
    float rotated = head_position < half_width ? -input[partner] : input[partner];
    out[i] = fma(input[i], cos(angle), rotated * sin(angle));
}

// qkv layout: [sequence, component(3), head, head_dim].
__kernel void attention_scores(__global const float *qkv, __global float *scores,
                               int seq, int dim, int heads, int half_window, int valid_len) {
    int i = get_global_id(0);
    if (i >= seq * heads * seq) return;
    int key = i % seq;
    int head = (i / seq) % heads;
    int query = i / (seq * heads);
    if (key >= valid_len || (half_window > 0 && abs(query - key) > half_window)) {
        scores[i] = -1.0e9f;
        return;
    }
    int head_dim = dim / heads;
    int qbase = query * 3 * dim + head * head_dim;
    int kbase = key * 3 * dim + dim + head * head_dim;
    float dot = 0.0f;
    for (int j = 0; j < head_dim; ++j) dot = fma(qkv[qbase + j], qkv[kbase + j], dot);
    scores[i] = dot * rsqrt((float)head_dim);
}

__kernel void attention_scores_vec8(__global const float *qkv, __global float *scores,
                                    int seq, int dim, int heads,
                                    int half_window, int valid_len) {
    int i = get_global_id(0);
    if (i >= seq * heads * seq) return;
    int key = i % seq;
    int head = (i / seq) % heads;
    int query = i / (seq * heads);
    if (key >= valid_len || (half_window > 0 && abs(query - key) > half_window)) {
        scores[i] = -1.0e9f;
        return;
    }
    int head_dim = dim / heads;
    int qbase = query * 3 * dim + head * head_dim;
    int kbase = key * 3 * dim + dim + head * head_dim;
    float8 partial = (float8)(0.0f);
    int blocks = head_dim / 8;
    for (int block = 0; block < blocks; ++block) {
        float8 q = vload8(block, qkv + qbase);
        float8 k = vload8(block, qkv + kbase);
        partial = fma(q, k, partial);
    }
    float dot = partial.s0+partial.s1+partial.s2+partial.s3
              + partial.s4+partial.s5+partial.s6+partial.s7;
    for (int feature = blocks * 8; feature < head_dim; ++feature)
        dot = fma(qkv[qbase + feature], qkv[kbase + feature], dot);
    scores[i] = dot * rsqrt((float)head_dim);
}

__kernel void softmax_rows(__global const float *scores, __global float *probabilities,
                           int rows, int cols) {
    int row = get_global_id(0);
    if (row >= rows) return;
    int offset = row * cols;
    float maximum = -INFINITY;
    for (int j = 0; j < cols; ++j) maximum = fmax(maximum, scores[offset + j]);
    float total = 0.0f;
    for (int j = 0; j < cols; ++j) {
        float value = exp(scores[offset + j] - maximum);
        probabilities[offset + j] = value;
        total += value;
    }
    for (int j = 0; j < cols; ++j) probabilities[offset + j] /= total;
}

__kernel void attention_context(__global const float *qkv, __global const float *probabilities,
                                __global float *context, int seq, int dim, int heads) {
    int i = get_global_id(0);
    if (i >= seq * dim) return;
    int query = i / dim;
    int head_dim = dim / heads;
    int head = (i % dim) / head_dim;
    int feature = i % head_dim;
    int pbase = (query * heads + head) * seq;
    float sum = 0.0f;
    for (int key = 0; key < seq; ++key) {
        int vbase = key * 3 * dim + 2 * dim + head * head_dim;
        sum = fma(probabilities[pbase + key], qkv[vbase + feature], sum);
    }
    context[i] = sum;
}

__kernel void swiglu(__global const float *input, __global float *output,
                      int rows, int width) {
    int i = get_global_id(0);
    if (i >= rows * width) return;
    int row = i / width;
    int col = i % width;
    float x = input[row * width * 2 + col];
    float gate = input[row * width * 2 + width + col];
    float gelu = 0.5f * x * (1.0f + erf(x * 0.7071067811865475f));
    output[i] = gelu * gate;
}

__kernel void relu(__global const float *input, __global float *output, int n) {
    int i = get_global_id(0);
    if (i < n) output[i] = fmax(input[i], 0.0f);
}
