#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

enum class test_result {
    pass,
    skip,
    fail,
};

static ggml_context_ptr make_context() {
    ggml_init_params params = {
        8 * 1024 * 1024,
        nullptr,
        true,
    };
    return ggml_context_ptr(ggml_init(params));
}

static bool is_cuda_device(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg && std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0;
}

static bool test_mxfp8_scale_reference() {
    if (ggml_ue8m0_to_fp32(0x7e) != 0.5f || ggml_ue8m0_to_fp32(0x7f) != 1.0f ||
            ggml_ue8m0_to_fp32(0x80) != 2.0f || ggml_ue8m0_to_fp32(0x00) != std::ldexp(1.0f, -127) ||
            !std::isnan(ggml_ue8m0_to_fp32(0xff))) {
        std::fprintf(stderr, "unexpected UE8M0 decoding\n");
        return false;
    }
    if (ggml_mxfp8_e4m3_scale(0.0f) != 0 || ggml_mxfp8_e4m3_scale(-1.0f) != 0 ||
            ggml_mxfp8_e4m3_scale(std::numeric_limits<float>::infinity()) != 0xfe ||
            ggml_mxfp8_e4m3_scale(std::numeric_limits<float>::denorm_min()) != 0x00 ||
            ggml_mxfp8_e4m3_scale(std::numeric_limits<float>::max()) != 0xf7) {
        std::fprintf(stderr, "unexpected MXFP8 scale boundary result\n");
        return false;
    }

    for (const int exponent : { -120, -20, -1, 0, 1, 20, 100 }) {
        const float boundary = std::ldexp(448.0f, exponent);
        const uint8_t expected = static_cast<uint8_t>(exponent + 127);
        if (ggml_mxfp8_e4m3_scale(std::nextafter(boundary, 0.0f)) != expected ||
                ggml_mxfp8_e4m3_scale(boundary) != expected ||
                ggml_mxfp8_e4m3_scale(std::nextafter(boundary, std::numeric_limits<float>::infinity())) != expected + 1) {
            std::fprintf(stderr, "unexpected MXFP8 scale around 448 * 2^%d\n", exponent);
            return false;
        }
    }
    return true;
}

static test_result test_scaled_set_rows(ggml_backend_t backend, int64_t head_dim) {
    constexpr int64_t n_head  = 2;
    constexpr int64_t n_cache = 5;
    constexpr int64_t n_rows  = 2;

    ggml_context_ptr ctx = make_context();
    if (!ctx) {
        std::fprintf(stderr, "failed to create GGML context\n");
        return test_result::fail;
    }

    ggml_tensor * cache   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F8_E4M3, head_dim * n_head, n_cache);
    ggml_tensor * scales  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_head, n_cache);
    ggml_tensor * source  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim * n_head, n_rows);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, n_rows);
    ggml_tensor * write =
        ggml_set_rows_f8_scaled(ctx.get(), cache, scales, source, indices, static_cast<int32_t>(head_dim));

    if (!ggml_backend_supports_op(backend, write)) {
        return test_result::skip;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, write);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate CUDA tensors for %lld-wide scaled SET_ROWS\n", (long long) head_dim);
        return test_result::fail;
    }

    constexpr uint8_t                        sentinel_byte    = 0xa5;
    constexpr float                          sentinel_scale   = -123.0f;
    const std::array<int64_t, n_rows>        row_indices      = { 3, 1 };
    const std::array<float, n_rows * n_head> input_scales     = { 0.25f, 0.0f, 2.0f, 0.5f };
    const std::array<float, 8>               normalized       = { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 448.0f };
    const std::array<uint8_t, 8>             normalized_bytes = { 0x00, 0x38, 0xb8, 0x30, 0xb0, 0x40, 0xc0, 0x7e };

    for (size_t i = 0; i < normalized.size(); ++i) {
        if (ggml_fp32_to_fp8_e4m3(normalized[i]) != normalized_bytes[i]) {
            std::fprintf(stderr, "unexpected E4M3 encoding for exact test value %g\n", normalized[i]);
            return test_result::fail;
        }
    }

    std::vector<float> source_data(ggml_nelements(source));
    for (int64_t row = 0; row < n_rows; ++row) {
        for (int64_t head = 0; head < n_head; ++head) {
            const float input_scale = input_scales[row * n_head + head];
            for (int64_t d = 0; d < head_dim; ++d) {
                source_data[row * head_dim * n_head + head * head_dim + d] =
                    input_scale == 0.0f ? 0.0f : input_scale * normalized[d % normalized.size()];
            }
        }
    }

    std::vector<uint8_t> cache_data(ggml_nbytes(cache), sentinel_byte);
    std::vector<float>   scale_data(ggml_nelements(scales), sentinel_scale);
    ggml_backend_tensor_set(cache, cache_data.data(), 0, cache_data.size());
    ggml_backend_tensor_set(scales, scale_data.data(), 0, scale_data.size() * sizeof(float));
    ggml_backend_tensor_set(source, source_data.data(), 0, source_data.size() * sizeof(float));
    ggml_backend_tensor_set(indices, row_indices.data(), 0, row_indices.size() * sizeof(int64_t));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%lld-wide scaled SET_ROWS graph compute failed\n", (long long) head_dim);
        return test_result::fail;
    }
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(cache, cache_data.data(), 0, cache_data.size());
    ggml_backend_tensor_get(scales, scale_data.data(), 0, scale_data.size() * sizeof(float));

    std::vector<uint8_t> expected_cache(ggml_nbytes(cache), sentinel_byte);
    std::vector<float>   expected_scales(ggml_nelements(scales), sentinel_scale);
    for (int64_t row = 0; row < n_rows; ++row) {
        const int64_t dst_row = row_indices[row];
        for (int64_t head = 0; head < n_head; ++head) {
            const float input_scale                  = input_scales[row * n_head + head];
            expected_scales[dst_row * n_head + head] = input_scale == 0.0f ? 1.0f : input_scale;
            for (int64_t d = 0; d < head_dim; ++d) {
                expected_cache[dst_row * head_dim * n_head + head * head_dim + d] =
                    input_scale == 0.0f ? 0x00 : normalized_bytes[d % normalized_bytes.size()];
            }
        }
    }

    if (cache_data != expected_cache) {
        for (size_t i = 0; i < cache_data.size(); ++i) {
            if (cache_data[i] != expected_cache[i]) {
                std::fprintf(stderr, "%lld-wide scaled SET_ROWS byte mismatch at %zu: got 0x%02x, expected 0x%02x\n",
                             (long long) head_dim, i, cache_data[i], expected_cache[i]);
                break;
            }
        }
        return test_result::fail;
    }
    if (scale_data != expected_scales) {
        for (size_t i = 0; i < scale_data.size(); ++i) {
            if (scale_data[i] != expected_scales[i]) {
                std::fprintf(stderr, "%lld-wide scaled SET_ROWS scale mismatch at %zu: got %g, expected %g\n",
                             (long long) head_dim, i, scale_data[i], expected_scales[i]);
                break;
            }
        }
        return test_result::fail;
    }

    return test_result::pass;
}

static test_result test_mxfp8_set_rows(ggml_backend_t backend, int64_t head_dim) {
    constexpr int64_t n_head  = 2;
    constexpr int64_t n_cache = 5;
    constexpr int64_t n_rows  = 2;
    constexpr int64_t block_size = 32;
    const int64_t groups_per_head = head_dim / block_size;

    ggml_context_ptr ctx = make_context();
    if (!ctx) {
        return test_result::fail;
    }

    ggml_tensor * cache   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F8_E4M3, head_dim * n_head, n_cache);
    ggml_tensor * scales  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, groups_per_head * n_head, n_cache);
    ggml_tensor * source  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim * n_head, n_rows);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, n_rows);
    ggml_tensor * write = ggml_set_rows_f8_block_scaled(
        ctx.get(), cache, scales, source, indices, static_cast<int32_t>(head_dim), block_size);

    if (!ggml_backend_supports_op(backend, write)) {
        return test_result::skip;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, write);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        return test_result::fail;
    }

    constexpr uint8_t sentinel = 0xa5;
    const std::array<int64_t, n_rows> row_indices = { 3, 1 };
    const std::array<float, 8> normalized = { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 448.0f };
    std::vector<float> source_data(ggml_nelements(source));
    std::vector<uint8_t> expected_cache(ggml_nbytes(cache), sentinel);
    std::vector<uint8_t> expected_scales(ggml_nbytes(scales), 0xff);

    for (int64_t row = 0; row < n_rows; ++row) {
        const int64_t dst_row = row_indices[row];
        for (int64_t head = 0; head < n_head; ++head) {
            for (int64_t group = 0; group < groups_per_head; ++group) {
                const bool zero = row == 1 && head == 0 && group == groups_per_head / 2;
                const int scale_exponent = int((3 * row + 5 * head + group) % 9) - 4;
                const float block_scale = std::ldexp(1.0f, scale_exponent);
                const uint8_t scale_byte = zero ? 0 : static_cast<uint8_t>(scale_exponent + 127);
                expected_scales[dst_row * groups_per_head * n_head + head * groups_per_head + group] = scale_byte;
                for (int64_t lane = 0; lane < block_size; ++lane) {
                    const int64_t d = group * block_size + lane;
                    const size_t src_i = row * head_dim * n_head + head * head_dim + d;
                    const size_t dst_i = dst_row * head_dim * n_head + head * head_dim + d;
                    source_data[src_i] = zero ? 0.0f : block_scale * normalized[lane % normalized.size()];
                    expected_cache[dst_i] = zero ? 0 : ggml_fp32_to_fp8_e4m3(normalized[lane % normalized.size()]);
                }
            }
        }
    }

    std::vector<uint8_t> cache_data(ggml_nbytes(cache), sentinel);
    std::vector<uint8_t> scale_data(ggml_nbytes(scales), 0xff);
    ggml_backend_tensor_set(cache, cache_data.data(), 0, cache_data.size());
    ggml_backend_tensor_set(scales, scale_data.data(), 0, scale_data.size());
    ggml_backend_tensor_set(source, source_data.data(), 0, source_data.size() * sizeof(float));
    ggml_backend_tensor_set(indices, row_indices.data(), 0, row_indices.size() * sizeof(int64_t));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%lld-wide MXFP8 SET_ROWS graph compute failed\n", (long long) head_dim);
        return test_result::fail;
    }
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(cache, cache_data.data(), 0, cache_data.size());
    ggml_backend_tensor_get(scales, scale_data.data(), 0, scale_data.size());

    if (cache_data != expected_cache || scale_data != expected_scales) {
        std::fprintf(stderr, "%lld-wide MXFP8 SET_ROWS result mismatch\n", (long long) head_dim);
        return test_result::fail;
    }
    return test_result::pass;
}

static float normalized_k(int64_t row, int64_t head, int64_t d, int64_t head_dim) {
    const int64_t anchor = (17 * row + 31 * head) % head_dim;
    if (d == anchor) {
        return (row + head) % 2 == 0 ? 448.0f : -448.0f;
    }
    return float((3 * d + 5 * row + 7 * head) % 17 - 8);
}

static float normalized_v(int64_t row, int64_t head, int64_t d, int64_t head_dim) {
    const int64_t anchor = (29 * row + 11 * head + 7) % head_dim;
    if (d == anchor) {
        return (row + 2 * head) % 2 == 0 ? -448.0f : 448.0f;
    }
    return 0.5f * float((5 * d + 3 * row + 11 * head) % 25 - 12);
}

template <int64_t n_kv = 5, int64_t n_head_q = 4>
static test_result test_scaled_flash_attention(
        ggml_backend_t backend, int64_t head_dim, int64_t n_query, bool tiny_scale = false) {
    constexpr int64_t n_head_kv = 2;
    constexpr int64_t n_seq     = 1;

    const float     attention_scale = 1.0f / std::sqrt(float(head_dim));
    constexpr float logit_softcap   = 0.75f;

    ggml_context_ptr ctx = make_context();
    if (!ctx) {
        std::fprintf(stderr, "failed to create GGML context\n");
        return test_result::fail;
    }

    ggml_tensor * k_cache = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F8_E4M3, head_dim * n_head_kv, n_kv, n_seq);
    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F8_E4M3, head_dim * n_head_kv, n_kv, n_seq);
    ggml_tensor * k_scale = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_head_kv, n_kv, n_seq);
    ggml_tensor * v_scale = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_head_kv, n_kv, n_seq);
    ggml_tensor * k_src   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim * n_head_kv, n_kv, n_seq);
    ggml_tensor * v_src   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim * n_head_kv, n_kv, n_seq);
    ggml_tensor * indices = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I64, n_kv, n_seq);
    ggml_tensor * q       = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, n_query, n_head_q, n_seq);
    ggml_tensor * mask    = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_kv, n_query, 1, n_seq);

    ggml_tensor * k_write =
        ggml_set_rows_f8_scaled(ctx.get(), k_cache, k_scale, k_src, indices, static_cast<int32_t>(head_dim));
    ggml_tensor * v_write =
        ggml_set_rows_f8_scaled(ctx.get(), v_cache, v_scale, v_src, indices, static_cast<int32_t>(head_dim));

    ggml_tensor * k =
        ggml_view_4d(ctx.get(), k_cache, head_dim, n_kv, n_head_kv, n_seq, k_cache->nb[1], head_dim, k_cache->nb[2], 0);
    ggml_tensor * v =
        ggml_view_4d(ctx.get(), v_cache, head_dim, n_kv, n_head_kv, n_seq, v_cache->nb[1], head_dim, v_cache->nb[2], 0);
    ggml_tensor * output = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, attention_scale, 0.0f, logit_softcap);
    ggml_flash_attn_ext_add_kv_scales(output, k_scale, v_scale);

    if (!ggml_backend_supports_op(backend, k_write) || !ggml_backend_supports_op(backend, v_write)) {
        return test_result::skip;
    }
    if (!ggml_backend_supports_op(backend, output)) {
        std::fprintf(stderr, "CUDA device supports %lld-wide FP8 cache writes but not scaled FP8 Flash Attention\n",
                     (long long) head_dim);
        return test_result::fail;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, k_write);
    ggml_build_forward_expand(graph, v_write);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate CUDA tensors for %lld-wide scaled FP8 Flash Attention\n",
                     (long long) head_dim);
        return test_result::fail;
    }

    std::array<int64_t, n_kv>       row_indices;
    for (int64_t row = 0; row < n_kv; ++row) {
        row_indices[row] = (17 * row + 2) % n_kv;
    }
    std::vector<float>              k_source(ggml_nelements(k_src));
    std::vector<float>              v_source(ggml_nelements(v_src));
    std::vector<float>              q_data(ggml_nelements(q));
    std::vector<ggml_fp16_t>        mask_data(ggml_nelements(mask));

    std::vector<uint8_t> k_expected(ggml_nbytes(k_cache), 0);
    std::vector<uint8_t> v_expected(ggml_nbytes(v_cache), 0);
    std::vector<float>   ks_expected(ggml_nelements(k_scale), 0.0f);
    std::vector<float>   vs_expected(ggml_nelements(v_scale), 0.0f);

    for (int64_t row = 0; row < n_kv; ++row) {
        const int64_t dst_row = row_indices[row];
        for (int64_t head = 0; head < n_head_kv; ++head) {
            const float ks = tiny_scale ? std::ldexp(1.0f, -26) : std::ldexp(1.0f, -9 + int((row + head) % 4));
            const float vs = tiny_scale ? std::ldexp(1.0f, -26) : std::ldexp(1.0f, -9 + int((2 * row + head) % 3));
            ks_expected[dst_row * n_head_kv + head] = ks;
            vs_expected[dst_row * n_head_kv + head] = vs;
            for (int64_t d = 0; d < head_dim; ++d) {
                const float  nk    = normalized_k(row, head, d, head_dim);
                const float  nv    = normalized_v(row, head, d, head_dim);
                const size_t src_i = row * head_dim * n_head_kv + head * head_dim + d;
                const size_t dst_i = dst_row * head_dim * n_head_kv + head * head_dim + d;
                k_source[src_i]    = ks * nk;
                v_source[src_i]    = vs * nv;
                k_expected[dst_i]  = ggml_fp32_to_fp8_e4m3(nk);
                v_expected[dst_i]  = ggml_fp32_to_fp8_e4m3(nv);
            }
        }
    }

    for (int64_t head = 0; head < n_head_q; ++head) {
        for (int64_t query = 0; query < n_query; ++query) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const int value                                 = int((7 * d + 11 * query + 13 * head) % 23) - 11;
                q_data[d + head_dim * (query + n_query * head)] =
                    std::ldexp(float(value), tiny_scale ? 12 : -7);
            }
        }
    }

    for (int64_t query = 0; query < n_query; ++query) {
        for (int64_t token = 0; token < n_kv; ++token) {
            float value = query == 0 ? -0.125f * float(token) : 0.0625f * float(token - 2);
            if (query == 1 && token >= n_kv - 2) {
                value = -std::numeric_limits<float>::infinity();
            }
            mask_data[token + n_kv * query] = ggml_fp32_to_fp16(value);
        }
    }

    std::vector<uint8_t> cache_zero(ggml_nbytes(k_cache), 0);
    std::vector<float>   scale_zero(ggml_nelements(k_scale), 0.0f);
    ggml_backend_tensor_set(k_cache, cache_zero.data(), 0, cache_zero.size());
    ggml_backend_tensor_set(v_cache, cache_zero.data(), 0, cache_zero.size());
    ggml_backend_tensor_set(k_scale, scale_zero.data(), 0, scale_zero.size() * sizeof(float));
    ggml_backend_tensor_set(v_scale, scale_zero.data(), 0, scale_zero.size() * sizeof(float));
    ggml_backend_tensor_set(k_src, k_source.data(), 0, k_source.size() * sizeof(float));
    ggml_backend_tensor_set(v_src, v_source.data(), 0, v_source.size() * sizeof(float));
    ggml_backend_tensor_set(indices, row_indices.data(), 0, row_indices.size() * sizeof(int64_t));
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%lld-wide scaled FP8 Flash Attention graph compute failed\n", (long long) head_dim);
        return test_result::fail;
    }
    ggml_backend_synchronize(backend);

    std::vector<float> output_data(ggml_nelements(output));
    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

    std::vector<float> reference(output_data.size(), 0.0f);
    constexpr int64_t  gqa_ratio = n_head_q / n_head_kv;
    for (int64_t head = 0; head < n_head_q; ++head) {
        const int64_t kv_head = head / gqa_ratio;
        for (int64_t query = 0; query < n_query; ++query) {
            std::array<float, n_kv> logits;
            float                   row_max = -std::numeric_limits<float>::infinity();
            for (int64_t token = 0; token < n_kv; ++token) {
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    const size_t q_i = d + head_dim * (query + n_query * head);
                    const size_t k_i = token * head_dim * n_head_kv + kv_head * head_dim + d;
                    float k_value =
                        ggml_fp8_e4m3_to_fp32(k_expected[k_i]) * ks_expected[token * n_head_kv + kv_head];
                    if (n_query > 2) {
                        k_value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(k_value));
                    }
                    dot += q_data[q_i] * k_value;
                }
                float logit = logit_softcap * std::tanh(dot * attention_scale / logit_softcap);
                logit += ggml_fp16_to_fp32(mask_data[token + n_kv * query]);
                logits[token] = logit;
                row_max       = std::max(row_max, logit);
            }

            float                   row_sum = 0.0f;
            std::array<float, n_kv> weights;
            for (int64_t token = 0; token < n_kv; ++token) {
                weights[token] = std::exp(logits[token] - row_max);
                row_sum += weights[token];
            }

            for (int64_t d = 0; d < head_dim; ++d) {
                float value = 0.0f;
                for (int64_t token = 0; token < n_kv; ++token) {
                    const size_t v_i = token * head_dim * n_head_kv + kv_head * head_dim + d;
                    float v_value =
                        ggml_fp8_e4m3_to_fp32(v_expected[v_i]) * vs_expected[token * n_head_kv + kv_head];
                    if (n_query > 2) {
                        v_value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v_value));
                    }
                    value += weights[token] * v_value;
                }
                reference[d + head_dim * (head + n_head_q * query)] = value / row_sum;
            }
        }
    }

    float  max_abs_error = 0.0f;
    size_t max_error_i   = 0;
    for (size_t i = 0; i < output_data.size(); ++i) {
        const float error = std::fabs(output_data[i] - reference[i]);
        if (error > max_abs_error) {
            max_abs_error = error;
            max_error_i   = i;
        }
        const float tolerance = tiny_scale ? 1.0e-7f + 5.0e-4f * std::fabs(reference[i])
                                           : 3.0e-4f + 3.0e-4f * std::fabs(reference[i]);
        if (!std::isfinite(output_data[i]) || error > tolerance) {
            std::fprintf(stderr,
                         "%lld-wide scaled FP8 Flash Attention mismatch at %zu: got %g, expected %g, error %g\n",
                         (long long) head_dim, i, output_data[i], reference[i], error);
            return test_result::fail;
        }
    }

    std::printf("%lld-wide scaled FP8 Flash Attention max absolute error: %g at %zu\n", (long long) head_dim,
                max_abs_error, max_error_i);
    return test_result::pass;
}

static std::vector<float> attention_reference(
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & q,
        const std::vector<ggml_fp16_t> & mask,
        int64_t head_dim,
        int64_t n_kv,
        int64_t n_query,
        int64_t n_head_q,
        int64_t n_head_kv,
        float attention_scale,
        float logit_softcap) {
    std::vector<float> result(head_dim * n_query * n_head_q, 0.0f);
    const int64_t gqa_ratio = n_head_q / n_head_kv;
    std::vector<float> logits(n_kv);
    std::vector<float> weights(n_kv);

    for (int64_t head = 0; head < n_head_q; ++head) {
        const int64_t kv_head = head / gqa_ratio;
        for (int64_t query = 0; query < n_query; ++query) {
            float row_max = -std::numeric_limits<float>::infinity();
            for (int64_t token = 0; token < n_kv; ++token) {
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q[d + head_dim * (query + n_query * head)] *
                        k[(token * n_head_kv + kv_head) * head_dim + d];
                }
                float logit = dot * attention_scale;
                if (logit_softcap != 0.0f) {
                    logit = logit_softcap * std::tanh(logit / logit_softcap);
                }
                logit += ggml_fp16_to_fp32(mask[token + n_kv * query]);
                logits[token] = logit;
                row_max = std::max(row_max, logit);
            }

            float row_sum = 0.0f;
            for (int64_t token = 0; token < n_kv; ++token) {
                weights[token] = std::exp(logits[token] - row_max);
                row_sum += weights[token];
            }
            for (int64_t d = 0; d < head_dim; ++d) {
                float value = 0.0f;
                for (int64_t token = 0; token < n_kv; ++token) {
                    value += weights[token] * v[(token * n_head_kv + kv_head) * head_dim + d];
                }
                result[d + head_dim * (head + n_head_q * query)] = value / row_sum;
            }
        }
    }
    return result;
}

static test_result check_attention(
        ggml_tensor * output,
        const std::vector<float> & reference,
        const char * label,
        float abs_tolerance = 8.0e-4f,
        float rel_tolerance = 8.0e-4f) {
    std::vector<float> output_data(ggml_nelements(output));
    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));
    for (size_t i = 0; i < output_data.size(); ++i) {
        const float error = std::fabs(output_data[i] - reference[i]);
        const float tolerance = abs_tolerance + rel_tolerance * std::fabs(reference[i]);
        if (!std::isfinite(output_data[i]) || error > tolerance) {
            std::fprintf(stderr, "%s mismatch at %zu: got %g, expected %g, error %g\n",
                label, i, output_data[i], reference[i], error);
            return test_result::fail;
        }
    }
    return test_result::pass;
}

static test_result test_mxfp8_flash_attention(ggml_backend_t backend, int64_t head_dim, int64_t n_query) {
    constexpr int64_t block_size = 32;
    constexpr int64_t n_kv       = 7;
    constexpr int64_t n_head_kv  = 2;
    constexpr int64_t n_head_q   = 4;
    constexpr int64_t n_seq      = 1;
    const int64_t groups_per_head = head_dim / block_size;
    const float attention_scale = 1.0f / std::sqrt(float(head_dim));
    constexpr float logit_softcap = 0.75f;

    ggml_context_ptr ctx = make_context();
    if (!ctx) {
        return test_result::fail;
    }

    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F8_E4M3, head_dim, n_kv, n_head_kv, n_seq);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F8_E4M3, head_dim, n_kv, n_head_kv, n_seq);
    ggml_tensor * ks = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, groups_per_head * n_head_kv, n_kv, n_seq);
    ggml_tensor * vs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, groups_per_head * n_head_kv, n_kv, n_seq);
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, n_query, n_head_q, n_seq);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_kv, n_query, 1, n_seq);
    ggml_tensor * output = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, attention_scale, 0.0f, logit_softcap);
    ggml_flash_attn_ext_add_kv_block_scales(output, ks, vs, block_size);

    if (!ggml_backend_supports_op(backend, output)) {
        return test_result::skip;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        return test_result::fail;
    }

    std::vector<uint8_t> k_data(ggml_nbytes(k));
    std::vector<uint8_t> v_data(ggml_nbytes(v));
    std::vector<uint8_t> ks_data(ggml_nbytes(ks));
    std::vector<uint8_t> vs_data(ggml_nbytes(vs));
    std::vector<float> k_decoded(head_dim * n_kv * n_head_kv);
    std::vector<float> v_decoded(head_dim * n_kv * n_head_kv);

    for (int64_t token = 0; token < n_kv; ++token) {
        for (int64_t head = 0; head < n_head_kv; ++head) {
            for (int64_t group = 0; group < groups_per_head; ++group) {
                std::array<float, block_size> k_group;
                std::array<float, block_size> v_group;
                float k_amax = 0.0f;
                float v_amax = 0.0f;
                for (int64_t lane = 0; lane < block_size; ++lane) {
                    const int64_t d = group * block_size + lane;
                    k_group[lane] = std::ldexp(float((7 * d + 11 * token + 13 * head) % 31 - 15), -4);
                    v_group[lane] = std::ldexp(float((5 * d + 3 * token + 17 * head) % 37 - 18), -5);
                    k_amax = std::max(k_amax, std::fabs(k_group[lane]));
                    v_amax = std::max(v_amax, std::fabs(v_group[lane]));
                }
                const uint8_t k_scale_byte = ggml_mxfp8_e4m3_scale(k_amax);
                const uint8_t v_scale_byte = ggml_mxfp8_e4m3_scale(v_amax);
                const float k_scale = ggml_ue8m0_to_fp32(k_scale_byte);
                const float v_scale = ggml_ue8m0_to_fp32(v_scale_byte);
                const size_t scale_i = token * groups_per_head * n_head_kv + head * groups_per_head + group;
                ks_data[scale_i] = k_scale_byte;
                vs_data[scale_i] = v_scale_byte;
                for (int64_t lane = 0; lane < block_size; ++lane) {
                    const int64_t d = group * block_size + lane;
                    const size_t cache_i = d + head_dim * (token + n_kv * head);
                    const size_t logical_i = (token * n_head_kv + head) * head_dim + d;
                    k_data[cache_i] = ggml_fp32_to_fp8_e4m3(k_group[lane] / k_scale);
                    v_data[cache_i] = ggml_fp32_to_fp8_e4m3(v_group[lane] / v_scale);
                    k_decoded[logical_i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                        ggml_fp8_e4m3_to_fp32(k_data[cache_i]) * k_scale));
                    v_decoded[logical_i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                        ggml_fp8_e4m3_to_fp32(v_data[cache_i]) * v_scale));
                }
            }
        }
    }

    std::vector<float> q_data(ggml_nelements(q));
    for (int64_t head = 0; head < n_head_q; ++head) {
        for (int64_t query = 0; query < n_query; ++query) {
            for (int64_t d = 0; d < head_dim; ++d) {
                q_data[d + head_dim * (query + n_query * head)] =
                    std::ldexp(float((3 * d + 5 * query + 7 * head) % 29 - 14), -6);
            }
        }
    }
    std::vector<ggml_fp16_t> mask_data(ggml_nelements(mask));
    for (int64_t query = 0; query < n_query; ++query) {
        for (int64_t token = 0; token < n_kv; ++token) {
            const float value = query % 2 != 0 && token == n_kv - 1 ?
                -std::numeric_limits<float>::infinity() : -0.03125f * float(token);
            mask_data[token + n_kv * query] = ggml_fp32_to_fp16(value);
        }
    }

    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size());
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size());
    ggml_backend_tensor_set(ks, ks_data.data(), 0, ks_data.size());
    ggml_backend_tensor_set(vs, vs_data.data(), 0, vs_data.size());
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%lld-wide MXFP8 Flash Attention graph compute failed\n", (long long) head_dim);
        return test_result::fail;
    }
    ggml_backend_synchronize(backend);

    const std::vector<float> reference = attention_reference(
        k_decoded, v_decoded, q_data, mask_data, head_dim, n_kv, n_query, n_head_q, n_head_kv,
        attention_scale, logit_softcap);
    return check_attention(output, reference, "MXFP8 Flash Attention");
}

static test_result test_mxfp8_hot_attention(
        ggml_backend_t backend, int64_t head_dim, int64_t n_tokens, int64_t n_kv) {
    constexpr int64_t block_size = 32;
    constexpr int64_t hot_size   = 8;
    constexpr int64_t sink_size  = 2;
    constexpr int64_t n_head_kv  = 2;
    constexpr int64_t n_head_q   = 12;
    constexpr int64_t n_query    = 2;
    const int64_t groups_per_head = head_dim / block_size;
    const int64_t cold_size = n_kv - hot_size;
    const float attention_scale = 1.0f / std::sqrt(float(head_dim));

    if (n_tokens <= 0 || n_tokens > n_kv || cold_size <= 0) {
        return test_result::fail;
    }

    ggml_context_ptr ctx = make_context();
    if (!ctx) {
        return test_result::fail;
    }

    ggml_tensor * k_hot = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, head_dim * n_head_kv, hot_size);
    ggml_tensor * v_hot = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, head_dim * n_head_kv, hot_size);
    ggml_tensor * k_cold = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F8_E4M3, head_dim * n_head_kv, cold_size);
    ggml_tensor * v_cold = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F8_E4M3, head_dim * n_head_kv, cold_size);
    ggml_tensor * ks = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, groups_per_head * n_head_kv, cold_size);
    ggml_tensor * vs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I8, groups_per_head * n_head_kv, cold_size);
    ggml_tensor * k_src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim * n_head_kv, n_tokens);
    ggml_tensor * v_src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim * n_head_kv, n_tokens);
    ggml_tensor * logical = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, n_tokens);
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, n_query, n_head_q, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, n_kv, n_query, 1, 1);

    const int64_t n_prefix = n_tokens < hot_size ? n_tokens : hot_size;
    const int64_t n_append = n_tokens - n_prefix;
    ggml_tensor * k_src_prefix = ggml_view_2d(ctx.get(), k_src, head_dim * n_head_kv, n_prefix, k_src->nb[1], 0);
    ggml_tensor * v_src_prefix = ggml_view_2d(ctx.get(), v_src, head_dim * n_head_kv, n_prefix, v_src->nb[1], 0);
    ggml_tensor * logical_prefix = ggml_view_1d(ctx.get(), logical, n_prefix, 0);
    ggml_tensor * k_write_prefix = ggml_set_rows_mxfp8_hot(ctx.get(), k_hot, k_cold, ks, k_src_prefix, logical_prefix,
        static_cast<int32_t>(head_dim), hot_size, sink_size);
    ggml_tensor * v_write_prefix = ggml_set_rows_mxfp8_hot(ctx.get(), v_hot, v_cold, vs, v_src_prefix, logical_prefix,
        static_cast<int32_t>(head_dim), hot_size, sink_size);
    ggml_tensor * k_write_append = nullptr;
    ggml_tensor * v_write_append = nullptr;
    ggml_tensor * logical_fa = logical_prefix;
    if (n_append > 0) {
        ggml_tensor * k_src_append = ggml_view_2d(ctx.get(), k_src, head_dim * n_head_kv, n_append,
            k_src->nb[1], n_prefix * k_src->nb[1]);
        ggml_tensor * v_src_append = ggml_view_2d(ctx.get(), v_src, head_dim * n_head_kv, n_append,
            v_src->nb[1], n_prefix * v_src->nb[1]);
        ggml_tensor * logical_append = ggml_view_1d(ctx.get(), logical, n_append, n_prefix * logical->nb[0]);
        k_write_append = ggml_set_rows_mxfp8_hot(ctx.get(), k_hot, k_cold, ks, k_src_append, logical_append,
            static_cast<int32_t>(head_dim), hot_size, sink_size);
        v_write_append = ggml_set_rows_mxfp8_hot(ctx.get(), v_hot, v_cold, vs, v_src_append, logical_append,
            static_cast<int32_t>(head_dim), hot_size, sink_size);
        logical_fa = logical_append;
    }
    ggml_tensor * k = ggml_view_4d(ctx.get(), k_cold, head_dim, cold_size, n_head_kv, 1,
        k_cold->nb[1], head_dim, k_cold->nb[2], 0);
    ggml_tensor * v = ggml_view_4d(ctx.get(), v_cold, head_dim, cold_size, n_head_kv, 1,
        v_cold->nb[1], head_dim, v_cold->nb[2], 0);
    ggml_tensor * k_hot_fa = ggml_view_4d(ctx.get(), k_hot, head_dim, hot_size, n_head_kv, 1,
        k_hot->nb[1], head_dim * sizeof(ggml_fp16_t), k_hot->nb[2], 0);
    ggml_tensor * v_hot_fa = ggml_view_4d(ctx.get(), v_hot, head_dim, hot_size, n_head_kv, 1,
        v_hot->nb[1], head_dim * sizeof(ggml_fp16_t), v_hot->nb[2], 0);
    ggml_tensor * output = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, attention_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_add_mxfp8_hot(output, ks, vs, k_hot_fa, v_hot_fa, logical_fa,
        hot_size, sink_size, block_size, static_cast<int32_t>(n_kv));

    if (!ggml_backend_supports_op(backend, k_write_prefix) || !ggml_backend_supports_op(backend, v_write_prefix) ||
            (k_write_append && !ggml_backend_supports_op(backend, k_write_append)) ||
            (v_write_append && !ggml_backend_supports_op(backend, v_write_append)) || !ggml_backend_supports_op(backend, output)) {
        return test_result::skip;
    }

    ggml_cgraph * graph_prefix = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph_prefix, k_write_prefix);
    ggml_build_forward_expand(graph_prefix, v_write_prefix);
    ggml_cgraph * graph_append = ggml_new_graph(ctx.get());
    if (k_write_append) {
        ggml_build_forward_expand(graph_append, k_write_append);
        ggml_build_forward_expand(graph_append, v_write_append);
    }
    ggml_build_forward_expand(graph_append, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        return test_result::fail;
    }

    std::vector<float> k_source(ggml_nelements(k_src));
    std::vector<float> v_source(ggml_nelements(v_src));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t head = 0; head < n_head_kv; ++head) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const size_t i = (token * n_head_kv + head) * head_dim + d;
                k_source[i] = std::ldexp(float((5 * d + 7 * token + 11 * head) % 41 - 20), -5);
                v_source[i] = std::ldexp(float((3 * d + 13 * token + 17 * head) % 43 - 21), -6);
            }
        }
    }
    std::vector<int64_t> logical_data(n_tokens);
    for (int64_t token = 0; token < n_tokens; ++token) {
        logical_data[token] = token;
    }

    const auto simulate_cache = [&](const std::vector<float> & source,
                                    std::vector<ggml_fp16_t> & hot_expected,
                                    std::vector<uint8_t> & cold_expected,
                                    std::vector<uint8_t> & scale_expected,
                                    std::vector<float> & decoded) {
        for (int64_t token = 0; token < n_tokens; ++token) {
            const int64_t tail_size = hot_size - sink_size;
            const int64_t slot = token < sink_size ? token : sink_size + (token - sink_size) % tail_size;
            if (token >= hot_size) {
                const int64_t cold_row = token - hot_size;
                for (int64_t head = 0; head < n_head_kv; ++head) {
                    for (int64_t group = 0; group < groups_per_head; ++group) {
                        float amax = 0.0f;
                        for (int64_t lane = 0; lane < block_size; ++lane) {
                            const int64_t d = group * block_size + lane;
                            const size_t hot_i = slot * head_dim * n_head_kv + head * head_dim + d;
                            amax = std::max(amax, std::fabs(ggml_fp16_to_fp32(hot_expected[hot_i])));
                        }
                        const uint8_t scale_byte = ggml_mxfp8_e4m3_scale(amax);
                        const float block_scale = ggml_ue8m0_to_fp32(scale_byte);
                        scale_expected[cold_row * groups_per_head * n_head_kv + head * groups_per_head + group] =
                            scale_byte;
                        for (int64_t lane = 0; lane < block_size; ++lane) {
                            const int64_t d = group * block_size + lane;
                            const size_t hot_i = slot * head_dim * n_head_kv + head * head_dim + d;
                            const size_t cold_i = cold_row * head_dim * n_head_kv + head * head_dim + d;
                            cold_expected[cold_i] = ggml_fp32_to_fp8_e4m3(
                                ggml_fp16_to_fp32(hot_expected[hot_i]) / block_scale);
                        }
                    }
                }
            }
            for (int64_t head = 0; head < n_head_kv; ++head) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    const size_t hot_i = slot * head_dim * n_head_kv + head * head_dim + d;
                    const size_t src_i = (token * n_head_kv + head) * head_dim + d;
                    hot_expected[hot_i] = ggml_fp32_to_fp16(source[src_i]);
                }
            }
        }

        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t head = 0; head < n_head_kv; ++head) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    const size_t logical_i = (token * n_head_kv + head) * head_dim + d;
                    decoded[logical_i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(source[logical_i]));
                }
            }
        }
        const int64_t cold_count = n_tokens > hot_size ? n_tokens - hot_size : 0;
        for (int64_t cold_row = 0; cold_row < cold_count; ++cold_row) {
            const int64_t token = sink_size + cold_row;
            for (int64_t head = 0; head < n_head_kv; ++head) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    const int64_t group = d / block_size;
                    const size_t cold_i = cold_row * head_dim * n_head_kv + head * head_dim + d;
                    const size_t scale_i = cold_row * groups_per_head * n_head_kv + head * groups_per_head + group;
                    const size_t logical_i = (token * n_head_kv + head) * head_dim + d;
                    decoded[logical_i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
                        ggml_fp8_e4m3_to_fp32(cold_expected[cold_i]) * ggml_ue8m0_to_fp32(scale_expected[scale_i])));
                }
            }
        }
    };

    std::vector<ggml_fp16_t> k_hot_expected(ggml_nelements(k_hot), ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> v_hot_expected(ggml_nelements(v_hot), ggml_fp32_to_fp16(0.0f));
    std::vector<uint8_t> k_cold_expected(ggml_nbytes(k_cold), 0xa5);
    std::vector<uint8_t> v_cold_expected(ggml_nbytes(v_cold), 0xa5);
    std::vector<uint8_t> ks_expected(ggml_nbytes(ks), 0xff);
    std::vector<uint8_t> vs_expected(ggml_nbytes(vs), 0xff);
    std::vector<float> k_decoded(head_dim * n_kv * n_head_kv, 0.0f);
    std::vector<float> v_decoded(head_dim * n_kv * n_head_kv, 0.0f);
    simulate_cache(k_source, k_hot_expected, k_cold_expected, ks_expected, k_decoded);
    simulate_cache(v_source, v_hot_expected, v_cold_expected, vs_expected, v_decoded);

    std::vector<float> q_data(ggml_nelements(q));
    for (int64_t head = 0; head < n_head_q; ++head) {
        for (int64_t query = 0; query < n_query; ++query) {
            for (int64_t d = 0; d < head_dim; ++d) {
                q_data[d + head_dim * (query + n_query * head)] =
                    std::ldexp(float((7 * d + 3 * query + 5 * head) % 31 - 15), -7);
            }
        }
    }
    std::vector<ggml_fp16_t> mask_data(ggml_nelements(mask));
    for (int64_t query = 0; query < n_query; ++query) {
        for (int64_t token = 0; token < n_kv; ++token) {
            const float value = token < n_tokens ? -0.015625f * float(token) :
                -std::numeric_limits<float>::infinity();
            mask_data[token + n_kv * query] = ggml_fp32_to_fp16(value);
        }
    }

    std::vector<ggml_fp16_t> hot_zero(ggml_nelements(k_hot), ggml_fp32_to_fp16(0.0f));
    std::vector<uint8_t> cold_sentinel(ggml_nbytes(k_cold), 0xa5);
    std::vector<uint8_t> scale_sentinel(ggml_nbytes(ks), 0xff);
    ggml_backend_tensor_set(k_hot, hot_zero.data(), 0, hot_zero.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v_hot, hot_zero.data(), 0, hot_zero.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(k_cold, cold_sentinel.data(), 0, cold_sentinel.size());
    ggml_backend_tensor_set(v_cold, cold_sentinel.data(), 0, cold_sentinel.size());
    ggml_backend_tensor_set(ks, scale_sentinel.data(), 0, scale_sentinel.size());
    ggml_backend_tensor_set(vs, scale_sentinel.data(), 0, scale_sentinel.size());
    ggml_backend_tensor_set(k_src, k_source.data(), 0, k_source.size() * sizeof(float));
    ggml_backend_tensor_set(v_src, v_source.data(), 0, v_source.size() * sizeof(float));
    ggml_backend_tensor_set(logical, logical_data.data(), 0, logical_data.size() * sizeof(int64_t));
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_graph_compute(backend, graph_prefix) != GGML_STATUS_SUCCESS ||
            ggml_backend_graph_compute(backend, graph_append) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%lld-wide incremental hybrid MXFP8 graph compute failed at T=%lld, n_kv=%lld\n",
            (long long) head_dim, (long long) n_tokens, (long long) n_kv);
        return test_result::fail;
    }
    ggml_backend_synchronize(backend);

    std::vector<ggml_fp16_t> k_hot_data(ggml_nelements(k_hot));
    std::vector<ggml_fp16_t> v_hot_data(ggml_nelements(v_hot));
    std::vector<uint8_t> k_cold_data(ggml_nbytes(k_cold));
    std::vector<uint8_t> v_cold_data(ggml_nbytes(v_cold));
    std::vector<uint8_t> ks_data(ggml_nbytes(ks));
    std::vector<uint8_t> vs_data(ggml_nbytes(vs));
    ggml_backend_tensor_get(k_hot, k_hot_data.data(), 0, k_hot_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(v_hot, v_hot_data.data(), 0, v_hot_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(k_cold, k_cold_data.data(), 0, k_cold_data.size());
    ggml_backend_tensor_get(v_cold, v_cold_data.data(), 0, v_cold_data.size());
    ggml_backend_tensor_get(ks, ks_data.data(), 0, ks_data.size());
    ggml_backend_tensor_get(vs, vs_data.data(), 0, vs_data.size());
    if (std::memcmp(k_hot_data.data(), k_hot_expected.data(), k_hot_data.size() * sizeof(ggml_fp16_t)) != 0 ||
            std::memcmp(v_hot_data.data(), v_hot_expected.data(), v_hot_data.size() * sizeof(ggml_fp16_t)) != 0 ||
            k_cold_data != k_cold_expected || v_cold_data != v_cold_expected ||
            ks_data != ks_expected || vs_data != vs_expected) {
        std::fprintf(stderr, "%lld-wide hybrid MXFP8 cache mismatch at T=%lld, n_kv=%lld\n",
            (long long) head_dim, (long long) n_tokens, (long long) n_kv);
        return test_result::fail;
    }

    const std::vector<float> reference = attention_reference(
        k_decoded, v_decoded, q_data, mask_data, head_dim, n_kv, n_query, n_head_q, n_head_kv,
        attention_scale, 0.0f);
    return check_attention(output, reference, "hybrid MXFP8 Flash Attention", 1.5e-3f, 1.5e-3f);
}

int main() {
    if (!test_mxfp8_scale_reference()) {
        return 1;
    }

    ggml_backend_load_all();

    bool found_cuda = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!is_cuda_device(dev)) {
            continue;
        }
        found_cuda = true;

        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        if (!backend) {
            continue;
        }

        bool supported_device = true;
        for (const int64_t head_dim : { int64_t(128), int64_t(256) }) {
            const test_result set_rows = test_scaled_set_rows(backend.get(), head_dim);
            if (set_rows == test_result::skip) {
                if (head_dim == 128) {
                    supported_device = false;
                    break;
                }
                std::fprintf(stderr, "CUDA SM120 device supports 128-wide but not 256-wide scaled FP8 cache writes\n");
                return 1;
            }
            if (set_rows == test_result::fail) {
                return 1;
            }

            const test_result mx_set_rows = test_mxfp8_set_rows(backend.get(), head_dim);
            if (mx_set_rows != test_result::pass) {
                std::fprintf(stderr, "CUDA SM120 device failed the %lld-wide MXFP8 cache write test\n",
                    (long long) head_dim);
                return 1;
            }

            for (const int64_t n_query : { int64_t(2), int64_t(8) }) {
                const test_result flash_attention = test_scaled_flash_attention(backend.get(), head_dim, n_query);
                if (flash_attention == test_result::fail) {
                    return 1;
                }
                if (flash_attention == test_result::skip) {
                    std::fprintf(stderr, "CUDA SM120 device supports scaled FP8 writes but not %lld-wide Flash Attention\n",
                                 (long long) head_dim);
                    return 1;
                }
            }
            for (const int64_t n_query : { int64_t(1), int64_t(2), int64_t(8) }) {
                if (test_mxfp8_flash_attention(backend.get(), head_dim, n_query) != test_result::pass) {
                    std::fprintf(stderr, "CUDA SM120 device failed the %lld-wide, Q=%lld MXFP8 Flash Attention test\n",
                        (long long) head_dim, (long long) n_query);
                    return 1;
                }
            }
            if (head_dim == 128 && test_scaled_flash_attention(backend.get(), head_dim, 8, true) != test_result::pass) {
                std::fprintf(stderr, "CUDA SM120 device failed the tiny-scale FP8 Flash Attention test\n");
                return 1;
            }
        }
        if (test_scaled_flash_attention<256, 12>(backend.get(), 128, 8) != test_result::pass) {
            std::fprintf(stderr, "CUDA SM120 device failed the GQA-6 FP8 Flash Attention test\n");
            return 1;
        }
        for (const std::array<int64_t, 2> boundary : {
                std::array<int64_t, 2>{ 7, 255 },
                std::array<int64_t, 2>{ 8, 256 },
                std::array<int64_t, 2>{ 9, 257 } }) {
            if (test_mxfp8_hot_attention(backend.get(), 128, boundary[0], boundary[1]) != test_result::pass) {
                std::fprintf(stderr, "CUDA SM120 device failed hybrid MXFP8 boundary T=%lld, n_kv=%lld\n",
                    (long long) boundary[0], (long long) boundary[1]);
                return 1;
            }
        }
        if (test_mxfp8_hot_attention(backend.get(), 256, 9, 257) != test_result::pass) {
            std::fprintf(stderr, "CUDA SM120 device failed 256-wide hybrid MXFP8 boundary test\n");
            return 1;
        }
        if (!supported_device) {
            continue;
        }

        std::printf("PASS: CUDA FP8 KV synthetic tests for 128-wide and 256-wide heads on %s\n",
                    ggml_backend_dev_name(dev));
        return 0;
    }

    if (!found_cuda) {
        std::printf("SKIP: no CUDA backend is available\n");
    } else {
        std::printf("SKIP: no CUDA SM120 device with the FP8 KV path is available\n");
    }
    return 0;
}
