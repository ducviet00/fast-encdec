// Thin pybind wrapper over the vendored vLLM CPU attention kernels
// (csrc/cpu/cpu_attn.cpp). See csrc/README.md for provenance/license.
#include <torch/extension.h>

#include <optional>
#include <string>

// Defined in cpu_attn.cpp (global namespace).
torch::Tensor get_scheduler_metadata(
    const int64_t num_req, const int64_t num_heads_q, const int64_t num_heads_kv,
    const int64_t head_dim, const torch::Tensor& seq_lens, at::ScalarType dtype,
    const torch::Tensor& query_start_loc, const bool causal,
    const int64_t window_size, const std::string& isa_hint,
    const bool enable_kv_split,
    const std::optional<torch::Tensor>& dynamic_causal,
    const std::string& kv_cache_dtype);

void cpu_attn_reshape_and_cache(
    const torch::Tensor& key, const torch::Tensor& value,
    torch::Tensor& key_cache, torch::Tensor& value_cache,
    const torch::Tensor& slot_mapping, const std::string& isa,
    const double k_scale, const double v_scale,
    const std::string& kv_cache_dtype);

void cpu_attention_with_kv_cache(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, torch::Tensor& output,
    const torch::Tensor& query_start_loc, const torch::Tensor& seq_lens,
    const double scale, const bool causal,
    const std::optional<torch::Tensor>& alibi_slopes,
    const int64_t sliding_window, const torch::Tensor& block_table,
    const double softcap, const torch::Tensor& scheduler_metadata,
    const std::optional<torch::Tensor>& s_aux,
    const std::optional<torch::Tensor>& dynamic_causal, const double k_scale,
    const double v_scale, const std::string& kv_cache_dtype);

namespace {

const std::string kAuto = "auto";

torch::Tensor build_metadata(int64_t num_req, int64_t num_heads_q,
                             int64_t num_heads_kv, int64_t head_dim,
                             torch::Tensor seq_lens, at::ScalarType dtype,
                             torch::Tensor query_start_loc, bool causal,
                             int64_t window_size, std::string isa,
                             bool enable_kv_split,
                             std::optional<torch::Tensor> dynamic_causal) {
  return get_scheduler_metadata(num_req, num_heads_q, num_heads_kv, head_dim,
                                seq_lens, dtype, query_start_loc, causal,
                                window_size, isa, enable_kv_split,
                                dynamic_causal, kAuto);
}

torch::Tensor attention_forward(torch::Tensor query, torch::Tensor key_cache,
                                torch::Tensor value_cache,
                                torch::Tensor query_start_loc,
                                torch::Tensor seq_lens, double scale,
                                bool causal, int64_t window_size,
                                torch::Tensor block_table,
                                torch::Tensor metadata,
                                std::optional<torch::Tensor> dynamic_causal) {
  auto output = torch::empty_like(query);
  cpu_attention_with_kv_cache(
      query, key_cache, value_cache, output, query_start_loc, seq_lens, scale,
      causal, std::nullopt, window_size, block_table, 0.0, metadata,
      std::nullopt, dynamic_causal, 1.0, 1.0, kAuto);
  return output;
}

void reshape_and_cache(torch::Tensor key, torch::Tensor value,
                       torch::Tensor key_cache, torch::Tensor value_cache,
                       torch::Tensor slot_mapping, std::string isa) {
  cpu_attn_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                             isa, 1.0, 1.0, kAuto);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_metadata", &build_metadata, "scheduler metadata");
  m.def("attention_forward", &attention_forward, "paged attention");
  m.def("reshape_and_cache", &reshape_and_cache, "scatter K/V into paged cache");
}
