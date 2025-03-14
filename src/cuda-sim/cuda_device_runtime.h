#ifndef __cuda_device_runtime_h__
#define __cuda_device_runtime_h__
// Jin: cuda_device_runtime.h
// Defines CUDA device runtime APIs for CDP support
class device_launch_config_t {
 public:
  device_launch_config_t() {}

  device_launch_config_t(dim3 _grid_dim, dim3 _block_dim,
                         unsigned int _shared_mem, function_info* _entry)
      : grid_dim(_grid_dim),
        block_dim(_block_dim),
        shared_mem(_shared_mem),
        entry(_entry) {}

  dim3 grid_dim;
  dim3 block_dim;
  unsigned int shared_mem;
  function_info* entry;
};

class device_launch_operation_t {
 public:
  device_launch_operation_t() {}
  device_launch_operation_t(kernel_info_t* _grid, CUstream_st* _stream)
      : grid(_grid), stream(_stream) {}

  kernel_info_t* grid;  // a new child grid

  CUstream_st* stream;
};

class gpgpu_context;

class cuda_device_runtime {
 public:
  cuda_device_runtime(gpgpu_context* ctx) {
    g_total_param_size = 0;
    g_max_total_param_size = 0;
    gpgpu_ctx = ctx;
  }
  unsigned long long g_total_param_size;
  std::map<void*, device_launch_config_t> g_cuda_device_launch_param_map;
  std::list<device_launch_operation_t> g_cuda_device_launch_op;
  unsigned g_kernel_launch_latency;
  unsigned g_TB_launch_latency;
  unsigned long long g_max_total_param_size;
  bool g_cdp_enabled;
  int g_max_sim_rt_kernels;
  // print the transactions vector in traceRay() function for all threads, in vulkan_ray_tracing.cc
  bool g_print_mem_transactions;
  // print the cache accesses in shader.cc
  bool g_print_cache_transactions;
  // Print CTA start/finish cycles
  bool g_print_cta_start_finish;
  // Print rt core arrive/complete cycles for all warps
  bool g_print_rt_start_finish;
  // Print per-warp rt latency distribution info
  bool g_print_rt_warp_latency_dist;  
  // print node parent map for rt
  bool g_print_node_parent_map;  
  // enable cooprt
  bool g_rt_coop_threads;
  // where do helper threads push child nodes? (should be enabled with cooprt)
  bool g_rt_coop_push_to_own_stack;
  // print rt timeline
  bool g_rt_print_timeline;
  // subwarp config for cooprt
  bool g_rt_coop_subwarp_config;

  // backward pointer
  class gpgpu_context* gpgpu_ctx;
#if (CUDART_VERSION >= 5000)
#pragma once
  void gpgpusim_cuda_launchDeviceV2(const ptx_instruction* pI,
                                    ptx_thread_info* thread,
                                    const function_info* target_func);
  void gpgpusim_cuda_streamCreateWithFlags(const ptx_instruction* pI,
                                           ptx_thread_info* thread,
                                           const function_info* target_func);
  void gpgpusim_cuda_getParameterBufferV2(const ptx_instruction* pI,
                                          ptx_thread_info* thread,
                                          const function_info* target_func);
  void launch_all_device_kernels();
  void launch_one_device_kernel();
#endif
};

#endif /* __cuda_device_runtime_h__  */
