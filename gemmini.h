#ifndef _GEMMINI_H
#define _GEMMINI_H

#include <riscv/extension.h>
#include <riscv/rocc.h>
#include <random>
#include <limits>
#include <deque>
#include "gemmini_params.h"

typedef acc_t output_t; // Systolic array output datatype (coming down from PEs, moving into accumulator)
static const uint32_t sp_matrices = (BANK_NUM * BANK_ROWS) / DIM; // Size the scratchpad to fit sp_matrices matrices
static const uint32_t accum_rows = ACC_ROWS; // Number of systolic array rows in the accumulator
static const uint64_t addr_len = ADDR_LEN; // Number of bits used to address the scratchpad/accumulator
#define LOAD_STATES 3

#ifndef NORM_STAT_IDS
#define NORM_STAT_IDS 4
#endif

// WARNING: If you change this, you must also change the bits in the counter op config register decoding union in gemmini.cc.
#define NUM_COUNTERS 8
#define NUM_EXTERNAL_COUNTERS 6 

// changed: added. N:M sparsity pattern of the sparse weight-stationary mode.
// The Makefile passes -DNM_N/-DNM_M (default 2:4); it must match the chip (GEMMINI_HW_NM_N/M when the header has them).
#if !defined(NM_N) || !defined(NM_M)
#error "NM_N/NM_M undefined: build through the Makefile, e.g. make NM_N=4 NM_M=8"
#endif
#if defined(GEMMINI_HW_NM_N) && defined(GEMMINI_HW_NM_M) && (GEMMINI_HW_NM_N != NM_N || GEMMINI_HW_NM_M != NM_M)
#error "NM_N/NM_M disagree with GEMMINI_HW_NM_N/GEMMINI_HW_NM_M in gemmini_params.h"
#endif

constexpr unsigned nm_log2_ceil(unsigned x) { return x <= 1 ? 0 : 1 + nm_log2_ceil((x + 1) / 2); }
static_assert(NM_N >= 1 && NM_M == 2 * NM_N, "the sparse array needs m == 2*n");
static_assert((NM_M & (NM_M - 1)) == 0, "m must be a power of 2");
static_assert(DIM % NM_N == 0, "DIM must be a multiple of n");
static_assert(DIM * nm_log2_ceil(NM_M) <= 64, "an index row (DIM * log2(m) bits) must fit in 64 bits");

#define NM_META_FIFO_DEPTH 4 // index tiles the RTL FIFO holds: send at most this many tiles ahead of their preloads

// changed: Fix A. Scratchpad rows one DIM x DIM index tile fills, packed tight (DIM*DIM*log2(m) bits over
// DIM*8*sizeof(elem_t)-bit rows, rounded up): 4 for 2:4, 6 for 4:8 at DIM 16 int8 (fix-a-spec.md section 3).
#define NM_SPAD_ROWS_PER_TILE_CALC ((DIM * DIM * nm_log2_ceil(NM_M) + DIM * 8 * sizeof(elem_t) - 1) / (DIM * 8 * sizeof(elem_t)))
static const size_t nm_spad_rows_per_tile = NM_SPAD_ROWS_PER_TILE_CALC;
#ifdef GEMMINI_HW_NM_SPAD_ROWS_PER_TILE
static_assert(GEMMINI_HW_NM_SPAD_ROWS_PER_TILE == NM_SPAD_ROWS_PER_TILE_CALC,
              "GEMMINI_HW_NM_SPAD_ROWS_PER_TILE in gemmini_params.h disagrees with the tight index tile packing");
#endif

#define MAKECUSTOMFN(opcode) custom ## opcode
#define CUSTOMFN(opcode) MAKECUSTOMFN(opcode)

struct gemmini_state_t
{
  enum Dataflow {OS, WS};
  enum Activation {NONE, RELU, LAYERNORM, IGELU, SOFTMAX};
  enum NormCmd {RESET, SUM, MEAN, VARIANCE, INV_STDDEV, MAX, SUM_EXP, INV_SUM_EXP};
  void reset();

  // 32-bit gemmini address space
  uint32_t output_sp_addr;
  uint32_t preload_sp_addr;
  uint16_t preload_cols, preload_rows;
  uint16_t output_cols, output_rows;
  Dataflow mode;
  Activation sys_act;
  Activation acc_act;
  reg_t sys_shift;
  acc_t igelu_qb, igelu_qc;
  acc_t qln2, qln2_inv;
  reg_t load_strides[LOAD_STATES];
  reg_t store_stride;
  uint16_t load_block_strides[LOAD_STATES];
  bool load_shrunks[LOAD_STATES];
#if defined(HAS_MVIN_SCALE) || defined(HAS_MVIN_ACC_SCALE)
  scale_t load_scales[LOAD_STATES];
#endif
  uint8_t pixels_per_rows[LOAD_STATES];
  acc_scale_t acc_shift;
  acc_scale_t sys_acc_shift;
  uint16_t c_stride;
  uint16_t a_stride;
  uint8_t pool_stride;
  uint8_t pool_size;
  uint8_t pool_out_dim;
  uint8_t pool_porows;
  uint8_t pool_pocols;
  uint8_t pool_orows;
  uint8_t pool_ocols;
  uint8_t pool_lpad;
  uint8_t pool_upad;
  bool a_transpose;
  bool b_transpose;
  uint16_t loop_ws_I, loop_ws_J, loop_ws_K;
  uint16_t loop_ws_pad_I, loop_ws_pad_J, loop_ws_pad_K;
  uint64_t loop_ws_A, loop_ws_B, loop_ws_D, loop_ws_C;
  uint64_t loop_ws_A_stride, loop_ws_B_stride, loop_ws_D_stride, loop_ws_C_stride;
  uint16_t loop_conv_ws_batch_size, loop_conv_ws_in_row_dim, loop_conv_ws_in_col_dim, loop_conv_ws_in_channels, loop_conv_ws_out_channels;
  uint16_t loop_conv_ws_in_stride, loop_conv_ws_weight_stride, loop_conv_ws_out_stride;
  uint16_t loop_conv_ws_out_row_dim, loop_conv_ws_pool_out_row_dim, loop_conv_ws_out_col_dim, loop_conv_ws_pool_out_col_dim, loop_conv_ws_stride, loop_conv_ws_padding;
  uint16_t loop_conv_ws_kernel_dim, loop_conv_ws_pool_size, loop_conv_ws_pool_stride, loop_conv_ws_pool_padding;
  uint16_t loop_conv_ws_batches, loop_conv_ws_porows, loop_conv_ws_pocols, loop_conv_ws_pochs;
  uint16_t loop_conv_ws_krows, loop_conv_ws_kcols, loop_conv_ws_kchs, loop_conv_ws_lpad;
  uint16_t loop_conv_ws_rpad, loop_conv_ws_upad, loop_conv_ws_dpad, loop_conv_ws_plpad;
  uint16_t loop_conv_ws_prad, loop_conv_ws_pupad, loop_conv_ws_pdpad, loop_conv_ws_orows;
  uint16_t loop_conv_ws_ocols, loop_conv_ws_kernel_dilation;
  uint64_t loop_conv_ws_input, loop_conv_ws_weights, loop_conv_ws_output, loop_conv_ws_bias;

  // Normalization statistics
  uint8_t norm_stat_id;
  acc_t norm_sum[NORM_STAT_IDS];
  acc_t norm_running_max[NORM_STAT_IDS];
  acc_t norm_max[NORM_STAT_IDS];
  acc_t norm_count[NORM_STAT_IDS];
  acc_t norm_mean[NORM_STAT_IDS];
  acc_scale_t norm_inv_stddev[NORM_STAT_IDS];
  acc_scale_t norm_inv_sum_exp[NORM_STAT_IDS];
  bool norm_reset[NORM_STAT_IDS];

  // Counter
  uint32_t counter_val[NUM_COUNTERS];
  uint32_t counter_snapshot_val[NUM_COUNTERS];
  uint16_t counter_config[NUM_COUNTERS];
  uint32_t counter_external[NUM_EXTERNAL_COUNTERS];
  bool counter_external_flag[NUM_COUNTERS];
  bool snapshot_enable;
  bool op_in_progress;

  bool enable;
  bool resetted = false;

  std::vector<std::vector<elem_t>> spad; // Scratchpad constructed as systolic array rows
  std::vector<std::vector<acc_t>> pe_state; // Stores each PE's internal accumulator state
  std::vector<std::vector<acc_t>> accumulator;

  // changed: added. N:M sparse state.
  // Index commands fill nm_asm row by row; whole tiles queue in nm_meta_q and each real sparse preload pops one.
  // The preload command latches its weights, indexes and mode (preload_*); the next compute_preloaded moves
  // them into the PEs. nm_buf_sparse is the mode latched per RTL weight buffer; nm_buf is the one in use.
  bool nm_sparse;
  std::vector<std::vector<uint8_t>> nm_asm;
  size_t nm_asm_row;
  std::deque<std::vector<std::vector<uint8_t>>> nm_meta_q;
  std::vector<std::vector<uint8_t>> pe_meta;
  std::vector<std::vector<uint8_t>> preload_meta;
  std::vector<std::vector<elem_t>> preload_d;
  bool preload_real; // the last preload was not GARBAGE
  bool preload_nm_sparse;
  bool nm_buf_sparse[2];
  unsigned nm_buf;
  bool nm_tile_in_flight; // a real sparse preload popped a tile; like the RTL it counts against the FIFO until its compute
  bool nm_ld_since_clear; // changed: Fix A, a scratchpad index load (rd == 2) since the last clear: register rows illegal
  uint64_t loop_ws_idx_dram; // changed: sparse-loop-fix, DRAM address of index tile 0 of the next sparse loop (funct 28)
  bool loop_ws_nm_cfg_seen; // changed: sparse-loop-fix, a funct 28 since the last LOOP_WS (asserts N3, N4)

  // cisc state
  reg_t a_addr, b_addr, c_addr, d_addr;
  reg_t m, n, k;
  bool repeating_bias;
};

class gemmini_t : public extension_t
{
public:
  gemmini_t() : cause(0), aux(0), debug(false) {}
  const char* name() const override { return "gemmini"; }

  reg_t CUSTOMFN(XCUSTOM_ACC)(rocc_insn_t insn, reg_t xs1, reg_t xs2);
  void reset();
  void set_processor(processor_t* p) { this->p = p; }

  void mvin(reg_t dram_addr, reg_t sp_addr, int state_id);
  void mvout(reg_t dram_addr, reg_t sp_addr);
  void preload(reg_t bd_addr, reg_t c_addr);
  void config(reg_t rs1, reg_t rs2);
  void compute(reg_t a_addr, reg_t bd_addr, bool preload);
  void compute_cisc();
  void nm_meta_load(reg_t rs1, reg_t rs2); // changed: added
  void nm_meta_clear(); // changed: added
  void nm_meta_load_spad(reg_t rs1, reg_t rs2); // changed: Fix A
  void load_idx(reg_t dram_addr, reg_t sp_addr); // changed: sparse-loop-fix, funct 27, mvin of index rows with fixed config
  void loop_ws_config_nm(reg_t rs1, reg_t rs2); // changed: sparse-loop-fix, funct 28
  reg_t counter_operation(reg_t rs1);

  void loop_ws(reg_t rs1, reg_t rs2);
  void loop_ws_config_bounds(reg_t rs1, reg_t rs2);
  void loop_ws_config_addrs_AB(reg_t rs1, reg_t rs2);
  void loop_ws_config_addrs_DC(reg_t rs1, reg_t rs2);
  void loop_ws_config_strides_AB(reg_t rs1, reg_t rs2);
  void loop_ws_config_strides_DC(reg_t rs1, reg_t rs2);

  void loop_conv_ws(reg_t rs1, reg_t rs2);
  void loop_conv_ws_config_1(reg_t rs1, reg_t rs2);
  void loop_conv_ws_config_2(reg_t rs1, reg_t rs2);
  void loop_conv_ws_config_3(reg_t rs1, reg_t rs2);
  void loop_conv_ws_config_4(reg_t rs1, reg_t rs2);
  void loop_conv_ws_config_5(reg_t rs1, reg_t rs2);
  void loop_conv_ws_config_6(reg_t rs1, reg_t rs2);

  virtual std::vector<insn_desc_t> get_instructions(const processor_t &p) override;
  virtual std::vector<disasm_insn_t*> get_disasms(const processor_t *p) override;

private:
  gemmini_state_t gemmini_state;
  reg_t cause;
  reg_t aux;
  processor_t* p;

  const unsigned config_funct = 0;
  const unsigned mvin_funct = 2;
  const unsigned mvin2_funct = 1;
  const unsigned mvin3_funct = 14;
  const unsigned mvout_funct = 3;
  const unsigned compute_preloaded_funct = 4;
  const unsigned compute_accumulated_funct = 5;
  const unsigned preload_funct = 6;
  const unsigned flush_funct = 7;
  const unsigned nm_meta_funct = 26; // changed: added, N:M index rows (rd == nm_meta_clear_rd: clear instead)
  const unsigned nm_meta_clear_rd = 1; // changed: added
  const unsigned nm_meta_push_rd = 0; // changed: Fix A, register rows (the original form)
  const unsigned nm_meta_load_spad_rd = 2; // changed: Fix A, index tiles from the scratchpad; rd 3..31 reserved
  const unsigned load_idx_funct = 27; // changed: sparse-loop-fix, LOAD_IDX_CMD (fix-b-spec.md 2.5.2)
  const unsigned loop_ws_config_nm_funct = 28; // changed: sparse-loop-fix, LOOP_WS_CONFIG_NM (fix-b-spec.md 2.5.2)
  const unsigned loop_ws_funct = 8;
  const unsigned loop_ws_config_bounds_funct = 9;
  const unsigned loop_ws_config_addrs_AB_funct = 10;
  const unsigned loop_ws_config_addrs_DC_funct = 11;
  const unsigned loop_ws_config_strides_AB_funct = 12;
  const unsigned loop_ws_config_strides_DC_funct = 13;

  const unsigned loop_conv_ws_funct = 15;
  const unsigned loop_conv_ws_config_1_funct = 16;
  const unsigned loop_conv_ws_config_2_funct = 17;
  const unsigned loop_conv_ws_config_3_funct = 18;
  const unsigned loop_conv_ws_config_4_funct = 19;
  const unsigned loop_conv_ws_config_5_funct = 20;
  const unsigned loop_conv_ws_config_6_funct = 21;

  const unsigned fence_funct = 127;

  //==========================================================================
  // gemmini-cisc opcodes
  //==========================================================================
  const unsigned config_cisc_ex_funct        = 10;
  const unsigned config_addr_AB_funct        = 11;
  const unsigned config_addr_CD_funct        = 12;
  const unsigned config_size0_funct          = 13;
  const unsigned config_size1_funct          = 14;
  const unsigned config_repeating_bias_funct = 15;
  const unsigned config_reset_funct          = 16;
  const unsigned compute_cisc_funct          = 17;
  const unsigned counter_op_funct            = 126;
  //==========================================================================

  bool debug;
  elem_t apply_activation(elem_t value, enum gemmini_state_t::Activation act);
  elem_t apply_activation_sys(elem_t value);
  elem_t apply_activation_acc(elem_t value);
  acc_t apply_pre_activation_acc(acc_t value);
  bool apply_norm(const acc_t * x, size_t len, enum gemmini_state_t::NormCmd cmd);

  enum gemmini_state_t::NormCmd non_terminating_norm_cmd(enum gemmini_state_t::NormCmd cmd);

#ifdef HAS_MVIN_SCALE
  elem_t mvin_scale(elem_t value, scale_t scale);
#endif

#ifdef HAS_MVIN_ACC_SCALE
  acc_t mvin_scale_acc(acc_t value, scale_acc_t scale);
#endif

  elem_t acc_scale(acc_t value, acc_scale_t acc);
  elem_t sys_shift(output_t value, unsigned int shift);

  template <class T>
  T read_from_dram(reg_t addr);

  template <class T>
  std::vector<std::vector<T>> *
  read_matrix_from_dram(reg_t addr, reg_t rows, reg_t cols, 
                        bool zeroable, bool repeating_bias);

  template <class T>
  void write_to_dram(reg_t addr, T data);

#ifdef ELEM_T_IS_FLOAT
  elem_t elem_t_bits_to_elem_t(elem_t_bits x);
  elem_t_bits elem_t_to_elem_t_bits(elem_t x);
  acc_t acc_t_bits_to_acc_t(acc_t_bits x);
  acc_t_bits acc_t_to_acc_t_bits(acc_t x);
#endif

#if defined(HAS_MVIN_SCALE) || defined(HAS_MVIN_ACC_SCALE)
  scale_t_bits scale_t_to_scale_t_bits(scale_t scale);
  scale_t scale_t_bits_to_scale_t(scale_t_bits bits);
#endif

  acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t scale);
  acc_scale_t acc_scale_t_bits_to_acc_scale_t(acc_scale_t_bits bits);

  void counter_increment(unsigned int counter_id);
  void counter_increment_random();
};

#endif
