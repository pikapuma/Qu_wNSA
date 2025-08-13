#ifndef MIMO_EPA
#define MIMO_EPA

#include "QuBLAS.h"
#include "mat_operation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>

static std::mt19937 generator{std::random_device{}()};  // NOLINT
static std::normal_distribution<double> normal{0, 1};   // NOLINT

// get the average energy of QAM
template <size_t QAM>
consteval double get_Es0() {
  if constexpr (QAM == 16) {
    return 10;
  } else if constexpr (QAM == 64) {
    return 42;
  } else if constexpr (QAM == 256) {
    return 170;
  } else {
    static_assert(false, "Invalid QAM!");
  }
}

// get symbols of QAM
template <size_t QAM>
consteval auto get_symbols() {
  if constexpr (QAM == 16) {
    return std::array<int, 4>{-3, -1, 1, 3};
  } else if constexpr (QAM == 64) {
    return std::array<int, 8>{-7, -5, -3, -1, 1, 3, 5, 7};
  } else if constexpr (QAM == 256) {
    return std::array<int, 16>{-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15};
  } else {
    static_assert(false, "Invalid QAM!");
  }
}

// get bit sequences(real part) of QAM
template <size_t QAM>
auto get_bits_r() {
  if constexpr (QAM == 16) {
    return std::array<std::string, 4>{"00", "01", "11", "10"};
  } else if constexpr (QAM == 64) {
    return std::array<std::string, 8>{"000", "001", "011", "010", "110", "111", "101", "100"};
  } else if constexpr (QAM == 256) {
    return std::array<std::string, 16>{"0000",
                                       "0001",
                                       "0011",
                                       "0010",
                                       "0110",
                                       "0111",
                                       "0101",
                                       "0100",
                                       "1100",
                                       "1101",
                                       "1111",
                                       "1110",
                                       "1010",
                                       "1011",
                                       "1001",
                                       "1000"};
  } else {
    static_assert(false, "Invalid QAM!");
  }
}

// get bit sequences(imag part) of QAM
template <size_t QAM>
auto get_bits_i() {
  if constexpr (QAM == 16) {
    return std::array<std::string, 4>{"10", "11", "01", "00"};
  } else if constexpr (QAM == 64) {
    return std::array<std::string, 8>{"100", "101", "111", "110", "010", "011", "001", "000"};
  } else if constexpr (QAM == 256) {
    return std::array<std::string, 16>{"1000",
                                       "1001",
                                       "1011",
                                       "1010",
                                       "1110",
                                       "1111",
                                       "1101",
                                       "1100",
                                       "0100",
                                       "0101",
                                       "0111",
                                       "0110",
                                       "0010",
                                       "0011",
                                       "0001",
                                       "0000"};
  } else {
    static_assert(false, "Invalid QAM!");
  }
}

// consteval function for log2
template <size_t num>
consteval size_t log2_floor() {
  size_t res    = 0;
  size_t shiftr = num;

  while ((shiftr >> 1U) > 0) {
    shiftr >>= 1U;
    res     += 1;
  }

  return res;
}

// template variables for MIMO configuration
template <size_t QAM>
constexpr auto symbols = get_symbols<QAM>();

template <size_t QAM>
auto const bits_r = get_bits_r<QAM>();

template <size_t QAM>
auto const bits_i = get_bits_i<QAM>();

// get the index of closest symbol in the symbol set
template <size_t QAM, typename T>
size_t get_symbol_id(T sym) {
  int const Mc = log2_floor<QAM>();

  for (size_t i = 0; i < Mc; ++i) {
    if constexpr (std::is_arithmetic_v<T>) {
      if (symbols<QAM>[i] + 1 > sym) {
        return i;
      }
    } else {
      if (symbols<QAM>[i] + 1 > sym.toDouble()) {
        return i;
      }
    }
  }

  return Mc - 1;  // 如果没有找到，返回-1
}

// symbol to bit sequences
template <size_t QAM>
std::string sym2bit(int sym, bool is_real) {
  int id = std::find(symbols<QAM>.begin(), symbols<QAM>.end(), sym);

  if (is_real) {
    return bits_r<QAM>[id];
  }

  return bits_i<QAM>[id];
}

// MIMO system, y=H*x+n
template <size_t Tx, size_t Rx, size_t QAM, typename H_t, typename H_RND_t, typename y_t, typename y_RND_t>
void MIMO_system(MAT<Qu<H_t, H_t>, Rx, Tx>& H,
                 VEC<Qu<y_t, y_t>, Rx>& y,
                 std::array<std::complex<double>, Tx>& x,
                 std::string& x_bits,
                 double Nv_real = 0) {
  using complex = std::complex<double>;

  double Es0      = get_Es0<QAM>();
  int bit_per_sym = log2_floor<QAM>() / 2;

  std::uniform_int_distribution<int> dist(0, std::log2(QAM) - 1);  // [0, log2(QAM)-1]之间随机整数生成器

  std::array<std::array<complex, Tx>, Rx> H_full{};
  std::array<complex, Rx> y_full{};

  Qu<H_RND_t, H_RND_t> H_tmp;
  Qu<y_RND_t, y_RND_t> y_tmp;

  for (size_t r = 0; r < Rx; ++r) {
    for (size_t c = 0; c < Tx; ++c) {
      H_full[r][c] = complex{normal(generator) * std::sqrt(0.5), normal(generator) * std::sqrt(0.5)};
      H_tmp        = H_full[r][c];
      H[r, c]      = H_tmp;
    }
  }

  for (size_t i = 0; i < Tx; ++i) {
    int x_r_id = dist(generator);
    int x_i_id = dist(generator);

    x[i] = complex{double(symbols<QAM>[x_r_id]), double(symbols<QAM>[x_i_id])} / std::sqrt(Es0);
    x_bits.replace(2 * i * bit_per_sym, bit_per_sym, bits_r<QAM>[x_r_id]);
    x_bits.replace((2 * i + 1) * bit_per_sym, bit_per_sym, bits_i<QAM>[x_i_id]);
  }

  for (size_t r = 0; r < Rx; ++r) {
    y_full[r] = complex{normal(generator) * std::sqrt(Nv_real), normal(generator) * std::sqrt(Nv_real)};

    for (size_t c = 0; c < Tx; ++c) {
      y_full[r] += H_full[r][c] * x[c];
    }

    y_tmp = y_full[r];
    y[r]  = y_tmp;
  }
}

// The preprocessing unit for EPA-wNSA algorithm
template <size_t Tx,
          size_t Rx,
          typename H_t,
          typename y_t,
          typename A_full_prec_t,
          typename A_diag_t,
          typename A_off_t,
          typename Dinv_t,
          typename buffer_t,
          typename b_full_prec_t,
          typename b_t,
          typename b_Es_t,
          typename wNSA_iter_t,
          typename EPA_iter_t>
void init_params_wNSA(MAT<Qu<H_t, H_t>, Rx, Tx> const& H,
                      VEC<Qu<y_t, y_t>, Rx> const& y,
                      MAT<buffer_t, 2 * Tx>& A_wNSA,
                      VEC<wNSA_iter_t, 2 * Tx>& b_wNSA,
                      MAT<buffer_t, 2 * Tx>& A_EPA,
                      VEC<EPA_iter_t, 2 * Tx>& b_EPA,
                      wNSA_iter_t k,
                      b_t sqrt_Es0,
                      A_diag_t alpha) {
  MAT<Qu<A_full_prec_t, A_full_prec_t>, Tx> A_full_prec{};
  VEC<Qu<b_t, b_t>, Tx> b{};

  conj_mult<Qu<H_t, H_t>, Qu<A_full_prec_t, A_full_prec_t>, Rx, Tx>(H, A_full_prec);
  for (size_t r = 0; r < Tx; r = r + 1) {
    for (size_t c = 0; c < Tx; c = c + 1) {
      if (r != c) {
        A_full_prec[r, c] = A_full_prec[r, c] + Qu<A_full_prec_t, A_full_prec_t>{1 / 8, 1 / 8};
      }
    }
  }

  Qu<b_full_prec_t, b_full_prec_t> b_tmp;
  for (size_t r = 0; r < Tx; ++r) {
    b_tmp = 0;

    for (size_t _k = 0; _k < Rx; ++_k) {
      b_tmp += Qmul<BasicComplexMul<acT<b_full_prec_t>,
                                    bdT<b_full_prec_t>,
                                    adT<b_full_prec_t>,
                                    bcT<b_full_prec_t>,
                                    acbdT<b_full_prec_t>,
                                    adbcT<b_full_prec_t>>>(my_conj(H[_k, r]), y[_k]);
    }

    b[r] = b_tmp + Qu<b_full_prec_t, b_full_prec_t>{1 / 32, 1 / 32};
  }

  for (size_t r = 0; r < Tx; ++r) {
    A_diag_t diag_elem = A_diag_t(A_full_prec[r, r].real) + alpha;

    int64_t dividend  = 1UZ << (A_diag_t::fracB + Dinv_t::fracB);
    int64_t divisor   = diag_elem.toDouble() * std::pow(2, A_diag_t::fracB);
    int64_t quot      = dividend / divisor;
    Dinv_t diag_inv   = 1.0 * quot / std::pow(2, Dinv_t::fracB);
    Dinv_t k_diag_inv = k * diag_inv;

    b_wNSA[r]      = Qmul<wNSA_iter_t>(Qmul<b_Es_t>(b[r].real, sqrt_Es0), k_diag_inv);
    b_wNSA[r + Tx] = Qmul<wNSA_iter_t>(Qmul<b_Es_t>(b[r].imag, sqrt_Es0), k_diag_inv);

    b_EPA[r]      = Qmul<EPA_iter_t>(Qmul<b_Es_t>(b[r].real, sqrt_Es0), diag_inv);
    b_EPA[r + Tx] = Qmul<EPA_iter_t>(Qmul<b_Es_t>(b[r].imag, sqrt_Es0), diag_inv);

    for (size_t c = 0; c < Tx; ++c) {
      buffer_t div_real;
      buffer_t div_imag;
      buffer_t k_div_real;
      buffer_t k_div_imag;

      if (r == c) {
        div_real   = A_diag_t(A_full_prec[r, c].real) * diag_inv;
        div_imag   = A_diag_t(A_full_prec[r, c].imag) * diag_inv;
        k_div_real = A_diag_t(A_full_prec[r, c].real) * k_diag_inv;
        k_div_imag = A_diag_t(A_full_prec[r, c].imag) * k_diag_inv;
      } else {
        div_real   = A_off_t(A_full_prec[r, c].real) * diag_inv;
        div_imag   = A_off_t(A_full_prec[r, c].imag) * diag_inv;
        k_div_real = A_off_t(A_full_prec[r, c].real) * k_diag_inv;
        k_div_imag = A_off_t(A_full_prec[r, c].imag) * k_diag_inv;
      }

      A_EPA[r, c]           = div_real;
      A_EPA[r, c + Tx]      = -div_imag;
      A_EPA[r + Tx, c]      = div_imag;
      A_EPA[r + Tx, c + Tx] = div_real;

      A_wNSA[r, c]           = k_div_real;
      A_wNSA[r, c + Tx]      = -k_div_imag;
      A_wNSA[r + Tx, c]      = k_div_imag;
      A_wNSA[r + Tx, c + Tx] = k_div_real;
    }
  }
}

// hard decision function
template <size_t QAM, typename T>
inline int hard_decision(T x) {
  size_t id = get_symbol_id<QAM>(x);

  return symbols<QAM>[id];
}

// EPA iteration
template <size_t QAM, typename EPA_t, typename tmp_t, size_t Tx>
void EPA(MAT<EPA_t, 2 * Tx> const& A_EPA,
         VEC<EPA_t, 2 * Tx> const& b_EPA,
         VEC<EPA_t, 2 * Tx> const& mu,
         VEC<EPA_t, 2 * Tx>& t,
         size_t iter_num) {
  using vec_t = VEC<EPA_t, 2 * Tx>;

  t = mu;
  vec_t eta{};
  vec_t mult_res;

  // t^(i) = (b - A * eta^(i) + eta^(i) + t^(i - 1)) / 2
  for (size_t i = 0; i < iter_num; ++i) {
    for (size_t r = 0; r < 2 * Tx; ++r) {
      eta[r] = hard_decision<QAM>(t[r]);
    }

    mat_mult<EPA_t, tmp_t, 2 * Tx, 2 * Tx>(A_EPA, eta, mult_res);

    t = vec_t(vec_t(b_EPA - mult_res) + eta) + t;
    t = t * EPA_t(0.5);
  }
}

// functions for bit sequences to symbols and error counting
template <size_t QAM, typename T, size_t Tx>
void line_sym_decision(VEC<T, 2 * Tx> const& t, VEC<T, 2 * Tx>& syms, std::string& bits) {
  int bit_per_sym = log2_floor<QAM>() / 2;

  for (size_t i = 0; i < Tx; ++i) {
    int x_r_id = get_symbol_id<QAM>(t[i]);
    int x_i_id = get_symbol_id<QAM>(t[i + Tx]);

    syms[i]      = symbols<QAM>[x_r_id];
    syms[i + Tx] = symbols<QAM>[x_i_id];

    bits.replace(2 * i * bit_per_sym, bit_per_sym, bits_r<QAM>[x_r_id]);
    bits.replace((2 * i + 1) * bit_per_sym, bit_per_sym, bits_i<QAM>[x_i_id]);
  }
}

inline uint str_diff_cnt(std::string_view lhs, std::string_view rhs) {
  size_t len = lhs.length();
  assert(len == rhs.length());

  uint cnt = 0;
  for (size_t i = 0; i < len; ++i) {
    if (lhs[i] != rhs[i]) {
      cnt++;
    }
  }

  return cnt;
}

template <size_t QAM, typename T, size_t Tx>
uint err_bit_cnt(VEC<T, 2 * Tx> const& t, std::string const& true_bits) {
  VEC<T, 2 * Tx> sim_syms;
  std::string sim_bits;
  sim_bits.resize(true_bits.size());
  line_sym_decision<QAM, T, Tx>(t, sim_syms, sim_bits);

  return str_diff_cnt(true_bits, sim_bits);
}

#endif