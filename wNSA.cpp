#include "MIMO_EPA.hpp"
#include "QuBLAS.h"
#include "mat_operation.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>

template <size_t data_width, size_t frac_width>
using fixed_t = Qu<isSigned<true>, intBits<data_width - frac_width - 1>, fracBits<frac_width>, OfMode<WRP::TCPL>>;

constexpr size_t H_DATA_WIDTH = 14;
constexpr size_t H_FRAC_WIDTH = 10;
constexpr size_t y_DATA_WIDTH = 16;
constexpr size_t y_FRAC_WIDTH = 10;

constexpr size_t A_DIAG_DATA_WIDTH = 18;
constexpr size_t A_DIAG_FRAC_WIDTH = 10;
constexpr size_t A_OFF_DATA_WIDTH  = 18;
constexpr size_t A_OFF_FRAC_WIDTH  = 12;
constexpr size_t A_FULL_FRAC_WIDTH = std::max(A_DIAG_FRAC_WIDTH, A_OFF_FRAC_WIDTH);
constexpr size_t A_FULL_DATA_WIDTH
    = A_FULL_FRAC_WIDTH + std::max(A_DIAG_DATA_WIDTH - A_DIAG_FRAC_WIDTH, A_OFF_DATA_WIDTH - A_OFF_FRAC_WIDTH);

constexpr size_t b_DATA_WIDTH = 20;
constexpr size_t b_FRAC_WIDTH = 8;

constexpr size_t wNSA_DATA_WIDTH = 18;
constexpr size_t wNSA_FRAC_WIDTH = 12;
constexpr size_t EPA_DATA_WIDTH  = 18;
constexpr size_t EPA_FRAC_WIDTH  = 12;

using H_t           = fixed_t<H_DATA_WIDTH, H_FRAC_WIDTH>;
using y_t           = fixed_t<y_DATA_WIDTH, y_FRAC_WIDTH>;
using A_full_prec_t = fixed_t<A_FULL_DATA_WIDTH, A_FULL_FRAC_WIDTH>;
using A_diag_t      = fixed_t<A_DIAG_DATA_WIDTH, A_DIAG_FRAC_WIDTH>;
using A_off_t       = fixed_t<A_OFF_DATA_WIDTH, A_OFF_FRAC_WIDTH>;
using b_t           = fixed_t<b_DATA_WIDTH, b_FRAC_WIDTH>;
using wNSA_t        = fixed_t<wNSA_DATA_WIDTH, wNSA_FRAC_WIDTH>;
using EPA_t         = fixed_t<EPA_DATA_WIDTH, EPA_FRAC_WIDTH>;
using complex       = std::complex<double>;

constexpr size_t Tx            = 32;
constexpr size_t Rx            = 64;
constexpr size_t QAM           = 64;
constexpr size_t wNSA_ITER_NUM = 25;
constexpr size_t EPA_ITER_NUM  = 8;
constexpr size_t test_num      = 20000;

constexpr double SNR = 18;
constexpr double TxE = Tx * Rx;
double const Nv      = TxE / (std::pow(10, SNR / 10) * std::log2(QAM) * Tx);
double const Nv_r    = Nv / 2;
constexpr wNSA_t w   = 0.6;

fixed_t<b_DATA_WIDTH, b_FRAC_WIDTH> const sqrt_Es0 = std::sqrt(get_Es0<QAM>());

int main() {
  MAT<Qu<H_t, H_t>, Rx, Tx> H{};
  std::array<complex, Tx> x{};
  VEC<Qu<y_t, y_t>, Rx> y{};

  VEC<EPA_t, 2 * Tx> t;
  VEC<EPA_t, 2 * Tx> sym_res;
  uint err_cnt = 0;

  size_t bit_per_sym = log2_floor<QAM>() / 2;
  std::string x_bits;
  x_bits.resize(bit_per_sym * 2 * Tx);
  std::string res_bits;
  res_bits.resize(bit_per_sym * 2 * Tx);

  for (size_t i = 0; i < test_num; ++i) {
    MIMO_system<Tx, Rx, QAM, H_t, y_t>(H, y, x, x_bits, Nv_r);

    MAT<wNSA_t, 2 * Tx> A_wNSA{};
    VEC<wNSA_t, 2 * Tx> b_wNSA{};
    MAT<wNSA_t, 2 * Tx> A_EPA_init{};
    VEC<wNSA_t, 2 * Tx> b_EPA_init{};
    VEC<wNSA_t, 2 * Tx> mu_long{};

    init_params_wNSA<Tx, Rx, H_t, y_t, A_full_prec_t, A_diag_t, A_off_t, b_t, wNSA_t>(
        H, y, A_wNSA, b_wNSA, A_EPA_init, b_EPA_init, w, sqrt_Es0, A_diag_t(Nv_r));

    wNSA_iter<2 * Tx, wNSA_t>(A_wNSA, b_wNSA, b_wNSA, mu_long, wNSA_ITER_NUM);

    MAT<EPA_t, 2 * Tx> A_EPA_short;
    VEC<EPA_t, 2 * Tx> b_EPA_short;
    VEC<EPA_t, 2 * Tx> mu_short;

    for (size_t r = 0; r < 2 * Tx; ++r) {
      for (size_t c = 0; c < 2 * Tx; ++c) {
        A_EPA_short[r, c] = A_EPA_init[r, c];
      }
      b_EPA_short[r] = b_EPA_init[r];
      mu_short[r]    = mu_long[r];
    }

    EPA<QAM, EPA_t, Tx>(A_EPA_short, b_EPA_short, mu_short, t, EPA_ITER_NUM);

    line_sym_decision<QAM, EPA_t, Tx>(t, sym_res, res_bits);
    err_cnt += str_diff_cnt(x_bits, res_bits);

    if (i > 0 && i % 1000 == 0) {
      double BER = 1.0 * err_cnt / static_cast<double>(bit_per_sym * 2 * Tx) / i;
      std::cout << "BER = " << std::scientific << BER << "  @runs=" << i << '\n';
    }
  }

  std::cout << "H_width = Q" << (H_DATA_WIDTH - H_FRAC_WIDTH) << '.' << H_FRAC_WIDTH;
  std::cout << ",   y_width = Q" << (y_DATA_WIDTH - y_FRAC_WIDTH) << '.' << y_FRAC_WIDTH;
  std::cout << ",   wNSA_width = Q" << (wNSA_DATA_WIDTH - wNSA_FRAC_WIDTH) << '.' << wNSA_FRAC_WIDTH;
  std::cout << ",   EPA_width = Q" << (EPA_DATA_WIDTH - wNSA_FRAC_WIDTH) << '.' << EPA_FRAC_WIDTH << '\n';

  double BER = 1.0 * err_cnt / static_cast<double>(bit_per_sym * 2 * Tx) / test_num;
  std::cout << "BER = " << std::scientific << BER << "  @runs=" << test_num << '\n';

  return 0;
}