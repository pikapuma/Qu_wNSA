#ifndef MAT_OPERATION
#define MAT_OPERATION

#include "QuBLAS.h"

#include <complex>
#include <cstddef>

template <typename Qu_scalar_t, size_t row, size_t col = row>
using MAT = Qu<dim<row, col>, Qu_scalar_t>;

template <typename Qu_scalar_t, size_t len>
using VEC = Qu<dim<len>, Qu_scalar_t>;

template <typename T>
concept is_mat_v = requires(T const& vec) { vec[0][0]; } || requires(T const& vec) { vec[0, 0]; };

template <typename T>
concept is_vec_v = requires(T const& vec) { vec[0]; } && (!is_mat_v<T>);

template <typename T>
concept is_tensor_v = is_vec_v<T> || is_mat_v<T>;

template <typename complex_t>
complex_t my_conj(complex_t const& in) {
  return {in.real, -in.imag};
}

template <typename T, size_t row, size_t col>
  requires(!is_vec_v<T>)
void elemwise(MAT<T, row, col> const& src, MAT<T, row, col>& result, auto&& pred) {
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      result[r, c] = pred(src[r, c]);
    }
  }
}

template <typename T, size_t len>
void complex_expand(VEC<std::complex<T>, len> const& src, VEC<T, 2 * len>& dest) {
  for (size_t i = 0; i < len; ++i) {
    dest[i]       = src[i].real;
    dest[i + len] = src[i].imag;
  }
}

template <typename T, size_t len>
  requires(!is_vec_v<T>)
void elemwise(VEC<T, len> const& src, VEC<T, len>& result, auto&& pred) {
  for (size_t r = 0; r < len; ++r) {
    result[r] = pred(src[r]);
  }
}

template <typename T, size_t row, size_t col>
  requires(!is_vec_v<T>)
void pairwise(MAT<T, row, col> const& lhs, MAT<T, row, col> const& rhs, MAT<T, col>& result, auto&& pred) {
  for (size_t r = 0; r < row; ++r) {
    for (size_t c = 0; c < col; ++c) {
      result[r, c] = pred(lhs[r, c], rhs[r, c]);
    }
  }
}

template <typename T, size_t len>
  requires(!is_vec_v<T>)
void pairwise(VEC<T, len> const& lhs, VEC<T, len> const& rhs, VEC<T, len>& result, auto&& pred) {
  for (size_t r = 0; r < len; ++r) {
    result[r] = pred(lhs[r], rhs[r]);
  }
}

template <typename T, size_t len>
  requires(!is_vec_v<T>)
void pairwise(T const lhs, VEC<T, len> const& rhs, VEC<T, len>& result, auto&& pred) {
  for (size_t r = 0; r < len; ++r) {
    result[r] = pred(lhs, rhs[r]);
  }
}

template <typename T, size_t row, size_t col>
  requires(!is_vec_v<T>)
void pairwise(T const lhs, MAT<T, row, col> const& rhs, MAT<T, col>& result, auto&& pred) {
  for (size_t r = 0; r < row; ++r) {
    for (size_t c = 0; c < col; ++c) {
      result[r, c] = pred(lhs, rhs[r, c]);
    }
  }
}

template <typename T, size_t row, size_t col>
void conj_mult(MAT<T, row, col> const& src, MAT<T, col>& result) {
  for (size_t r = 0; r < col; ++r) {
    for (size_t c = 0; c <= r; ++c) {
      result[r, c] = 0;

      for (size_t k = 0; k < row; ++k) {
        result[r, c] += my_conj(src[k, r]) * src[k, c];
      }

      result[c, r] = my_conj(result[r, c]);
    }
  }
}

template <typename T, size_t row1, size_t col1, size_t col2>
void conj_mult(MAT<T, row1, col1> const& lhs, MAT<T, row1, col2> const& rhs, MAT<T, col1, col2>& result) {
  for (size_t r = 0; r < col1; ++r) {
    for (size_t c = 0; c < col2; ++c) {
      result[r, c] = 0;

      for (size_t k = 0; k < row1; ++k) {
        result[r, c] += my_conj(lhs[k, r]) * rhs[k, c];
      }
    }
  }
}

template <typename T, size_t row, size_t col>
void conj_mult(MAT<T, row, col> const& lhs, VEC<T, row> const& rhs, VEC<T, col>& result) {
  for (size_t r = 0; r < col; ++r) {
    result[r] = 0;

    for (size_t k = 0; k < row; ++k) {
      result[r] += my_conj(lhs[k, r]) * rhs[k];
    }
  }
}

template <typename src_t, typename res_t, size_t row, size_t col>
void conj_mult(MAT<src_t, row, col> const& src, MAT<res_t, col>& result) {
  for (size_t r = 0; r < col; ++r) {
    for (size_t c = 0; c <= r; ++c) {
      result[r, c] = 0;

      for (size_t k = 0; k < row; ++k) {
        result[r, c] += Qmul<BasicComplexMul<acT<typename res_t::realType>,
                                             bdT<typename res_t::realType>,
                                             adT<typename res_t::realType>,
                                             bcT<typename res_t::realType>,
                                             acbdT<typename res_t::realType>,
                                             adbcT<typename res_t::realType>>>(my_conj(src[k, r]), src[k, c]);
      }

      result[c, r] = my_conj(result[r, c]);
    }
  }
}

template <typename T, size_t row1, size_t col1, size_t col2>
void mat_mult(MAT<T, row1, col1> const& lhs, MAT<T, col1, col2> const& rhs, MAT<T, row1, col2>& result) {
  for (size_t r = 0; r < row1; ++r) {
    for (size_t c = 0; c < col2; ++c) {
      result[r, c] = 0;

      for (size_t k = 0; k < col1; ++k) {
        result[r, c] += lhs[r, k] * rhs[k, c];
      }
    }
  }
}

template <typename T, size_t row, size_t col>
void mat_mult(MAT<T, row, col> const& lhs, VEC<T, col> const& rhs, VEC<T, row>& result) {
  for (size_t r = 0; r < row; ++r) {
    result[r] = 0;

    for (size_t c = 0; c < col; ++c) {
      result[r] += lhs[r, c] * rhs[c];
    }
  }
}

// μ(i+1) = b − W * μ(i)
template <size_t len, typename T>
void first_order_iter(
    MAT<T, len> const& factor, VEC<T, len> const& constant, VEC<T, len> const& init, VEC<T, len>& dest, int iter_num) {
  dest = init;
  MAT<T, len> one{};
  for (size_t i = 0; i < len; ++i) {
    one[i, i] = 1;
  }

  MAT<T, len> mult_mat = one - factor;
  VEC<T, len> mult_res;

  for (int i = 0; i < iter_num; ++i) {
    mat_mult<T, len, len>(mult_mat, dest, mult_res);
    dest = constant + mult_res;
  }
}

// μ(i+1) = μ(i−1) + (b − W * μ(i)) + k * (μ(i) − μ(i−1))
template <typename T, size_t len>
void second_order_iter(MAT<T, len> const& factor,    // W
                       VEC<T, len> const& constant,  // b
                       VEC<T, len> const& init0,     // μ^(0)
                       VEC<T, len> const& init1,     // μ^(1)
                       VEC<T, len>& dest,
                       T k,
                       int iter_num) {
  dest             = init1;
  VEC<T, len> prev = init0;
  VEC<T, len> tmp;
  VEC<T, len> mat_vec_mult_res;

  for (int i = 0; i < iter_num; ++i) {
    tmp = dest;

    mat_mult(factor, dest, mat_vec_mult_res);
    dest = prev + constant - mat_vec_mult_res + k * (dest - prev);

    prev = tmp;
  }
}

#endif