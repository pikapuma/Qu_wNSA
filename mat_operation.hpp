#ifndef MAT_OPERATION
#define MAT_OPERATION

#include "QuBLAS.h"

// type alias
template <typename Qu_scalar_t, size_t row, size_t col = row>
using MAT = Qu<dim<row, col>, Qu_scalar_t>;

template <typename Qu_scalar_t, size_t len>
using VEC = Qu<dim<len>, Qu_scalar_t>;

// complex conjugate
template <typename complex_t>
complex_t my_conj(complex_t const& in) {
  return {in.real, -in.imag};
}

// process a tensor elementwise
// (maybe add begin() and end() to Qu tensor and use std::views::transform()?)
template <typename T, size_t row, size_t col>
void elemwise(MAT<T, row, col> const& src, MAT<T, row, col>& result, auto&& op) {
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      result[r, c] = op(src[r, c]);
    }
  }
}

template <typename T, size_t len>
void elemwise(VEC<T, len> const& src, VEC<T, len>& result, auto&& op) {
  for (size_t r = 0; r < len; ++r) {
    result[r] = op(src[r]);
  }
}

// process two tensors pairwise
template <typename T, size_t row, size_t col>
void pairwise(MAT<T, row, col> const& lhs, MAT<T, row, col> const& rhs, MAT<T, col>& result, auto&& op) {
  for (size_t r = 0; r < row; ++r) {
    for (size_t c = 0; c < col; ++c) {
      result[r, c] = op(lhs[r, c], rhs[r, c]);
    }
  }
}

template <typename T, size_t len>
void pairwise(VEC<T, len> const& lhs, VEC<T, len> const& rhs, VEC<T, len>& result, auto&& op) {
  for (size_t r = 0; r < len; ++r) {
    result[r] = op(lhs[r], rhs[r]);
  }
}

template <typename T, size_t len>
void pairwise(T const lhs, VEC<T, len> const& rhs, VEC<T, len>& result, auto&& op) {
  for (size_t r = 0; r < len; ++r) {
    result[r] = op(lhs, rhs[r]);
  }
}

template <typename T, size_t row, size_t col>
void pairwise(T const lhs, MAT<T, row, col> const& rhs, MAT<T, col>& result, auto&& op) {
  for (size_t r = 0; r < row; ++r) {
    for (size_t c = 0; c < col; ++c) {
      result[r, c] = op(lhs, rhs[r, c]);
    }
  }
}

// matrix conjugate transpose multiplication
template <typename T, size_t row, size_t col>
void conj_mult(MAT<T, row, col> const& src, MAT<T, col>& result) {
  for (size_t r = 0; r < col; ++r) {
    for (size_t c = 0; c <= r; ++c) {
      VEC<T, row> tmp_mult;
      for (size_t k = 0; k < row; ++k) {
        tmp_mult[k] = my_conj(src[k, r]) * src[k, c];
      }

      result[r, c] = Qreduce<T>(tmp_mult);
      result[c, r] = my_conj(result[r, c]);
    }
  }
}

template <typename T, size_t row1, size_t col1, size_t col2>
void conj_mult(MAT<T, row1, col1> const& lhs, MAT<T, row1, col2> const& rhs, MAT<T, col1, col2>& result) {
  for (size_t r = 0; r < col1; ++r) {
    for (size_t c = 0; c < col2; ++c) {
      VEC<T, row1> tmp_mult;
      for (size_t k = 0; k < row1; ++k) {
        tmp_mult[k] = my_conj(lhs[k, r]) * rhs[k, c];
      }

      result[r, c] = Qreduce<T>(tmp_mult);
    }
  }
}

template <typename T, size_t row, size_t col>
void conj_mult(MAT<T, row, col> const& lhs, VEC<T, row> const& rhs, VEC<T, col>& result) {
  for (size_t r = 0; r < col; ++r) {
    VEC<T, row> tmp_mult;
    for (size_t k = 0; k < row; ++k) {
      tmp_mult[k] = my_conj(lhs[k, r]) * rhs[k];
    }

    result[r] = Qreduce<T>(tmp_mult);
  }
}

template <typename src_t, typename res_t, size_t row, size_t col>
void conj_mult(MAT<src_t, row, col> const& src, MAT<res_t, col>& result) {
  for (size_t r = 0; r < col; ++r) {
    for (size_t c = 0; c <= r; ++c) {
      VEC<res_t, row> tmp_mult;

      for (size_t k = 0; k < row; ++k) {
        tmp_mult[k] = Qmul<BasicComplexMul<acT<typename res_t::realType>,
                                           bdT<typename res_t::realType>,
                                           adT<typename res_t::imagType>,
                                           bcT<typename res_t::imagType>,
                                           acbdT<typename res_t::realType>,
                                           adbcT<typename res_t::imagType>>>(my_conj(src[k, r]), src[k, c]);
      }

      result[r, c] = Qreduce<res_t>(tmp_mult);
      result[c, r] = my_conj(result[r, c]);
    }
  }
}

// matrix multiplication
template <typename T, size_t row1, size_t col1, size_t col2>
void mat_mult(MAT<T, row1, col1> const& lhs, MAT<T, col1, col2> const& rhs, MAT<T, row1, col2>& result) {
  for (size_t r = 0; r < row1; ++r) {
    for (size_t c = 0; c < col2; ++c) {
      VEC<T, col1> tmp_mult;

      for (size_t k = 0; k < col1; ++k) {
        tmp_mult[k] = lhs[r, k] * rhs[k, c];
      }

      result[r, c] = Qreduce<T>(tmp_mult);
    }
  }
}

template <typename T, size_t row, size_t col>
void mat_mult(MAT<T, row, col> const& lhs, VEC<T, col> const& rhs, VEC<T, row>& result) {
  for (size_t r = 0; r < row; ++r) {
    VEC<T, col> tmp_mult;

    for (size_t c = 0; c < col; ++c) {
      tmp_mult[c] = lhs[r, c] * rhs[c];
    }

    result[r] = Qreduce<T>(tmp_mult);
  }
}

// wNSA iteration: μ(i+1) = b − (I-W) * μ(i)
template <size_t len, typename T>
void wNSA_iter(
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

// secodn order iteration: μ(i+1) = μ(i−1) + (b − W * μ(i)) + k * (μ(i) − μ(i−1))
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