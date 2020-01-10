#ifndef MMU_COMMON_IO_H
#define MMU_COMMON_IO_H

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace mmu {
/// Read variable of the arbitrary type
template <typename T>
void read_variable(std::istream &in, T &podRef) {
  in.read((char *)&podRef, sizeof(T));
}

/// Read std::vector of the arbitrary type
template <typename T>
uint32_t read_vector(std::istream &in, T *vec) {
  uint32_t size;
  in.read((char *)&size, sizeof(uint32_t));
  vec = new T[size];
  in.read((char *)vec, size * sizeof(T));
  return size;
}

/// Read std::vector of the arbitrary type
template <typename T>
void read_vector(std::istream &in, std::vector<T> &vec) {
  uint32_t size;
  in.read((char *)&size, sizeof(uint32_t));
  vec.resize(size);
  in.read((char *)vec.data(), size * sizeof(T));
}

/// Write variable of the arbitrary type
template <typename T>
void write_variable(std::ostream &out, const T &val) {
  out.write((char *)&val, sizeof(T));
}

/// Write std::vector in the fvec/ivec/bvec format
template <typename T>
void write_vector(std::ostream &out, std::vector<T> &vec) {
  const uint32_t size = vec.size();
  out.write((char *)&size, sizeof(uint32_t));
  out.write((char *)vec.data(), size * sizeof(T));
}

template <typename T>
void write_vector(std::ostream &out, T *vec, uint32_t size) {
  out.write((char *)&size, sizeof(uint32_t));
  out.write((char *)vec, size * sizeof(T));
}

/// Read fvec/ivec/bvec format vectors
template <typename T>
void readXvec(std::ifstream &in, T *data, const size_t d, const size_t n = 1) {
  uint32_t dim = d;
  for (size_t i = 0; i < n; i++) {
    in.read((char *)&dim, sizeof(uint32_t));
    if (dim != d) {
      std::cout << "file error\n";
      exit(1);
    }
    in.read((char *)(data + i * dim), dim * sizeof(T));
  }
}

/// Write fvec/ivec/bvec format vectors
template <typename T>
void writeXvec(std::ofstream &out, T *data, const size_t d, const size_t n = 1) {
  const uint32_t dim = d;
  for (size_t i = 0; i < n; i++) {
    out.write((char *)&dim, sizeof(uint32_t));
    out.write((char *)(data + i * dim), dim * sizeof(T));
  }
}

/// Read fvec/ivec/bvec format vectors and convert them to the float array
template <typename T>
void readXvecFvec(std::ifstream &in, float *data, const size_t d, const size_t n = 1) {
  uint32_t dim = d;
  T mass[d];

  for (size_t i = 0; i < n; i++) {
    in.read((char *)&dim, sizeof(uint32_t));
    if (dim != d) {
      std::cout << "file error\n";
      exit(1);
    }
    in.read((char *)mass, dim * sizeof(T));
    for (size_t j = 0; j < d; j++) data[i * dim + j] = 1. * mass[j];
  }
}

/// Check if file exists
inline bool exists(const char *path) {
  std::ifstream f(path);
  return f.good();
}

}  // namespace mmu

#endif