#pragma once


#include <sstream>
#include <memory>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stddef.h>
#include <functional>

#include <string>
#include <vector>
#include <glog/logging.h>
#include <faiss/Clustering.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexFlat.h>


using std::string;
using std::vector;

#ifdef __cplusplus
extern "C"{
#endif

#include <yael/kmeans.h>
#include <yael/vector.h>
#include <yael/matrix.h>
#include <yael/binheap.h>
long fvecs_fread (FILE * f, float * v, long n, int d_alloc);
int fvecs_read(const char *fname, int d, int n, float *v);
int ivecs_new_read(const char *fname, int *d_out, int **vi);
void fmat_mul_full(const float *left, const float *right,
                   int m, int n, int k,
                   const char *transp,
                   float *result);
int fvec_read(const char *fname, int d, float *a, int o_f);
int* ivec_new_read(const char *fname, int *d_out);
void fmat_rev_subtract_from_columns(int d,int n,float *m,const float *avg);
void fvec_sub(float * v1, const float * v2, long n);
void fvec_add(float * v1, const float * v2, long n);
float* fmat_new_transp(const float *a, int ncol, int nrow);
float kmeans (int d, int n, int k, int niter, 
        const float * v, int flags, long seed, int redo, 
        float * centroids, float * dis, 
        int * assign, int * nassign);
float* fmat_new_transp (const float *a, int ncol, int nrow);
struct fbinheap_s * fbinheap_new (int maxk);
int ivecs_new_read (const char *fname, int *d_out, int **vi);
#ifdef __cplusplus
}
#endif


namespace gnoimi {
inline bool end_with (const std::string& a, const std::string& b) {
  return a.size()>=b.size() && a.compare(a.size()-b.size(),b.size(),b) == 0;
}
using Call_back = std::function<void(size_t n, float *x, size_t start_id)>;
inline size_t file_size (const char* fname) {
    FILE *f = fopen(fname, "r");
    if(!f) {
      return 0;
    }
    struct stat st;
    fstat(fileno(f), &st);
    return st.st_size;
}
float * fvecs_read (const char *fname,
                    size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    if( *d_out == 0) {
       *d_out = d;
    }
    if(*n_out == 0) {
       *n_out = n;
    }
    assert(*d_out == d);
    CHECK(*n_out <= n) << "*n_out:"<< *n_out <<",n:"<<n<< ",file:"<<fname;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

std::vector<std::string> split(const std::string &str,const std::string &pattern)
{
    std::vector<std::string> resVec;

    if ("" == str)
    {
        return resVec;
    }
    std::string strs = str + pattern;

    size_t pos = strs.find(pattern);
    size_t size = strs.size();

    while (pos != std::string::npos)
    {
        std::string x = strs.substr(0,pos);
        resVec.push_back(x);
        strs = strs.substr(pos+1,size);
        pos = strs.find(pattern);
    }
    return resVec;
}

static long xvecs_fsize(long unitsize, const char * fname, size_t *d_out, size_t *n_out)
{
    int d, ret;
    long nbytes;

    *d_out = -1;
    *n_out = -1;

    FILE * f = fopen (fname, "r");

    if(!f) {
        fprintf(stderr, "xvecs_fsize %s: %s\n", fname, strerror(errno));
        return -1;
    }
    /* read the dimension from the first vector */
    ret = fread (&d, sizeof (d), 1, f);
    if (ret == 0) { /* empty file */
        *n_out = 0;
        return ret;
    }

    fseek (f, 0, SEEK_END);
    nbytes = ftell (f);
    fclose (f);

    if(nbytes % (unitsize * d + 4) != 0) {
        fprintf(stderr, "xvecs_size %s: weird file size %ld for vectors of dimension %d\n", fname, nbytes, d);
        return -1;
    }

    *d_out = d;
    *n_out = nbytes / (unitsize * d + 4);
    return nbytes;
}

long bvecs_fsize (const char * fname, size_t *d_out, size_t *n_out)
{
    return xvecs_fsize (sizeof(unsigned char), fname, d_out, n_out);
}

int b2fvec_fread (FILE * f, float * v)
{
    int d, j;
    int ret = fread (&d, sizeof (int), 1, f);
    if (feof (f))
        return 0;

    if (ret != 1) {
        perror ("# bvec_fread error 1");
        return -1;
    }

    unsigned char * vb = (unsigned char *) malloc (sizeof (*vb) * d);

    ret = fread (vb, sizeof (*vb), d, f);
    if (ret != d) {
        perror ("# bvec_fread error 2");
        return -1;
    }

    float sum = 0.0;
    for(j = 0; j < d; j++)
    {
        sum += float(vb[j]) * float(vb[j]);
    }
    float l2norm = sqrt(sum);
    for (j = 0 ; j < d ; j++)
    {
        v[j] = vb[j]/l2norm;
    }
    // std::cout << "finish assign v\n";
    free (vb);
    return d;
}

int fvec_fwrite (FILE *fo, const float *v, int d) 
{
    int ret;
      ret = fwrite (&d, sizeof (int), 1, fo);
        if (ret != 1) {
              perror ("fvec_fwrite: write error 1");
                  return -1;
                    }
          ret = fwrite (v, sizeof (float), d, fo);
            if (ret != d) {
                  perror ("fvec_fwrite: write error 2");
                      return -1;
                        }  
              return 0;
}
long ivecs_fread (FILE * f, int * v, long n);

int b2fvecs_read_callback (const char *fname, size_t &d, size_t &n, size_t each_loop_num,
                              Call_back call_back, bool normalized = true) {
    size_t n_new;
    size_t d_new;
    bool is_fvecs = false;

    if(end_with(fname,".fvecs")) {
      is_fvecs = true;
      xvecs_fsize (sizeof(float), fname, &d_new, &n_new);
    } else if(end_with(fname,".bvecs")){
      xvecs_fsize (sizeof(char), fname, &d_new, &n_new);
    } else {
      CHECK(false) << " file "<<fname << ",format wrong";
    }
    LOG(INFO) << "b2fvecs_read_callback file "<< fname <<",format:" 
      << (is_fvecs ? "fevc" : "bvec") << ", in file d:" << d_new
      << ",file has doc:"<<n_new <<",want read docnum:" << n;

    if(d==0) {
      d = d_new;
    }
    if(n==0) {
      n = n_new;
    }
    assert (d_new == d);
    assert (n <= n_new);
    float* v = new float[d * each_loop_num];
    FILE * f = fopen (fname, "r");
    assert (f || "b2fvecs_read: Unable to open the file");
    size_t left = n;
    size_t each_loop = each_loop_num;
    size_t start_id = 0;
    while(left!=0) {
      if(left < each_loop_num) {
        each_loop = left;
      }
      LOG(INFO) << "b2fvecs_read_callback: left:" << left <<",loop_num:" << each_loop;
      if(is_fvecs) {
        fvecs_fread(f, v, each_loop, (int)d);
      }else {
        b2fvecs_fread(f, v, each_loop);
      }
      if(normalized) {
        faiss::fvec_renorm_L2(d,each_loop,v);
      }
      call_back(each_loop,v,start_id);
      left -= each_loop;
      start_id += each_loop;
    }
    fclose (f);
    delete[] v;
    return 0;
}
float* b2fvecs_read (const char *fname, size_t &d, size_t &n)
{
    size_t n_new;
    size_t d_new;
    bvecs_fsize (fname, &d_new, &n_new);
    if(d==0) {
      d = d_new;
    }
    if(n==0) {
      n = n_new;
    }
    assert (d_new == d);
    assert (n <= n_new);
    float* v = new float[d * n];

    FILE * f = fopen (fname, "r");
    b2fvecs_fread(f, v, n);
    assert (f || "b2fvecs_read: Unable to open the file");
    fclose (f);
    return v;
}

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}
inline std::shared_ptr<float> read_bfvecs(std::string file, size_t &d, size_t& n, bool normalized) {
  float *p =nullptr;
  if(end_with(file,".bvecs")) {
    LOG(INFO) << " read .bvec " << file;
    p = b2fvecs_read(file.c_str(), d, n);
  }else if(end_with(file,".fvecs") || end_with(file,".ivecs")) {
    LOG(INFO) << " read .fvec or ivecs" << file;
    p = fvecs_read(file.c_str(), &d, &n);
  }
  CHECK(p!=nullptr) << " not want format .bvec or .fvec";
  if(normalized) {
    LOG(INFO) << "xxxxxxx" << d << n;
    faiss::fvec_renorm_L2(d,n,p);
  }
  return std::shared_ptr<float>(p);

}
// not very clean, but works as long as sizeof(int) == sizeof(float)
std::shared_ptr<int> ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
    return std::shared_ptr<int>((int*)fvecs_read(fname, d_out, n_out));
}
void print_elements (char* start, size_t n) {
  std::stringstream ss;
  ss << "[";
  for(size_t i = 0; i < n; i++) {
    ss << (int)start[i] << (i==n-1 ? "":",");
  }
  ss<<"]";
  LOG(INFO) << ss.str();
}
template<typename T>
void print_elements(const T* start, size_t n) {
  std::stringstream ss;
  ss << "[";
  for(size_t i = 0; i < n; i++) {
    ss << start[i] << (i==n-1 ? "":",");
  }
  ss<<"]";
  LOG(INFO) << ss.str();
}

} //end of gnoimi
namespace faiss {
  inline float kmeans_clustering (size_t d, size_t n, size_t k,
                           const float *x,
                           float *centroids, const ClusteringParameters &cp)
  {
      Clustering clus (d, k, cp);
      // display logs if > 1Gflop per iteration
      IndexFlatL2 index (d);
      clus.train (n, x, index);
      memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * k);
      return clus.obj.back();
  }
}
