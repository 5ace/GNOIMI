#include <cblas.h>
#include <string.h>

#include "tool/utils.h"
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include <faiss/impl/HNSW.h>
#include <gflags/gflags.h>
#include <faiss/index_io.h>

using std::cout;
using std::ios;
using std::string;
using std::vector;
using namespace gnoimi;
#include <immintrin.h>

#define GNOIMI_QUERY_DEBUG
DEFINE_string(file,"","format bvec or fvecs");
DEFINE_uint64(n,200000,"read vec num");
DEFINE_uint64(d,8,"L2 dim");
static inline __m128 masked_read(int d, const float* x) {
  assert(0 <= d && d < 4);
  __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
  switch (d) {
    case 3:
      buf[2] = x[2];
    case 2:
      buf[1] = x[1];
    case 1:
      buf[0] = x[0];
  }
  return _mm_load_ps (buf);
  // cannot use AVX2 _mm_mask_set1_epi32
}
inline float fvec_L2sqr_ref (const float * x,
                     const float * y,
                     size_t d)
{
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
       res += tmp * tmp;
    }
    return res;
}
static float sseL2Distance(const float * x,  const float * y, const size_t& p_dim) {
  
  size_t d = p_dim;
  __m128 msum1 = _mm_setzero_ps();
  while (d >= 4) {
    __m128 mx = _mm_loadu_ps (x); x += 4;
    __m128 my = _mm_loadu_ps (y); y += 4;
    const __m128 a_m_b1 = mx - my;
    msum1 += a_m_b1 * a_m_b1;
    d -= 4;
  }

  if (d > 0) {
    // add the last 1, 2 or 3 values
    __m128 mx = masked_read (d, x);
    __m128 my = masked_read (d, y);
    __m128 a_m_b1 = mx - my;
    msum1 += a_m_b1 * a_m_b1;
  }

  msum1 = _mm_hadd_ps (msum1, msum1);
  msum1 = _mm_hadd_ps (msum1, msum1);
  return  _mm_cvtss_f32 (msum1);
}
int main(int argc, char** argv) {
    gnoimi::print_elements(argv,argc);

    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true; //日志全部输出到标准错误

    CHECK(FLAGS_n >= 10000 && !FLAGS_file.empty() && FLAGS_d > 0);
    uint64_t d = 0;
    std::shared_ptr<float> v = gnoimi::read_bfvecs(FLAGS_file,d,FLAGS_n,false);
    
    uint64_t num  = FLAGS_n*d/FLAGS_d/2;
    LOG(INFO) <<"read from " << FLAGS_file<<",d:"<<d<<",n:"<<FLAGS_n<<",L2 dim:"
      <<FLAGS_d<<",calc l2 num:"<< num;
    float* v1 = v.get();
    float* v2 = v.get() + num * FLAGS_d;
    double sum1 = 0.0;
    

    //随机打乱顺序模拟cachemiss比较严重的情况
    std::vector<int> idxs(num);
    faiss::rand_perm(idxs.data(),num,1234);
    
    #ifdef __SSE__
    LOG(INFO) <<"__SSE__";
    #endif
    #ifdef __aarch64__ 
    LOG(INFO) <<"__aarch64__";
    #endif
    #ifdef __AVX__
    LOG(INFO) <<"__AVX__";
    #endif

    //最傻的l2
    double t0 = elapsed();
    for(uint64_t i = 0; i < num; i++) {
      sum1 += fvec_L2sqr_ref(&v1[idxs[i] * FLAGS_d],&v2[ idxs[i] * FLAGS_d], FLAGS_d);
    }
    double t1 = elapsed();

    double sum2 = 0.0;
    for(uint64_t i = 0; i < num; i++) {
      sum2 += faiss::fvec_L2sqr(&v1[idxs[i] * FLAGS_d],&v2[ idxs[i] * FLAGS_d], FLAGS_d);
    }
    double t2 = elapsed();

    double sum3 = 0.0;
    for(uint64_t i = 0; i < num; i++) {
      sum3 += sseL2Distance(&v1[idxs[i] * FLAGS_d],&v2[ idxs[i] * FLAGS_d], FLAGS_d);
    }
    double t3 = elapsed();
    LOG(INFO) <<"Finish "<<",mine:"<<(t1-t0)*1e6/num<<"us,sum:"<<sum1<<",faiss:"<<(t2-t1)*1e6/num<<"us,sum:"<<sum2
        <<",sse:"<<(t3-t2)*1e6/num<<"us,sum:"<<sum3;

    return 0;
}
