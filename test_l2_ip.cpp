#include <cblas.h>
#include <string.h>

#include "tool/utils.h"
#include <faiss/utils/distances.h>
#include <faiss/impl/HNSW.h>
#include <gflags/gflags.h>
#include <faiss/index_io.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

using std::cout;
using std::ios;
using std::string;
using std::vector;

DEFINE_int32(num,1000000,"for num");
DEFINE_int32(batch,1,"batch_size");
DEFINE_int32(d,128,"dimension");
using namespace gnoimi;

int main(int argc, char** argv) {
    gnoimi::print_elements(argv,argc);

    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true; //日志全部输出到标准错误

    int n = 100;
    vector<float> v1(FLAGS_d * n);
    vector<float> v2(FLAGS_d * n);
    faiss::float_rand(v1.data(), FLAGS_d * n, 12345);
    faiss::float_rand(v2.data(), FLAGS_d * n, 12347);
    faiss::fvec_renorm_L2(FLAGS_d, 1 , v1.data());
    faiss::fvec_renorm_L2(FLAGS_d, 1 , v2.data());

    omp_set_num_threads(1);

    double t0 = elapsed();
    double s0 = 0;
    double s1 = 0;
    for(uint64_t i = 0; i < FLAGS_num; i++) {
      for(int j = 0; j < n; j ++) {      
        s0 += faiss::fvec_L2sqr(v1.data() + j * FLAGS_d, v2.data()  + j * FLAGS_d, FLAGS_d);
      }
    }
    double t1 = elapsed();
    for(uint64_t i = 0; i < FLAGS_num; i++) {
      for(int j = 0; j < n; j ++) {      
        s1 += faiss::fvec_inner_product(v1.data()  + j * FLAGS_d, v2.data()  + j * FLAGS_d, FLAGS_d);
      }
    }
    double t2 = elapsed();

    LOG(INFO) <<"Finish "<<",l2:"<<(t1-t0)*1e6/FLAGS_num<<"us,sum:"<<s0<<",ip:"<<(t2-t1)*1e6/FLAGS_num<<"us,sum:"<<s1;

    return 0;
}
