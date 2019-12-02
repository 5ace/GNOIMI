#include "tool/utils.h"
#include <faiss/utils/distances.h>
#include <gflags/gflags.h>
DEFINE_string(input_file,"","input file fvecs or bvecs");
DEFINE_uint64(N,0,"read num from file");
int main(int argc, char** argv) {
    gnoimi::print_elements(argv,argc);
    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true; //日志全部输出到标准错误
    
    CHECK(gnoimi::end_with(FLAGS_input_file,"vecs"));
    bool is_ivecs = gnoimi::end_with(FLAGS_input_file,".ivecs");
    size_t d = 0;
    auto ptr = gnoimi::read_bfvecs(FLAGS_input_file,d,FLAGS_N,false);
    LOG(INFO) << "read from " << FLAGS_input_file << ", d:" << d
      << ",n:"<< FLAGS_N;
    for(int i = 0; i < FLAGS_N; i++){
      LOG(INFO) << "====="<<i<<"=====";
      if(is_ivecs)
        gnoimi::print_elements((int*)(ptr.get())+i*d,d);
      else
        gnoimi::print_elements(ptr.get()+i*d,d);
      LOG(INFO) << "norml2:" << faiss::fvec_norm_L2sqr(ptr.get()+i*d,d) ;
    }
    return 0;
}
