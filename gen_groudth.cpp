#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <thread>
#include <functional>
#include <vector>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <limits>

#include <cblas.h>
#include <string.h>

#include "tool/utils.h"
#include <gflags/gflags.h>
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>


DEFINE_string(base_file,"","format bvec or fvec");
DEFINE_uint64(N,0,"index vec num,0 mean all");
DEFINE_uint64(D,128,"index vec num,0 mean all");
DEFINE_string(groud_truth_file,"","format ivec");
DEFINE_string(query_filename,"","format fvec");
DEFINE_uint64(query_N,0,"query vec num,0 mean all");
DEFINE_uint64(K,1000,"get top k groud truth for each query");

int main(int argc, char** argv) {
    gnoimi::print_elements(argv,argc);

    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true; //日志全部输出到标准错误
    
    CHECK((gnoimi::end_with(FLAGS_base_file,".bvecs") || gnoimi::end_with(FLAGS_base_file,".fvecs")) &&
      gnoimi::end_with(FLAGS_groud_truth_file,".ivecs") && (gnoimi::end_with(FLAGS_query_filename,".fvecs")
      || gnoimi::end_with(FLAGS_query_filename,".bvecs") ) );
    CHECK(FLAGS_K>0);
    // 读取query
    size_t q_num = FLAGS_query_N;
    auto query = gnoimi::read_bfvecs(FLAGS_query_filename.c_str(), FLAGS_D, q_num, false);

    faiss::IndexFlat index(FLAGS_D);
    auto AddIndex = [](faiss::IndexFlat* index ,size_t n,const float *x,size_t start) {index->add(n,x);}; 
    gnoimi::b2fvecs_read_callback(FLAGS_base_file.c_str(),
      FLAGS_D , FLAGS_N, 10000,std::bind(AddIndex,&index,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3));

    std::vector<float> dis(q_num*FLAGS_K);
    std::vector<int64_t> pos(q_num*FLAGS_K);
    std::vector<int> pos_i;
    index.search(q_num,query.get(),FLAGS_K,dis.data(),pos.data());
    for(auto i:pos) {
      pos_i.push_back((int)i);
      LOG_EVERY_N(INFO,1000) << "gt " << i;
    }
    LOG(INFO) << "query:" << q_num <<",pos_i size:" << pos_i.size() << ",index.size:" <<index.xb.size();
    ivecs_write(FLAGS_groud_truth_file.c_str(),FLAGS_K,q_num,pos_i.data());
    return 0;
}
