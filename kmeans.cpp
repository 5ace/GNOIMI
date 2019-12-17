#include <iostream>
#include <functional>
#include <ctime>
#include <chrono>
#include "tool/utils.h"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <gflags/gflags.h>
#include <faiss/utils/distances.h>
#include <faiss/VectorTransform.h>

using namespace std;
using namespace gnoimi;

DEFINE_uint64(d,128,"vector dim");
DEFINE_uint64(n,0,"read train num, 0 = all");
DEFINE_string(train_file,"","train file name, .bvecs or .fvecs");
DEFINE_string(kmeans_file,"","out kmeans, .fvecs");
DEFINE_uint64(k,0,"kmean num");
DEFINE_uint64(thread_num,20,"threadnum");
DEFINE_uint64(redo,1,"redo");
DEFINE_uint64(iter,0,"iternum");


int main(int argc, char** argv) {
    print_elements(argv,argc);
    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true; //日志全部输出到标准错误

    CHECK(end_with(FLAGS_train_file,"vecs") &&  FLAGS_d>=0 && FLAGS_n>=0 && end_with(FLAGS_kmeans_file,".fvecs") && FLAGS_k > 0); 
    
    // read train vectors
    std::shared_ptr<float> features = read_bfvecs(FLAGS_train_file, FLAGS_d,  FLAGS_n, true); 

    LOG(INFO) << " load " << FLAGS_n << " vecs from " << FLAGS_train_file << " dim " << FLAGS_d;
    LOG(INFO) << "train vec Squared L2 norm " << faiss::fvec_norm_L2sqr(features.get(),FLAGS_d);

    std::unique_ptr<float> coarse_centroids(new float[FLAGS_d*FLAGS_k]);
    std::unique_ptr<float> residual_centroids(new float[FLAGS_d*FLAGS_k]);
    std::unique_ptr<int> closest_id_int(new int[FLAGS_n]);
    std::unique_ptr<float> closest_dis(new float[FLAGS_n]);
    std::unique_ptr<int>   nassign(new int[FLAGS_n]);

    float error = kmeans(FLAGS_d,FLAGS_n,FLAGS_k,FLAGS_iter,features.get(),
      FLAGS_thread_num,1234,FLAGS_redo,coarse_centroids.get(),closest_dis.get(),closest_id_int.get(),nassign.get());

    LOG(INFO) << "kmeans finish,error:"<<error<<",write file to " << FLAGS_kmeans_file;
    fvecs_write(FLAGS_kmeans_file.c_str(),FLAGS_d,FLAGS_k,coarse_centroids.get());
    return 0;
}
