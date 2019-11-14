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
/* --------------------------------------------------------------------------*/
/**
 * This program is for generating all the required data for learn_GNOIMI 
 * 
 * feature file list:	    input original features (end with .ivec or .fvec)
 * K                :	    the K-mean's K
 * coarse kmean file name:  the estimated K-mean centroids and stored as fvecs format
 * fine kmean file name:    the fine K-mean centroids and stored as fvecs format 
 * normalized feature:      the normalized features of the original featurs
 * label file name:         the corresponding label of the original featureas 
 *  
 */
/* ----------------------------------------------------------------------------*/
DEFINE_int32(d,128,"vector dim");
DEFINE_int32(M,16,"opq 's sub vec num");
DEFINE_uint64(n,0,"read train num, 0 = all");
DEFINE_string(train_file,"","train file name, .bvecs or .fvecs");
DEFINE_string(opq_matrix_file,"","opq_matrix_file .fvecs");
DEFINE_string(opq_train_file,"","train file after opq, .fvecs");


int main(int argc, char** argv) {
    print_elements(argv,argc);
    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true; //日志全部输出到标准错误

    string f_file_list = FLAGS_train_file; 
    size_t d = FLAGS_d;
    size_t n = FLAGS_n;

    CHECK(end_with(FLAGS_train_file,"vecs") &&  d>=0 && n>=0 && end_with(FLAGS_opq_train_file,".fvecs") &&
      end_with(FLAGS_opq_matrix_file,".fvecs"));
    
    // read train vectors
    std::shared_ptr<float> features = read_bfvecs(f_file_list, d,  n, true); 

    LOG(INFO) << " load " << n << " vecs from " << f_file_list << " dim " << d;
    LOG(INFO) << "train vec Squared L2 norm " << faiss::fvec_norm_L2sqr(features.get(),d);
    
    // train opq matrix
    // 计算OPQ旋转矩阵
    LOG(INFO) << "train opq start ";
    faiss::OPQMatrix opq(d,FLAGS_M);
    opq.verbose = true;
    opq.train(n,features.get());
    LOG(INFO) << "========opq[2] =======";
    gnoimi::print_elements(opq.A.data()+2*d,d);
    float* opq_result1 = opq.apply(1,features.get());
    float* opq_result2 = new float[d];
    fmat_mul_full(opq.A.data(), features.get(),
                  d, 1, d, "TN", opq_result2);
    LOG(INFO) << "========opq_result1 =======";
    gnoimi::print_elements(opq_result1,d);
    LOG(INFO) << "========opq_result2 =======";
    gnoimi::print_elements(opq_result2,d);
    LOG(INFO) << "train opq finish ";
    fvecs_write(FLAGS_opq_matrix_file.c_str(),d,d,opq.A.data());
    LOG(INFO) << "========rotarion train =======";
    float* opq_result = new float[d*n];
    fmat_mul_full(opq.A.data(), features.get(),
                  d, n, d, "TN", opq_result);
    fvecs_write(FLAGS_opq_train_file.c_str(),d,n,opq_result);
    LOG(INFO) << "========rotarion train finish=======";
    
    return 0;
}
