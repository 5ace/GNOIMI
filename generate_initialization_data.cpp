#include <iostream>
#include <functional>
#include <ctime>
#include <chrono>
#include "tool/utils.h"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <gflags/gflags.h>
#include <faiss/utils/distances.h>

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
DEFINE_int32(k,256,"default k means num");
DEFINE_int32(d,128,"vector dim");
DEFINE_int32(M,16,"opq 's sub vec num");
DEFINE_int32(thread_num,10,"train kmeans thread num");
DEFINE_int32(n,200000,"train kmeans vec num");
DEFINE_string(train_file,"","train file name, .bvecs or .fvecs");
DEFINE_string(coarse_centriods_file,"","coarse_centriods_file");
DEFINE_string(residual_centriods_file,"","residul_centriods_file");
DEFINE_bool(use_yael,false,"use yael for kmeans");
DEFINE_int32(iter_num,100000,"iter_num for kmeans");


int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true; //日志全部输出到标准错误


    string f_file_list = FLAGS_train_file; 
    size_t k = FLAGS_k;
    string coarseFile_name = FLAGS_coarse_centriods_file; 
    string fineFile_name = FLAGS_residual_centriods_file;
    size_t d = FLAGS_d;
    size_t n = FLAGS_n;
    faiss::ClusteringParameters cp;
    cp.niter = FLAGS_iter_num;
    cp.verbose = true;

    CHECK(!f_file_list.empty() && !coarseFile_name.empty() && !fineFile_name.empty() && k > 0 && d>=0 && n>=0);
    
    // read train vectors
    std::shared_ptr<float> features = read_bfvecs(f_file_list, d,  n, true); 

    LOG(INFO) << " load " << n << " vecs from " << f_file_list << " dim " << d;
    LOG(INFO) << "train vec Squared L2 norm " << faiss::fvec_norm_L2sqr(features.get(),d);
    
    std::unique_ptr<float> coarse_centroids(new float[d*k]);
    std::unique_ptr<float> residual_centroids(new float[d*k]);
    std::unique_ptr<faiss::Index::idx_t> closest_id(new faiss::Index::idx_t[n]);
    std::unique_ptr<int> closest_id_int(new int[n]);
    std::unique_ptr<float> closest_dis(new float[n]);
    std::unique_ptr<int>   nassign(new int[k]);

    //estimate coarse kmean
    LOG(INFO) << "Estimate coarse kmean";
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); 
    float error = 0;
    if(!FLAGS_use_yael) {
      error = faiss::kmeans_clustering(d,n,k,features.get(),coarse_centroids.get(),cp);
    } else {
      error = kmeans(d,n,k,FLAGS_iter_num,features.get(),FLAGS_thread_num,1234,1,coarse_centroids.get(),closest_dis.get(),
        closest_id_int.get(),nassign.get());  
      LOG(INFO) << "yael:closest_id =====";
      gnoimi::print_elements(closest_id_int.get(),100);
    }
    LOG(INFO) << "coarse kmean, use yael:" <<FLAGS_use_yael << ",error:" << error;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    LOG(INFO) << "Coarse Kmean time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    gnoimi::print_elements(coarse_centroids.get(),d);
    gnoimi::print_elements(coarse_centroids.get()+d,d);
    gnoimi::print_elements(coarse_centroids.get()+2*d,d);

    //normalize coarse centroids 
    //faiss::fvec_renorm_L2(d,k,coarse_centroids.get());

    //cal residual 
    cout << "calculate residual" << endl;
    std::shared_ptr<faiss::Index> index = std::make_shared<faiss::IndexFlatL2>(d);
    index->add(k,coarse_centroids.get());

    std::unique_ptr<float> residuals(new float[d*n]);
    index->assign(n,features.get(),closest_id.get());
    LOG(INFO) << "faiss search:closest_id =====";
    gnoimi::print_elements(closest_id.get(),100);
    index->compute_residual_n(n,features.get(),residuals.get(),closest_id.get());

    //estimate residual kmean
    cout << "Estimate residual kmean" << endl;
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now(); 
    if(!FLAGS_use_yael) {
      error = faiss::kmeans_clustering(d,n,k,residuals.get(),residual_centroids.get(),cp);
    } else {
      error = kmeans(d,n,k,FLAGS_iter_num,residuals.get(),FLAGS_thread_num,1234,1,residual_centroids.get(),closest_dis.get(),
        closest_id_int.get(),nassign.get());  
      LOG(INFO) << "yael:closest_id =====";
      gnoimi::print_elements(closest_id_int.get(),100);
    }
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    LOG(INFO) << "Coarse Kmean time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count();
    gnoimi::print_elements(residual_centroids.get(),d);

    //write to fvecs
    fvecs_write(coarseFile_name.c_str(),d,k,coarse_centroids.get());
    fvecs_write(fineFile_name.c_str(),d,k,residual_centroids.get());

    LOG(INFO) << "finish";
    return 0;
}
