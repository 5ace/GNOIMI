/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */



#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>
#include <sys/time.h>

#include <faiss/index_io.h>
#include <faiss/AutoTune.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_factory.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>
#include <omp.h>
#include <gflags/gflags.h>
#include "tool/utils.h"

using namespace gnoimi;
using namespace faiss;
DEFINE_string(base_file,"","file content vectors to index,format:bvecs or fvecs");
DEFINE_string(index_file,"","index_file_name ");
DEFINE_uint64(query_thread_num,1,"");
DEFINE_uint64(N,1000000000,"index vectors num");
DEFINE_uint64(D,128,"index vectors dim");
DEFINE_uint64(topk,100,"query topk");
DEFINE_string(learn_file,"","file content vectors to learn,format:bvecs or fvecs");
DEFINE_uint64(learn_N,0,"want read vectors num");
DEFINE_uint64(query_N,0,"want read vectors num");
DEFINE_string(index_factory_str,"","like IMI2x8,PQ8+16");
DEFINE_bool(make_index,false,"制作index");
DEFINE_bool(search_index,true,"是否查询index");
DEFINE_string(groundtruth_file,"","file content vectors to learn,format:ivecs");
DEFINE_string(query_file,"","file content vectors to learn,format:ivecs");
DEFINE_bool(query_single,true,"为了保证可比性，与gnoimi一样单个query查询，避免faiss的batch加速");
DEFINE_uint64(efSearch,100,"hnsw参数");
DEFINE_uint64(efConstruction,200,"hnsw参数");
DEFINE_string(search_args,"","k_factor=xx,k_factor_rf=xx,nprobe=xx,ht=xx,max_codes=xx,efSearch=xx,\
  if you want set multi search_args use | concat them, eg:nprobe=10|nprobe=20|nprob=30");

int main(int argc,char** argv)
{
    gnoimi::print_elements(argv,argc);

    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true; //日志全部输出到标准错误

    double t0 = elapsed();

    CHECK(!FLAGS_base_file.empty() && !FLAGS_learn_file.empty() && !FLAGS_index_factory_str.empty() && !FLAGS_index_file.empty());

    std::vector<std::string> selected_params_multi = split(FLAGS_search_args,"|");
    //bool do_polysemous_training = false;
    for(auto & query_params: selected_params_multi) {
       LOG(INFO) <<"get query_params:"<<query_params.c_str();
      //if(query_params.find("ht=")!=std::string::npos) {
      //  do_polysemous_training = true; 
      //}
    }
    LOG(INFO) << "selected_params_multi size:" << selected_params_multi.size();
    if(selected_params_multi.empty())
      selected_params_multi.push_back("");

    std::string query_file = FLAGS_query_file;
    std::string groundtruth_file = FLAGS_groundtruth_file;

    const char *index_key = FLAGS_index_factory_str.c_str();
    std::string learn_file = FLAGS_learn_file;
    std::string base_file = FLAGS_base_file;
    
    CHECK(!access(learn_file.c_str(),R_OK) && !access(base_file.c_str(),R_OK)&& !access(query_file.c_str(),R_OK) && 
      !access(groundtruth_file.c_str(),R_OK));

    faiss::Index * index;

    size_t d = FLAGS_D;
    faiss::IndexHNSWFlat * hnsw = nullptr;
if (FLAGS_make_index) {
    {
        LOG(INFO) <<"["<<elapsed() - t0<<" s] Loading train set";

        size_t nt = FLAGS_learn_N;
        auto queries =  gnoimi::read_bfvecs(FLAGS_learn_file.c_str(), FLAGS_D, nt,true);
        float *xt = queries.get();
        LOG(INFO) << "L2 norm " << faiss::fvec_norm_L2sqr(queries.get(), FLAGS_D);

        LOG(INFO) <<"["<<elapsed() - t0<<" s] Preparing index "<<index_key<<" d=" <<d;
        index = faiss::index_factory(d, index_key);
        index->verbose = true;
        faiss::Index *real_index = index;
        faiss::IndexPreTransform* p = dynamic_cast<faiss::IndexPreTransform* >(index);
        if(p) {
          LOG(INFO) <<"this is  a IndexPreTransform";
          real_index = p->index;
        }
        real_index->verbose = true;
        faiss::IndexIVFPQ* pp = dynamic_cast<faiss::IndexIVFPQ *>(real_index);
        if(pp != nullptr) {
          //pp->do_polysemous_training=do_polysemous_training;
          LOG(INFO) <<"this is a IndexIVFPQ, do_polysemous_training:"<<pp->do_polysemous_training;
        } else {
          LOG(INFO) <<"this is not a  IndexIVFPQ";
        }
        hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(real_index);
        if(hnsw != nullptr) {
          LOG(INFO) << "this is HNSWFLAT index,set efSearch:" << FLAGS_efSearch <<",efConstruction:" << FLAGS_efConstruction; 
          hnsw->hnsw.efSearch = FLAGS_efSearch;
          hnsw->hnsw.efConstruction = FLAGS_efConstruction;
        }
        LOG(INFO) <<"["<<elapsed() - t0<<" s] Training on "<<nt<<" vectors";
        index->train(nt, xt);
    }
      size_t read_num = FLAGS_N;
      gnoimi::b2fvecs_read_callback(FLAGS_base_file.c_str(),
         FLAGS_D, read_num, 100000,std::bind( [](faiss::Index* index, size_t n, float *x, size_t start_id){ index->add(n,x);},index, std::placeholders::_1, 
         std::placeholders::_2, std::placeholders::_3));
      LOG(INFO) << "end index,read " << read_num << ",want " << FLAGS_N;
    faiss::write_index(index, FLAGS_index_file.c_str());
    LOG(INFO) <<"finish write index to " << FLAGS_index_file.c_str();
}
    if(!FLAGS_search_index) {
      LOG(INFO) << "not search index";
      return 0;
    }
    CHECK(!access(FLAGS_index_file.c_str(),0));
    index = faiss::read_index(FLAGS_index_file.c_str());
    faiss::IndexIVFPQ* pp = nullptr;
    {
        index->verbose = true;
        faiss::Index *real_index = index;
        faiss::IndexPreTransform* p = dynamic_cast<faiss::IndexPreTransform* >(index);
        if(p) {
          printf("this is  a IndexPreTransform \n");
          real_index = p->index;
        }
        real_index->verbose = true;
        d = real_index->d;
        pp = dynamic_cast<faiss::IndexIVFPQ *>(real_index);
        if(pp != nullptr) {
          //pp->do_polysemous_training=do_polysemous_training;
          printf("this is a IndexIVFPQ, do_polysemous_training:%d\n",pp->do_polysemous_training);
        } else {
          printf("this is not a  IndexIVFPQ\n");
        }
        hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(real_index);
        if(hnsw != nullptr) {
          LOG(INFO) << "this is HNSWFLAT index, get efSearch:" << hnsw->hnsw.efSearch <<",efConstruction:" << hnsw->hnsw.efConstruction
            << ",max_level:"<< hnsw->hnsw.max_level; 
        }
    }
    size_t nq = FLAGS_query_N;
    auto queries =  gnoimi::read_bfvecs(FLAGS_query_file.c_str(), FLAGS_D, nq,true);
    float *xq = queries.get();
    size_t gt_d = 0; // nb of results per query in the GT
    faiss::Index::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors

    {
        printf ("[%.3f s] Loading ground truth for %ld queries\n",
                elapsed() - t0, nq);

        // load ground-truth and convert int to long
        size_t nq2 = 0;
        auto tmp = ivecs_read(groundtruth_file.c_str(), &gt_d, &nq2);
        int *gt_int = tmp.get();
        CHECK(nq2 >= nq) << "nq2:"<<nq2 << ",nq:"<<nq ;
        LOG(INFO) <<"get nq:"<< nq2 <<" k:"<< gt_d<<" from gt file:" << groundtruth_file.c_str();

        gt = new faiss::Index::idx_t[gt_d * nq];
        for(int i = 0; i < gt_d * nq; i++) {
            gt[i] = gt_int[i];
        }
    }
    omp_set_num_threads(FLAGS_query_thread_num);
    LOG(INFO) << "query with " << FLAGS_query_thread_num << "threads";
    for(auto & selected_params: selected_params_multi)
    { // Use the found configuration to perform a search

        faiss::ParameterSpace params;
        params.verbose = 2;
        LOG(INFO) << "["<<elapsed() - t0<<" s] Setting parameter configuration  on index " <<selected_params.c_str();
        if(selected_params.size() !=0)
          params.set_index_parameters (index, selected_params.c_str());

        if(pp) {
          LOG(INFO) <<"IndexIVFPQ->polysemous_ht: "<< pp->polysemous_ht;
        }
        size_t k = FLAGS_topk;
        // output buffers
        faiss::Index::idx_t *I = new  faiss::Index::idx_t[nq * k];
        float *D = new float[nq * k];
        int loop = 10000/nq+1;
        loop = 1;
        double t1 = elapsed();
        for(int i=0;i<loop;++i) {
          if(FLAGS_query_single == false) {
            index->search(nq, xq, k, D, I);
          } else {
            for(int j = 0;j < nq;j++) {
              index->search(1,xq + j * FLAGS_D, k, D + j * k, I + j * k);    
            }
          }
        }
        double t2 = elapsed();

        std::cout << "["<<elapsed() - t0<<" s] Compute recalls, query "<<nq*loop<<", cost:"<<(t2 - t1)*1e3<<"ms, "<<(t2 - t1)*1e3/(nq*loop)<<"ms each query"
          <<",doc compare/q:" << indexIVF_stats.ndis*1.0/(nq*loop) <<",args="<< selected_params <<"\n";
        if(pp) {
          printf("indexIVFPQ_stats.n_hamming_pass:%ld,nrefine:%ld,search_cycles:%ld,refine_cycles:%ld\n",
               indexIVFPQ_stats.n_hamming_pass,indexIVFPQ_stats.nrefine,indexIVFPQ_stats.search_cycles,
               indexIVFPQ_stats.refine_cycles);
          printf("indexIVF_stats.nq:%ld,nlist:%ld,ndis:%ld,nheap_updates:%ld,quantization_time:%.2f ms,search_time:%.2f ms, hamming_jump_rate:%.2f,nprobe=%d\n",
               indexIVF_stats.nq,indexIVF_stats.nlist/(nq*loop),indexIVF_stats.ndis/(nq*loop),indexIVF_stats.nheap_updates/(nq*loop),
               indexIVF_stats.quantization_time/(nq*loop),indexIVF_stats.search_time/(nq*loop),
               indexIVF_stats.ndis==0?0.0:(1.0-indexIVFPQ_stats.n_hamming_pass*1.0/indexIVF_stats.ndis),pp->nprobe);
          indexIVFPQ_stats.reset();
          indexIVF_stats.reset();
        }
        if(hnsw) {
          std::cout << "hnsw_stats.nreorder:" << hnsw_stats.nreorder << ",n1:"<<hnsw_stats.n1 <<",n2:" << hnsw_stats.n2
            << ",n3:" << hnsw_stats.n3 <<",ndis:" << hnsw_stats.ndis <<",view:"<<hnsw_stats.view << ",efSearch:"<<hnsw->hnsw.efSearch <<"\n";
          hnsw_stats.reset();
        }
        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        size_t hit_num = 0;
        int limit = std::min(std::min((int)gt_d,(int)k),100);
        // 遍历所有query
        for(int i = 0; i < nq; i++) {
            int gt_nn = gt[i * gt_d];
            // 遍历每个query的topk
            for(int j = 0; j < k; j++) {
                if (I[i * k + j] == gt_nn) {
                    if(j < 1) n_1++;
                    if(j < 10) n_10++;
                    if(j < 100) n_100++;
                }
                if(j < limit) {
                  for(int n = 0; n < limit; n++) {
                    if(I[i * k + j] == gt[i * gt_d + n]) {
                      hit_num++;
                      break;
                    }
                  }
                }
            }
        }
        std::cout <<"R@1:" << n_1 / float(nq) << "\n";
        std::cout <<"R@10:" << n_10 / float(nq) << "\n";
        std::cout <<"R@100:" << n_100 / float(nq) << "\n";
        std::cout <<"SameInTop"<<limit<<":" << float(hit_num)/nq/limit << "\n";
        
        {
            char query_result_ctr[2048];
            sprintf(query_result_ctr,"%s.%s.queryresult.ivecs",
              FLAGS_index_file.c_str(),
              selected_params.c_str());
            string query_result(query_result_ctr);
            vector<int> qr(nq * FLAGS_topk);
            for(int i = 0; i < nq * FLAGS_topk; i++) {
                qr[i] = int(I[i]); 
            }
            ivecs_write(query_result.c_str(), FLAGS_topk, nq, qr.data());
            LOG(INFO) << "write query result to " << query_result;
        }
    }
}
