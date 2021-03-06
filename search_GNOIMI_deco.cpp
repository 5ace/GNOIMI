#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <functional>
#include <vector>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <utility>
#include <tuple>

#include <cblas.h>
#include <string.h>

#include "tool/utils.h"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/distances.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/HNSW.h>
#include <gflags/gflags.h>
#include <map>
#include <faiss/index_io.h>
#include "io.h"

using std::cout;
using std::ios;
using std::string;
using std::vector;
using namespace mmu;
using namespace gnoimi;

//#define GNOIMI_QUERY_DEBUG

DEFINE_string(groud_truth_file,"","format ivec");
DEFINE_string(query_filename,"","format fvec");
DEFINE_bool(make_index,true,"制作index还是查询index, true:only make index, false: only search index");
DEFINE_string(model_prefix,"./train_gnoimi_","prefix of model files");
DEFINE_string(pq_prefix,"","if empty then same as model_prefix");
DEFINE_string(index_prefix,"./index_gnoimi_","prefix of index files");
DEFINE_string(base_file,"","file content vectors to index,format:bvecs or fvecs");
DEFINE_uint64(K,4096,"coarse and residual centriods num");
DEFINE_uint64(D,128,"feature dim");
DEFINE_uint64(M,16,"PQ's subvector num");
DEFINE_uint64(N,1000000000,"index vectors num");
DEFINE_uint64(topk,100,"返回结果的topk");
DEFINE_int64(L,32,"coarse search num for GNOIMI");
DEFINE_int32(rerankK,256,"PQ number of bit per subvector index,");
DEFINE_int32(threadsCount,10,"thread num");
DEFINE_int32(nprobe,200," if not in (0,K*L] else  set K*L");
DEFINE_int32(neighborsCount,5000,"coarse search num for GNOIMI");
DEFINE_int32(queriesCount,0,"load query count");
DEFINE_bool(residual_opq,true,"是否是对残差的opq,用原始向量训练一级码本，用残差训练pq；否则是先训练opq，再训练一级码本，再训练pq");
DEFINE_bool(print_gt_in_LK,true,"是否输出gt在粗排中的命中率");
DEFINE_string(lpq_file_prefix,"","lpq输出的多个opq pq faisspq的文件名前缀,如果是空就不用lopq");
DEFINE_bool(pq_minus_mean,false,"是否在pq或opq之前减去均值");
DEFINE_bool(dump_mmu_model,false,"是否dump mmu model");
DEFINE_bool(normalize_before_read,true,"读取索引和查询是否归一化");


std::vector<int> grouth_cellid;
struct SearchStats {
  int gt_in_LK = 0; // gt在 L个倒排链中的query个数
  int gt_in_retrievaled = 0; // gt被遍历过的query 个数
  int inner_table_num = 0;// decomposition时计算距离表的次数
  double ivf_cost = 0.0; //ms
  double pq_cost = 0.0; //ms
  double opq_cost = 0.0;
  uint64_t travel_doc = 0;
  uint64_t table_dis_doc = 0;
  uint64_t nprobe = 0;

  void Reset() {
    gt_in_LK = 0;
    gt_in_retrievaled = 0;
    ivf_cost = 0.0;
    pq_cost = 0.0;
    opq_cost = 0.0;
    travel_doc = 0;
    table_dis_doc = 0;
    nprobe = 0;
    inner_table_num = 0;
  }
} search_stats;

const uint64_t M = 16;
struct Record {
  uint32_t pointId;
  float term2;
  unsigned char bytes[M];
};

struct Searcher {
  // = 1 ; use decomposition
  // = 0 ; use redisuil table
  // else calc on the fly
  int use_precomputed_table = 1; 
  float* coarseVocab = nullptr;
  float* coarseNorms = nullptr;
  float* fineVocab = nullptr;
  float* fineNorms = nullptr;
  float* alpha = nullptr;
  float* coarseFineProducts = nullptr;
  Record* index = nullptr;
  uint64_t docnum;
  int* cellEdges; //cellEdges[0] 表示第一个倒排链的END，第二个倒排链的开头
  float* rerankRotation = nullptr;
  float* lopqRerankRotation = nullptr;// 每个倒排链一个opq矩阵
  float* rerankVocabs = nullptr;//global pq的码本
  float* lpqRerankVocabs = nullptr; //每个倒排链单独的码本
  float* gopq_residual_mean = nullptr; //gopq的均值 D维
  float* lopq_residual_mean = nullptr; //lopq的均值 K * D
  float* lopq_residual_mean_rotation = nullptr; //R * lopq_residual_mean for decomposition
  string model_prefix_,index_prefix_,pq_prefix_;
  int D;
  int K;
  int M;
  int subDim;
  int rerankK;
  int threadsCount;
  faiss::IndexPQ* pq_with_data = nullptr;

  vector<vector<int>> ivf; //cellis and it's docidlist

  string coarseCodebookFilename;
  string fineCodebookFilename;
  string alphaFilename, rerankRotationFilename, rerankVocabsFilename, cellEdgesFilename,rawIndexFilename,rerankPQFaissFilename,rerankPQFaissWithDataFilename;
  string lpq_file_prefix;
  string gopq_residual_mean_file;
  

  std::vector<uint8_t> codes; //所有doc的pq编码,按照add的顺序存储，与倒排链无关
  std::vector<float> term2; //所有doc的索引的与query无关的距离
  faiss::IndexPQ* pq = nullptr; // global的pq
  bool is_lopq = false; //是否每个倒排链独立o,独立o则独立mean
  bool is_lpq = false; //是否每个倒排链独立pq
  std::vector<faiss::IndexPQ*> lpqs; //每个倒排链各自存储的pq 是个数组
  uint64_t compute_table_docnum_threshold;//查询时倒排链长度大于这个就用查表法
  uint64_t compute_table_docnum_threshold_term3;

  Searcher(string model_prefix,string pq_prefix, string index_prefix,string lpq_file_prefix, uint64_t D,uint64_t K, uint64_t M, int rerankK,int threadsCount) {
    pq = nullptr;
    pq_with_data = nullptr;
    index = nullptr;
    model_prefix_ = model_prefix;
    index_prefix_ = index_prefix;
    pq_prefix_ = pq_prefix.empty() ? model_prefix_ : pq_prefix;
    this->D = D;
    this->K = K;
    this->M = M;
    // 累加距离的时候默认就是4的倍数了
    CHECK( M % 4 == 0 );
    this->rerankK = rerankK;
    this->threadsCount = threadsCount;
    subDim = this->D / this->M;
    alphaFilename = model_prefix_ +"alpha.fvecs";
    fineCodebookFilename = model_prefix_ + "fine.fvecs";
    coarseCodebookFilename = model_prefix_ + "coarse.fvecs";
    rerankRotationFilename = pq_prefix_ + "opq_matrix.fvecs";
    rerankVocabsFilename = pq_prefix_ + "pq.fvecs";
    rerankPQFaissFilename = pq_prefix_ + "pq.faiss.index";
    gopq_residual_mean_file = pq_prefix_ + "gopq_residual_mean.fvecs"; // gopq residual 的均值
    rerankPQFaissWithDataFilename = index_prefix_ + "pq.faiss.withdata.index";
    cellEdgesFilename = index_prefix_+ "cellEdges.dat";
    rawIndexFilename = index_prefix_+ "rawIndex.dat"; //作者设计的格式
    this->lpq_file_prefix = lpq_file_prefix;
    // 加载模型数据
    CHECK(ReadAndPrecomputeVocabsData()); 
    ivf.resize(K*K);
    compute_table_docnum_threshold = 1.3 * rerankK;
    compute_table_docnum_threshold_term3 = 0;
    LOG(INFO) << "Searcher construct ok,is_lpq:"<< is_lpq <<",is_lopq:" << is_lopq <<", compute_table_docnum_threshold:"<< compute_table_docnum_threshold;
    if(FLAGS_dump_mmu_model) {
      SaveModelForMMU();
      LOG(INFO) << "finish save model exit~";
      exit(0);
    }
    //LOG(INFO) << "alpha k2047:";
    //gnoimi::print_elements(alpha + 2047*K,K);
    //LOG(INFO)<<"lopq_residual_mean_rotation k2047:";
    //gnoimi::print_elements(lopq_residual_mean_rotation + 2047 * D, D);
    //LOG(INFO)<<"coarseFineProducts k2047:";
    //gnoimi::print_elements(coarseFineProducts + 2047 * K, K);

  }
  void LoadCellEdgesPart(int startId, int count) {
    std::ifstream inputCellEdges(cellEdgesFilename.c_str(), ios::binary | ios::in);
    CHECK(inputCellEdges.good());
    inputCellEdges.seekg(startId * sizeof(int));
    for(int i = 0; i < count; ++i) {
      inputCellEdges.read((char*)&(cellEdges[startId + i]), sizeof(int));
    }
    inputCellEdges.close();
  }
  
  void LoadCellEdges(int N) {
    int perThreadCount = N / threadsCount;
    std::vector<std::thread> threads;
    for (int threadId = 0; threadId < threadsCount; ++ threadId) {
      int startId = threadId * perThreadCount;
      int count = (threadId == threadsCount-1) ? (N - startId) : perThreadCount;
      threads.push_back(std::thread(std::bind(&Searcher::LoadCellEdgesPart, this, startId, count)));
    }
    for (int threadId = 0; threadId < threadsCount; ++threadId) {
      threads[threadId].join();
    }
    int last_cell = 0;
    int empty_cell = 0;
    float variance = 0;
    float avg = cellEdges[K*K - 1]*1.0 / (K*K);
    for(int i = 0; i < K * K ; i++) {
      //LOG_IF(INFO,cellEdges[i] - last_cell != 0) << "LoadCellEdges cell:" << i << ",docnum:" <<cellEdges[i] - last_cell;
      //if(cellEdges[i] - last_cell != 0) {
      //  for(int j = 0; j < cellEdges[i] - last_cell; j++)
      //    LOG(INFO) << "=cell "<< i << ",contains:" << index[last_cell+j].pointId;
      //}
      if(last_cell == cellEdges[i]) {
        empty_cell++;
      }
      variance += (cellEdges[i] - last_cell - avg) * (cellEdges[i] - last_cell - avg);
      last_cell = cellEdges[i];
    }
    LOG(INFO) << "LoadCellEdges finish,invert num:" << K*K << ",doc num:" 
      << cellEdges[K*K - 1] <<",empry cell num:" << empty_cell <<", variance of cell num:"
      << std::sqrt(variance)/(K*K);
  }
  bool LoadPQWithData() {
    CHECK(pq_with_data == nullptr && access(rerankPQFaissWithDataFilename.c_str(),0) == 0);
    LOG(INFO) << "start load Faiss PQ with data" << rerankPQFaissWithDataFilename;
    pq_with_data = dynamic_cast<faiss::IndexPQ*>(faiss::read_index(rerankPQFaissWithDataFilename.c_str()));
    pq_with_data->pq.compute_sdc_table();
    return pq_with_data != nullptr;
  }
  // prepare for search index
  bool LoadIndex() {
    uint64_t fz = gnoimi::file_size(rawIndexFilename.c_str());
    CHECK(fz > 0 && fz % sizeof(Record) == 0) << " check file " << rawIndexFilename;
    docnum = fz / sizeof(Record);
    LOG(INFO) << "LoadIndex find " << docnum << " docs in " << rawIndexFilename;
    index = (Record*) malloc(docnum * sizeof(Record));
    // 多线程读入id和编码信息到Record* index中
    LoadIDAndPQCode(docnum);
    cellEdges = (int*) malloc(K * K * sizeof(int));
    LoadCellEdges(K*K);
    return true;
  }
  void SaveModelForMMU() {
    std::string model_path = lpq_file_prefix + "mmu.model"; 
    std::ofstream ofs(model_path, std::ios::binary);
    CHECK(ofs.is_open()) << "open model failed:" << model_path;
    write_variable(ofs, D);
    write_variable(ofs, K);
    write_variable(ofs, M);
    write_variable(ofs, rerankK);
    write_vector(ofs, coarseVocab, K * D);
    write_vector(ofs, fineVocab, K * D);
    write_vector(ofs, alpha, K * K);
    write_vector(ofs, lpqRerankVocabs, rerankK * D * K);
    write_vector(ofs, lopqRerankRotation, D * D * K);
    write_vector(ofs, lopq_residual_mean, K * D);
    ofs.close();
    LOG(INFO) << "out mmu model file:" << model_path;
  }
  bool ReadAndPrecomputeVocabsData() {
    CHECK(!access(coarseCodebookFilename.c_str(),0));
    CHECK(!access(fineCodebookFilename.c_str(),0));
    CHECK(!access(alphaFilename.c_str(),0));
    CHECK(!access(rerankRotationFilename.c_str(),0));
    CHECK(!access(rerankVocabsFilename.c_str(),0));


    lpqs.resize(K);

    if(!lpq_file_prefix.empty()) {
      is_lpq = true;
      LOG(INFO) << "this is a lpq indexer and searcher:"<< lpq_file_prefix;
      lpqRerankVocabs = (float*)malloc(rerankK * D * K * sizeof(float));
      // 读取lpq and lopq
      for(uint64_t i = 0 ; i < K ; i++) {
        string lopq_file = lpq_file_prefix+"_"+std::to_string(i)+".opq_matrix.fvecs";
        string lpq_file = lpq_file_prefix+"_"+std::to_string(i)+".pq.faiss.index";
        if( i== 0 ) {
          if(0 == access(lopq_file.c_str(),R_OK)) {
            is_lopq = true; 
            lopqRerankRotation = (float*) malloc(D * D * K * sizeof(float));
            LOG(INFO) << "this is local o for each ivf lopq";
          } else {
            is_lopq = false; 
            LOG(INFO) << "this is global o, due to file not exist " << lopq_file;
          }
        } 

        if(is_lopq) {
            LOG(INFO) << "loading local o from " << lopq_file;
            CHECK(fvecs_read(lopq_file.c_str(),D,D, lopqRerankRotation + i * D * D) == D) << "read file error "<< lopq_file; 
        }
        LOG(INFO) << "loading local pq from " << lpq_file;
        lpqs[i] = dynamic_cast<faiss::IndexPQ*>(faiss::read_index(lpq_file.c_str())); 
        CHECK(lpqs[i] != nullptr) << " read error from " << lpq_file; 
        CHECK(lpqs[i]->pq.code_size == M);
        CHECK(lpqs[i]->pq.ksub == rerankK && lpqs[i]->pq.dsub*lpqs[i]->pq.M == D);
        CHECK(lpqs[i]->pq.centroids.size() == rerankK * D);

        //每个一级码本类中心训练了一个PQ,拷贝进连续内存,连续内存可能好一些
        memcpy(lpqRerankVocabs + i * rerankK * D,lpqs[i]->pq.centroids.data(), sizeof(float) * rerankK * D);
      }
    }
    if(FLAGS_pq_minus_mean) {
      LOG(INFO) << "Load gopq_residual_mean_file:" << gopq_residual_mean_file;
      gopq_residual_mean = (float*)malloc(sizeof(float) * D);
      CHECK(fvecs_read(gopq_residual_mean_file.c_str(),D,1,gopq_residual_mean) == 1) << "read file error "<< gopq_residual_mean_file; 
      if(is_lopq) {
        lopq_residual_mean = (float*)malloc(sizeof(float) * D * K);
        for(int i = 0 ;i < K ; i++) {
          string lopq_residual_mean_file = lpq_file_prefix+"_"+std::to_string(i)+".lopq_residual_mean.fvecs";
          CHECK(fvecs_read(lopq_residual_mean_file.c_str(), D, 1,lopq_residual_mean + i * D) == 1) << "read file error "<< lopq_residual_mean_file; 
        }
        lopq_residual_mean_rotation = (float*)malloc(sizeof(float) * D * K);
        for(int i = 0; i < K; i++) {
          fmat_mul_full(lopqRerankRotation + i * D * D, lopq_residual_mean + i * D,
                    D, 1, D, "TN", lopq_residual_mean_rotation  + i * D);
        }
      } 
    }
    LOG(INFO) << "start load Faiss PQ " << rerankPQFaissFilename;
    pq = dynamic_cast<faiss::IndexPQ*>(faiss::read_index(rerankPQFaissFilename.c_str()));
    CHECK(pq->pq.code_size == M);
    coarseVocab = (float*) malloc(K * D * sizeof(float));
    fvecs_read(coarseCodebookFilename.c_str(), D, K, coarseVocab);
    fineVocab = (float*) malloc(K * D * sizeof(float));
    fvecs_read(fineCodebookFilename.c_str(), D, K, fineVocab);
    alpha = (float*) malloc(K * K * sizeof(float));
    fvecs_read(alphaFilename.c_str(), K, K, alpha);
    rerankRotation = (float*) malloc(D * D * sizeof(float));
    fvecs_read(rerankRotationFilename.c_str(), D, D, rerankRotation);
    //残差旋转r=q-s-alpha*t; Rr=R(q-s-alpha*t)
    // FLAGS_residual_opq = true 表示coarseVocab fineVocab是用原始矩阵训练的不是o旋转后的矩阵训练的,强制如此
    // is_lopq == false 所有倒排链公用o，于是R*query,R*码本=R*残差，对于同一个query的多个倒排链只旋转一次即可，不用每个倒排链的残差乘一次
    // is_lopq == true  所有倒排链单独o，R*残差，对于每个倒排链的残差单独乘一次
    if(FLAGS_residual_opq == true && is_lopq == false) {
      LOG(INFO) << "[GLOBAL PQ] opq for coarseVocab and fineVocab";
      OpqMatrix(coarseVocab, FLAGS_K);
      OpqMatrix(fineVocab, FLAGS_K);
      if(FLAGS_pq_minus_mean) {
        OpqMatrix(gopq_residual_mean, 1); //也旋转一下均值 R * (q - s - alpha*t  -mu)
      }
    }
    coarseNorms = (float*) malloc(K * sizeof(float));
    fineNorms = (float*) malloc(K * sizeof(float));
    for(int i = 0; i < K; ++i) {
      // 计算类中心的模得的平方/2,用于后续计算距离
      coarseNorms[i] = fvec_norm2sqr(coarseVocab + D * i, D) / 2;
      fineNorms[i] = fvec_norm2sqr(fineVocab + D * i, D) / 2;
    }
    float* temp = (float*) malloc(K * K * sizeof(float));
    fmat_mul_full(coarseVocab, fineVocab,
                  K, K, D, "TN", temp);
    // 计算searcher.coarseFineProducts 以及和二级码本的内积就是文章中的Sk*Tl
    coarseFineProducts = fmat_new_transp(temp, K, K);
    free(temp);
    
    //这里读取公用的PQ
    rerankVocabs = (float*)malloc(rerankK * D * sizeof(float));
    LOG(INFO) << "read " << rerankVocabsFilename << ", num:" << D / M <<",dim:" << M * rerankK;
    fvecs_read(rerankVocabsFilename.c_str(), D / M, M * rerankK, rerankVocabs);
    return true;
  }
  void LoadIDAndPQCodePart(int startId, int count) {
    LOG(INFO) << "LoadIndexPart startId:" << startId << ",count:" << count;
    //std::ifstream inputIndex(rawIndexFilename.c_str(), ios::binary | ios::in);
    //inputIndex.seekg(startId * sizeof(int));
    //std::ifstream inputRerank(rerankFilename.c_str(), ios::binary | ios::in);
    //inputRerank.seekg(startId * sizeof(unsigned char) * M);
    //for(int i = 0; i < count; ++i) {
    //  inputIndex.read((char*)&(index[startId + i].pointId), sizeof(int));
    //  for(int m = 0; m < M; ++m) {
    //    inputRerank.read((char*)&(index[startId + i].bytes[m]), sizeof(unsigned char));
    //  }
    //  if(index[startId + i].pointId < 100) {
    //    LOG(INFO) << "=====codeinfo read index docid:" << index[startId + i].pointId;
    //    gnoimi::print_elements((char*)&(index[startId + i].bytes[0]),M);
    //  }
    //}
    //inputIndex.close();
    //inputRerank.close();
    std::ifstream inputIndex(rawIndexFilename.c_str(), ios::binary | ios::in);
    CHECK(inputIndex.good());
    inputIndex.seekg(startId * sizeof(Record));
    for(int i = 0; i < count; ++i) {
      inputIndex.read((char*)&(index[startId + i]), sizeof(Record));
      if(index[startId + i].pointId < 100) {
        LOG(INFO) << "=====codeinfo read index docid:" << index[startId + i].pointId << ",pos:"<< inputIndex.tellg();
        gnoimi::print_elements((char*)&(index[startId + i].bytes[0]),M);
      }
    }
    inputIndex.close();

  }
  void LoadIDAndPQCode(int N) {
    int perThreadCount = N / threadsCount;
    std::vector<std::thread> threads;
    for (int threadId = 0; threadId < threadsCount; ++ threadId) {
      int startId = threadId * perThreadCount;
      int count = (threadId == threadsCount -1)? (N - startId) : perThreadCount;
      threads.push_back(std::thread(std::bind(&Searcher::LoadIDAndPQCodePart,this, startId, count)));
    }
    for (int threadId = 0; threadId < threadsCount; ++ threadId) {
      threads[threadId].join();
    }  
  }
  bool SaveIndex() {
    CHECK(codes.size() == docnum * FLAGS_M) << "pq code size:"
      << codes.size() << ", want:" << docnum * FLAGS_M;
    LOG(INFO) << "start saving index to " << cellEdgesFilename << "," << rawIndexFilename << ",codesize:" << codes.size(); 
    std::ofstream outputCellEdges(cellEdgesFilename.c_str(), ios::binary | ios::out);
    std::ofstream outIndex(rawIndexFilename.c_str(), ios::binary | ios::out);
    CHECK(outputCellEdges.good());
    CHECK(outIndex.good());
    if(is_lopq) {
      CHECK(term2.size() == docnum) << ",lopq but term2.size() not match " << term2.size(); 
    } else {
      CHECK(term2.size() == 0) << ",lopq but term2.size() not zero " << term2.size(); 
    }
    int cellEdge = 0;
    Record r;
    for(int cellid = 0; cellid < ivf.size(); cellid++) {
       auto& doclist = ivf[cellid];
       cellEdge += doclist.size();
       outputCellEdges.write((char*)&cellEdge,sizeof(int));

       if(doclist.size() != 0) {
        LOG(INFO) << "cell id " << cellid <<",doc num:" << doclist.size();
       } else {
        continue;
       }
       for(auto docid : doclist) {
          r.pointId = docid; 
          if(term2.size() > 0) {
            r.term2 = term2[docid];
          }
          memcpy((char*)&(r.bytes[0]),(char*)(codes.data() + (uint64_t)docid * M), M);
          if(docid < 100) {
            LOG(INFO) << "=====codeinfo save index docid:" << docid;
            gnoimi::print_elements((char*)(codes.data() + docid * M), M);
          }
          outIndex.write((char*)&r,sizeof(r));
       }
       //outIndex.write((char*)doclist.data(),sizeof(int)*doclist.size());
       //for(auto docid : doclist) {
       // outPQencde.write((char*)(pq->codes.data()+docid*M), M);
       // if(docid < 100) {
       //   LOG(INFO) << "=====codeinfo save index docid:" << docid;
       //   gnoimi::print_elements((char*)(pq->codes.data()+docid*M),M);
       // }
       //}
    }
    outputCellEdges.close();
    outIndex.close();
    ivf.clear();
    faiss::write_index(pq,rerankPQFaissWithDataFilename.c_str());
    delete pq;
    LOG(INFO) << "finsih save index total write " << cellEdge << " want " << docnum;
    return true;
  };
  inline uint64_t GetCoarseIdFromCellId(uint64_t cell_id) {
    return cell_id/K;
  }
  inline uint64_t GetFineIdFromCellId(uint64_t cell_id) {
    return cell_id % K;
  }
  // 不能多线程调用
  void AddIndex(int L, uint64_t n, float *x, uint64_t start_id) {
    static vector<vector<int>> cellids;
    static vector<vector<float>> dists;
    static vector<vector<float>> residuals;
    static vector<vector<int>> recall_cell_num_for_each_querys;
    if(start_id == 0) {
        LOG(INFO) << "======ori id 0, L:" << L << ",n:" << n <<",D:" << D;
        gnoimi::print_elements(x,D);
    }
    //预申请本次addIndex的编码空间
    if(codes.size() < (start_id + n) * M ) {
      LOG(INFO) <<"resize codes" << (start_id + n) * M;
      codes.resize((start_id + n) * M);
      if(is_lopq) {
        term2.resize(start_id + n);
      }
    }
    int loopN = 10000;
    int loopNum = (n + loopN - 1)/loopN;
    cellids.resize(loopNum);
    residuals.resize(loopNum);
    dists.resize(loopNum);
    recall_cell_num_for_each_querys.resize(loopNum);

    int nprobe = 1;
    // 找到索引doc对应的cellid
    #pragma omp parallel for if(FLAGS_threadsCount > 1)  num_threads(FLAGS_threadsCount)
    for(uint64_t i = 0; i < loopNum; ++i) {
      int tmp_n = (i==loopNum-1) ? n - (loopNum-1) * loopN : loopN;
      LOG(INFO) << "loopN " << i * loopN * D << ",loopNum:"<<loopNum<<",n:"<< n <<",start_id:"<<start_id <<",tmp_n:"<<tmp_n;
      SearchIvf(tmp_n, x + i * loopN * D, L, nprobe, cellids[i], dists[i], residuals[i],recall_cell_num_for_each_querys[i]);  
    }
    
    //单独起线程建立倒排链
    auto t1 = std::thread( [this,start_id]() {
        uint64_t id = start_id;
        for(auto & loop_cell : cellids) {
          for(auto cellid : loop_cell) {
            ivf[cellid].push_back(id);
            //LOG(INFO) << "[INDEX] doc " << id << ", add to cellid:" << cellid << "["
            //  << cellid/FLAGS_K << "|" << cellid%FLAGS_K << "]";
            ++id;
          }
        }
    });
    //单独起线程计算pq存储,按照id的顺序存储，最后save写磁盘的时候再按照倒排链的规律写
    auto t2 = std::thread( [this,start_id]() {
        int batch_num = residuals.size();
        uint64_t idx = start_id;
        // 检索回来的残差结果
        for(int j = 0; j < batch_num; j++) {
          auto & batch_residual = residuals[j];
          int batch_size = batch_residual.size() / D;

          if(FLAGS_pq_minus_mean) {
            //需要减去均值  
            #pragma omp parallel for
            for(size_t i = 0; i < batch_size; i++) {
              if(is_lopq) {
                //单独o
                minus_mu(D, 1, batch_residual.data() + i * D, lopq_residual_mean + GetCoarseIdFromCellId(cellids[j][i]) * D);
              } else {
                //公用o
                minus_mu(D, 1, batch_residual.data() + i * D, gopq_residual_mean);
              }
            }
          } 
          if(is_lpq) {
            //遍历每一个残差
            #pragma omp parallel for
            for(size_t i = 0; i < batch_size; i++) {
              int cell_id = cellids[j][i];
              if(is_lopq) {
                //单独o
                //根据此残差的倒排链找到对应的o旋转然后加入code
                LopqMatrix(batch_residual.data() + i * D, batch_residual.data() + i * D,1, cell_id);
              } 
              // pq编码
              lpqs[GetCoarseIdFromCellId(cell_id)] -> pq.compute_code(batch_residual.data() + i * D, &codes[(idx + i) * M]);

              if(is_lopq) {
                /*
                    d = || R * (x - y_C - mean) - y_R) ||^2
                      = || R * x - R * y_C  - (R * mean + y_R) ||^2 
                      = || x - y_C ||^2 + || R * mean +  y_R ||^2  + 2 * ( R * y_C | (y_R + R * mean)) - 2 * (R * x | y_R) - 2 * (R * x | R * mean)
                         -------------   ------------------------------------------------------------       -------------       ------------------- 
                            term1                        term2                                                term3               term4
                */
                std::vector<float> y_r(D);
                std::vector<float> y_c(D);
                lpqs[GetCoarseIdFromCellId(cell_id)] -> pq.decode(&codes[(idx + i) * M], y_r.data(), 1);
                if(FLAGS_pq_minus_mean) {
                    faiss::fvec_madd(D, y_r.data(), 1.0f, lopq_residual_mean_rotation + GetCoarseIdFromCellId(cell_id) * D, y_r.data());
                } 
                // y_C
                faiss::fvec_madd(D, coarseVocab + GetCoarseIdFromCellId(cell_id) * D, alpha[cell_id], 
                  fineVocab + GetFineIdFromCellId(cell_id) * D, y_c.data());
                LopqMatrix(y_c.data(), y_c.data(), 1, cell_id);
                // save term2
                term2[idx + i] = faiss::fvec_norm_L2sqr(y_r.data(),D) + 2 * faiss::fvec_inner_product(y_r.data(),y_c.data(), D); 
                // debug
                // LOG(INFO) <<"ADDindex query:"<<idx + i<<",cell_id:"<<cell_id<<",term2:"<<term2[idx + i]<<",dists:"<<dists[j][i];
                // gnoimi::print_elements(&codes[(idx + i) * M],M);
              }

            }
          } else {
            //公用pq 公用o
            pq->pq.compute_codes(batch_residual.data(), &codes[idx * M], batch_size);
          }
          // 计算下一次codes存储的位置
          idx += batch_size;
        }
    } );
    t1.join();
    t2.join();
    //pq->add(n,residuals.data());
    //for(auto cellid:cellids) {
    //  ivf[cellid].push_back(start_id++);    
    //}
    //for(int i = 0; i < n; i++) {
    //  //LOG(INFO) << "AddIndex doc:" << start_id + i << ",add to cell:" << cellids[i*nprobe];
    //  if(start_id + i< 99) {
    //    vector<uint8_t> code(pq->pq.code_size); 
    //    pq->pq.compute_codes(residuals.data()+i*nprobe*D,code.data(),1);
    //    LOG(INFO) << "=====codeinfo add index docid:" << start_id + i<<",size:"<<pq->pq.code_size;
    //    gnoimi::print_elements(residuals.data()+i*nprobe*D,D);
    //    gnoimi::print_elements((char*)code.data(),M);
    //  }
    //}
  }
  
  void LopqMatrix(float* x, float *y, uint64_t n, uint64_t cell_id){
      thread_local vector<float> temp(n * D);
      temp.resize( n * D);
      fmat_mul_full(lopqRerankRotation + GetCoarseIdFromCellId(cell_id) * D * D, x,
                    D, n , D, "TN", temp.data());
      memcpy(y, temp.data(), n * D * sizeof(float));
  }
  void LopqMatrixTranspose(float* x, float*y, uint64_t n, uint64_t cell_id){
      thread_local vector<float> temp(n * D);
      temp.resize( n * D);
      fmat_mul_full(lopqRerankRotation + GetCoarseIdFromCellId(cell_id) * D * D, x,
                    D, n , D, "NN", temp.data());
      memcpy(y, temp.data(), n * D * sizeof(float));
  }
  void OpqMatrix(float* x, uint64_t n){
      thread_local vector<float> temp(n * D);
      temp.resize( n * D);
      fmat_mul_full(rerankRotation, x,
                    D, n , D, "TN", temp.data());
      memcpy(x, temp.data(), n * D * sizeof(float));
  }
  void OpqMatrixTranspose(float* x,uint64_t n){
      thread_local vector<float> temp(n * D);
      temp.resize( n * D);
      fmat_mul_full(rerankRotation, x,
                    D, n , D, "NN", temp.data());
      memcpy(x, temp.data(), n * D * sizeof(float));
  }
  /*
    @param neighborsCount:为了减少无效残差计算传入查询想要的neighborsCount,
      query 的时候空倒排链跳过，已计算倒排链的个数>=neighborsCount 停止计算
    @residuals cellids 在query情况下，只返回倒排链不为空的cellid和其残差，并且达到neighborsCount之后不再计算,
      即residuals cellids的size虽然是queriesCount * nprobe * D 和 queriesCount * nprobe,但实际有意义的只在前几个
  */
  bool SearchIvf(uint64_t queriesCount, float *queries, uint64_t L, int &nprobe, vector<int> &cellids,
      vector<float>& dists, vector<float> &residuals, 
      vector<int> &recall_cell_num_for_each_query, bool is_query = false,
      uint64_t start_id = 0, uint64_t neighborsCount = 0xFFFFFFFFUL) {
    if(nprobe > L * K || nprobe <= 0) {
      nprobe = L * K;
    }
    residuals.resize(queriesCount * nprobe * D);
    cellids.resize(queriesCount * nprobe);
    dists.resize(queriesCount * nprobe);
    recall_cell_num_for_each_query.resize(queriesCount);

    int subDim = D / M;
    std::clock_t c_start = std::clock();
    double t0 = elapsed();
    if(is_lopq == false) {
      //大家公用o的情况下提前对q使用o
      LOG(INFO) << "global OPQ for query";
      OpqMatrix(queries, queriesCount);
    }
    double t1 = elapsed();
    
    double c1 = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;
    double c4 = 0.0;
    double c5 = 0.0;

    double z1 = 0.0;
    double z2 = 0.0;

    for(int qid = 0; qid < queriesCount; ++qid) {
      double t0 = elapsed();
      float* dist = dists.data() + qid * nprobe;
      float* query_residual = residuals.data() + qid * nprobe * D;
      int* candi_cell_id = cellids.data() + qid * nprobe;
    
      thread_local vector<std::pair<float, int> > scores(L * K);
      thread_local vector<std::pair<float, int> > coarseList(K);
      thread_local vector<float> queryCoarseDistance(K);
      thread_local vector<float> queryFineDistance(K);
      // 记录一个query对应的L个q-S 
      // 计算query和各个类中心的内积，用于计算query和一级类中心的距离
      thread_local vector<float> coarseResiduals (L * D);
      // 一级粗排coarseId与其对应排名的对应关系
      thread_local vector<int> coarseIdToTopId(K);

      // 求内积
      fmat_mul_full(coarseVocab, queries + qid * D,
                  K, 1, D, "TN", &(queryCoarseDistance[0]));
      fmat_mul_full(fineVocab, queries + qid * D,
                  K, 1, D, "TN", &(queryFineDistance[0]));
      // 计算query和一级类中心的距离,|x-y|^2/2 = x^2/2 - x*y + y^2/2;y^2/2都一样就不要了，作比较没用
      for(int c = 0; c < K; ++c) {
        coarseList[c].first = coarseNorms[c] - queryCoarseDistance[c] + 0.5; //这里假设query都是归一化的了
        coarseList[c].second = c;
      }
      double t1 = elapsed();
      c1 += (t1 - t0);
      std::partial_sort(coarseList.begin(),coarseList.begin() + L ,coarseList.end());
      double t2 = elapsed();
      c2 += (t2 - t1);

      for(int l = 0; l < L; ++l) {
        int coarseId = coarseList[l].second;
        coarseIdToTopId[coarseId] = l;
        // 统计代码
        if(grouth_cellid.size() != 0) {
          if(grouth_cellid[start_id + qid]/K == coarseId) {
              search_stats.gt_in_LK++; 
              LOG(INFO) << "GT CELL query:" << start_id + qid << ", hit corse cell:" << l;
          }
        } 
        for(int k = 0; k < K; ++k) {
          int cell_id = coarseId * K + k;
          float alphaFactor = alpha[cell_id];
          // 见公式
          scores[l*K+k].first = 2 * (coarseList[l].first + fineNorms[k] * alphaFactor * alphaFactor
                              - queryFineDistance[k] * alphaFactor + coarseFineProducts[cell_id] * alphaFactor);
          scores[l*K+k].second = cell_id;
        }
        if(is_query==false) {
          memcpy(coarseResiduals.data() + l * D, coarseVocab + D * coarseId, D * sizeof(float));
          // 记录一级码本的残差 q-S
          fvec_rev_sub(coarseResiduals.data() + l * D, queries + qid * D, D);
        }
        // 拷贝前L个倒排链的PQ的码本
        //memcpy(preallocatedVocabs + l * rerankK * D, searcher.rerankVocabs + coarseId * rerankK * D, rerankK * D * sizeof(float));
      }
      double t3 = elapsed();
      c3 += (t3 - t2);
      // 倒排链得分排序,取倒排链
      std::partial_sort(scores.begin(), scores.begin() + nprobe, scores.end(),[](const std::pair<float, int> & a, const std::pair<float, int> & b) ->bool{return a.first < b.first;});
      double t4 = elapsed();
      c4 += (t4 - t3);

      //实际返回倒排链的个数
      int recall_doclist_num = 0;
      int recall_doc_num = 0;
      for(int travesered_list_num = 0; travesered_list_num < nprobe; ++travesered_list_num) {
        
        //得到cellid
        int cell_id = scores[travesered_list_num].second;

        if(is_query == true) {
          int last = (cell_id == 0 ? 0 : cellEdges[cell_id-1]);
          //是query情形,过滤空doclist
          if(recall_doc_num >= neighborsCount) {
            break;
          }
          if(cellEdges[cell_id] - last == 0) {
            continue;
          }
          recall_doc_num += (cellEdges[cell_id] - last);
        }
        if(is_query == false) { // query的时候且用分解式计算最终距离的时候，不计算残差
        int topListId = coarseIdToTopId[cell_id / K]; //得到这个cellid对应的粗排的顺序
          //拷贝p-S
          //double t1 = elapsed();
          memcpy(query_residual + recall_doclist_num * D , coarseResiduals.data() + topListId * D, D * sizeof(float));
          //double t2 = elapsed();
          //residual = p -S - alpha*T
          //cblas_saxpy(D, -1.0 * alpha[cell_id], fineVocab + (cell_id % K) * D, 1, query_residual + recall_doclist_num * D, 1);
          faiss::fvec_madd(D, query_residual + recall_doclist_num * D, -1.0 * alpha[cell_id], fineVocab + (cell_id % K) * D, query_residual + recall_doclist_num * D);
        }
        //double t3 = elapsed();
        //z1 += (t2 - t1);
        //z2 += (t3 - t2);
        candi_cell_id[recall_doclist_num] = cell_id;
        dist[recall_doclist_num] = scores[travesered_list_num].first;
        ++recall_doclist_num;
      }
      recall_cell_num_for_each_query[qid] = recall_doclist_num; 
      double t5 = elapsed();
      c5 += (t5 - t4);
    }
    LOG(INFO) << " SearchIvf COST:[" << (t1-t0)*1000/queriesCount <<","<< c1*1000/queriesCount << ","<< c2*1000/queriesCount
      << "," << c3*1000/queriesCount << "," << c4*1000/queriesCount<<","<< c5*1000/queriesCount<<"] ms" << ",memcpy:" << z1*1000/queriesCount << ","
      << "saxpy:" << z2*1000/queriesCount << ",nprobe:" << nprobe;
    return true;
  }
  void minus_mean(float * cell_residual, size_t cell_id) {
      if(is_lopq) {
            //需要减去均值  
           minus_mu(D, 1, cell_residual, lopq_residual_mean + GetCoarseIdFromCellId(cell_id) * D);
      } else {
           minus_mu(D, 1, cell_residual, gopq_residual_mean);
      }
  }
  void compute_inner_prod_table(const float * x, float * dis_table, float *centroid) {
    for(size_t m = 0 ; m < M; ++m) {
      float * t = dis_table + m * rerankK;
      const float * x_sub = x + m * subDim; 
      const float * centroid_sub = centroid + m * rerankK * subDim;
      for(size_t k = 0; k < rerankK; ++k) {
           t[k] = faiss::fvec_inner_product(x_sub, centroid_sub, subDim);
           centroid_sub += subDim;
      }
    }
  }
  void get_centroid(float * x, int cell_id) {
    faiss::fvec_madd(D, coarseVocab + GetCoarseIdFromCellId(cell_id) * D, alpha[cell_id], 
      fineVocab + GetFineIdFromCellId(cell_id) * D, x);
  }
  void decode(float * x, uint8_t *code, int cell_id) {
    float * cellVocab = nullptr;
    if(is_lpq) {
        cellVocab = lpqRerankVocabs + GetCoarseIdFromCellId(cell_id) * D * rerankK;
    } else {
        cellVocab = rerankVocabs;
    }
    for(size_t m = 0; m < M; m++) {
        memcpy(x + m * subDim, cellVocab + m * rerankK * subDim +  code[m] * subDim, subDim * sizeof(float));  
    }
  }
  void reconstruct(float *x, uint8_t *code, int cell_id) {
    size_t coarse_id = GetCoarseIdFromCellId(cell_id);
    size_t fine_id = GetFineIdFromCellId(cell_id);
    decode(x , code, cell_id);
    if(is_lopq) 
        LopqMatrixTranspose(x, x, 1, cell_id);

    if(FLAGS_pq_minus_mean) {
      if(is_lopq) {
          faiss::fvec_madd(D, x, 1.0f, lopq_residual_mean + coarse_id * D, x);
      } else {
          faiss::fvec_madd(D, x, 1.0f, gopq_residual_mean, x);
      }
    }

    for(size_t i = 0; i < D; ++i) {
        x[i] = x[i] + coarseVocab[coarse_id * D + i] + alpha[cell_id] * fineVocab[fine_id * D + i]; 
    }
   
    if(is_lopq == false)
        OpqMatrixTranspose(x, 1); 
  }
  uint64_t SearchNearestNeighbors(float* x,
                              int n, int L, int nprobe, int neighborsCount, 
                              vector<vector<std::pair<float, int> > >& result, uint64_t start_id, uint64_t topk) {
    result.resize(n);
    //result查询结果每个query保存neighborsCount个结果
    thread_local vector<int> cellids;
    thread_local vector<float> residuals;
    thread_local vector<int> recall_cell_num_for_each_query; // query的时候对于空倒排连不计算残差了，所以返回的cell个数<=nprobe
    thread_local vector<float> dists;

    // LOG(INFO) << "query vec Squared L2 norm " << faiss::fvec_norm_L2sqr(x,D);

    // 根据neighborsCount预估需要遍历倒排链的个数主要是为了减少std::partial_sort(L*K)的时间
    double t0 = elapsed();
    SearchIvf(n, x, L, nprobe, cellids, dists, residuals, recall_cell_num_for_each_query, true,start_id,neighborsCount);  
    double t1 = elapsed();

    //for(int i = 0; i < n; i++) {
    //  LOG(INFO) << "QueryIndex doc:" << start_id + i << ",search to cell:" << cellids[i*nprobe] << ",nprobe:"<<nprobe;
    //  if((start_id + i) % 10000 == 0) {
    //    LOG(INFO) << "======query id:" << start_id + i << ",residual===";
    //    gnoimi::print_elements(residuals.data()+i*nprobe*D,D);
    //  }
    //}
    uint64_t travel_doc = 0;
    float* cellVocab = rerankVocabs;


    for(int qid = 0; qid < n; ++qid) {
        //针对每个pq建立一个距离表缓存
        std::map<int,vector<float>> term34_table;

        result[qid].resize(neighborsCount, std::make_pair(std::numeric_limits<float>::max(), -1));
        float* query_residual = residuals.data() + qid * nprobe * D;
        int* candi_cell_id = cellids.data() + qid * nprobe;
        int found = 0;
        int check_cell_num = 0;
        int empty_cell_num = 0;
        //遍历多个倒排链
        for(int i = 0; i < recall_cell_num_for_each_query[qid]; i++) {
          //得到倒排链数据
          int cell_id = candi_cell_id[i];
          int cellStart = (cell_id == 0) ? 0 : cellEdges[cell_id - 1];
          int cellFinish = cellEdges[cell_id];
          LOG(INFO)<<"query nprobe i "<< i <<" cellid:" <<cell_id <<",dis:"<<dists[i];

          if(found >= neighborsCount) {
              break;
          }

          if(cellStart == cellFinish) {
            empty_cell_num++;
            continue;
          }
          check_cell_num++;
          #ifdef GNOIMI_QUERY_DEBUG
          if(grouth_cellid.size() != 0) {
            if(grouth_cellid[start_id + qid] == cell_id) {
              search_stats.gt_in_retrievaled++; 
              LOG(INFO) << "GT CELL query:" << start_id + qid << ", hit retrieval cell:" << cell_id;
            }
          }
          #endif
          float* cell_residual = query_residual + i * D;
          if(is_lpq) {
            //LOG(INFO) << "[LPQ] lpq";
            //每个倒排链单独pq
            cellVocab = lpqRerankVocabs + GetCoarseIdFromCellId(cell_id) * D * rerankK;
          }

          std::unique_ptr<faiss::DistanceComputer> dis_computer;
          //TODO:@zhangcunyi 长倒排链用查标累加法，短倒排链直接计算这个临界值我觉得大概是rerankK*2 = 512
          if(use_precomputed_table == 0 && cellFinish - cellStart >= compute_table_docnum_threshold && neighborsCount - found >= compute_table_docnum_threshold) {
              if(FLAGS_pq_minus_mean) {
                  minus_mean(cell_residual,cell_id);
              }
              if(is_lopq) LopqMatrix(cell_residual, cell_residual, 1, cell_id);
              // debug 
              //gnoimi::print_elements(cell_residual, D);

              thread_local vector<float> table(M * rerankK);
              gnoimi::compute_distance_table(cell_residual, cellVocab, table.data(), M, subDim, rerankK);
              for(int id = cellStart; id < cellFinish && found < neighborsCount; ++id) {
                result[qid][found].second = index[id].pointId;
                result[qid][found].first = 0.0;
                float sum0 = 0.0; float sum1 = 0.0; float sum2 = 0.0; float sum3 = 0.0;
                // M是4的倍数这个是强制条件初始化的时候会CHECK
                unsigned char* bytes = index[id].bytes;
                for(size_t m = 0; m < M; m += 4) {
                    sum0 += table[rerankK * m + bytes[m]];
                    sum1 += table[rerankK * (m + 1) + bytes[m + 1]];
                    sum2 += table[rerankK * (m + 2) + bytes[m + 2]];
                    sum3 += table[rerankK * (m + 3) + bytes[m + 3]];
                }
                result[qid][found].first = sum0 + sum1 + sum2 + sum3;
                ++found;
                ++search_stats.table_dis_doc; 
              }
          } else if (is_lopq && use_precomputed_table == 1 && (term34_table.count(GetCoarseIdFromCellId(cell_id)) > 0 
              || (cellFinish - cellStart > compute_table_docnum_threshold_term3 && neighborsCount - found >= compute_table_docnum_threshold_term3) )){

              float term1 = dists[i]; 
              /*
              {
                // 验证term1 正确
                vector<float> yr(D);
                get_centroid(yr.data(),cell_id);
                float term11 = faiss::fvec_L2sqr(x + qid * D, yr.data(), D); 
                faiss::fvec_madd(D, x + qid * D, -1.0f, yr.data(), yr.data());
                gnoimi::print_elements(yr.data(),D);
                gnoimi::print_elements(cell_residual,D);
                LOG(INFO) <<"term1:"<<term1<<","<< faiss::fvec_norm_L2sqr(yr.data(), D)<<","<<term11;
              }
              */
              
              //* 验证 dis时才需要用到，需提前计算的
              /*
              if(FLAGS_pq_minus_mean) {
                  minus_mean(cell_residual, cell_id);
              }
              if(is_lopq) LopqMatrix(cell_residual, cell_residual, 1, cell_id);
              vector<float> rx(D);
              LopqMatrix( x + qid * D, rx.data(), 1, cell_id);
              */
              // 用分解
              //如果之前计算过就复用以前的，否则重新计算term3的table \ term4
              if(term34_table.count(GetCoarseIdFromCellId(cell_id)) == 0) {
                  vector<float> rx(D);
                  LopqMatrix( x + qid * D, rx.data(), 1, cell_id);
                  term34_table.emplace(std::piecewise_construct, std::forward_as_tuple(GetCoarseIdFromCellId(cell_id)), std::forward_as_tuple(rerankK * M + 1)); // + 1多出来一个float存放term4
                  compute_inner_prod_table( rx.data(), term34_table[GetCoarseIdFromCellId(cell_id)].data(), cellVocab);
                  float term4 = faiss::fvec_inner_product(rx.data(), lopq_residual_mean_rotation + GetCoarseIdFromCellId(cell_id) * D, D);
                  term34_table[GetCoarseIdFromCellId(cell_id)][rerankK * M] = term4;
              }
              const vector<float> & table = term34_table[GetCoarseIdFromCellId(cell_id)];
              for(int id = cellStart; id < cellFinish && found < neighborsCount; ++id) {
                result[qid][found].second = index[id].pointId;
                result[qid][found].first = 0.0f;
                float sum0 = 0.0; float sum1 = 0.0; float sum2 = 0.0; float sum3 = 0.0;
                unsigned char* bytes = index[id].bytes;
                for(size_t m = 0; m < M; m += 4) {
                    sum0 += table[rerankK * m + bytes[m]];
                    sum1 += table[rerankK * (m + 1) + bytes[m + 1]];
                    sum2 += table[rerankK * (m + 2) + bytes[m + 2]];
                    sum3 += table[rerankK * (m + 3) + bytes[m + 3]];
                }
                /*
                {
                  //验证term3
                  vector<float> d(D);
                  decode(d.data(), bytes, cell_id);
                  LOG(INFO) <<"term3:"<<sum0 + sum1 + sum2 + sum3 <<","<<faiss::fvec_inner_product(rx.data(),d.data(),D);
                }
                */
                result[qid][found].first = term1 + index[id].term2 - 2 * (sum0 + sum1 + sum2 + sum3) - 2 * table[rerankK * M];
                /*{
                  // NOTICE:如果用分解式计算距离，query的时候不需要计算残差，searchIVF为了加快计算可能没有返回残差所以如果需要debug
                  // 计算dis2 则需要修改一下返回残差

                  // 验证 dis 
                  float dis2 = 0.0f;
                  for(size_t m = 0; m < M; ++m) {
                      float* codeword = cellVocab + m * rerankK * subDim + index[id].bytes[m] * subDim;
                      float* residualSubvector = cell_residual + m * subDim;
                      dis2 += faiss::fvec_L2sqr(residualSubvector, codeword, subDim);
                  }
                  vector<float> r(D);
                  reconstruct(r.data(), bytes, cell_id);
                  LOG(INFO) << "docid:"<< result[qid][found].second <<",decompostion dis1:"<<result[qid][found].first<<",dis2:"<<dis2<<",dis3:"<<faiss::fvec_L2sqr(x, r.data(), D);
                }*/
                ++found;
                ++search_stats.table_dis_doc; 
              }
          } else {
            if(FLAGS_pq_minus_mean) {
                minus_mean(cell_residual,cell_id);
            }
            if(is_lopq) LopqMatrix(cell_residual,cell_residual, 1, cell_id);
            for(int id = cellStart; id < cellFinish && found < neighborsCount; ++id) {
                result[qid][found].second = index[id].pointId;
                result[qid][found].first = 0.0;
                //实时计算查询向量和索引编码的距离
                for(size_t m = 0; m < M; ++m) {
                  float* codeword = cellVocab + m * rerankK * subDim + index[id].bytes[m] * subDim;
                  float* residualSubvector = cell_residual + m * subDim;
                  //for(int d = 0; d < subDim; ++d) {
                  //  // 实时计算查询向量和对应编码类中心的距离
                  //  float diff = residualSubvector[d] - codeword[d];
                  //  result[qid][found].first += diff * diff;
                  //}
                  result[qid][found].first += faiss::fvec_L2sqr(residualSubvector, codeword, subDim);
                }
                ++found;
            }
          }
        }
        if(topk + 500 >= result[qid].size()) {
          std::sort(result[qid].begin(), result[qid].end());
        } else {
          std::partial_sort(result[qid].begin(), result[qid].begin() + topk , result[qid].end());
        }
        //LOG(INFO) << "query " << qid << " check_cell_num "<< check_cell_num << " empty_cell_num "
        //  << empty_cell_num << " found " << found ;
        travel_doc += found;
        search_stats.nprobe += check_cell_num; 
        search_stats.inner_table_num += term34_table.size();
    }
    double t2 = elapsed();
    #ifdef GNOIMI_QUERY_DEBUG
    LOG(INFO) << "search end, query n:" << n <<", ivf cost:" << (t1-t0)*1000/n <<"ms, pq cost:" << (t2-t1)*1000/n << "ms,total:" << (t2-t0)*1000/n;
    #endif
    search_stats.pq_cost += (t2-t1);
    search_stats.ivf_cost += (t1-t0);
    return travel_doc;
}
};




#if 0
void SearchNearestNeighbors_MultiPQ(const Searcher& searcher,
                            const float* queries,
                            int neighborsCount,
                            vector<vector<std::pair<float, int> > >& result) {
  result.resize(queriesCount);
  vector<float> queryCoarseDistance(searcher.K);
  vector<float> queryFineDistance(searcher.K);
  vector<std::pair<float, int> > coarseList(searcher.K);
  //result查询结果每个query保存neighborsCount=5000个结果
  for(int qid = 0; qid < queriesCount; ++qid) {
    result[qid].resize(neighborsCount, std::make_pair(std::numeric_limits<float>::max(), 0));
  }
  vector<int> topPointers(searcher.L);
  vector<float> coarseResiduals (searcher.L * searcher.D);
  float* residual = (float*)malloc(searcher.D * sizeof(float));
  float* preallocatedVocabs = (float*)malloc(searcher.L * searcher.rerankK * searcher.D * sizeof(float));//本次查询前L个倒排链的PQ的码本
  int subDim = searcher.D / searcher.M;
  vector<std::pair<float, int> > scores(searcher.L * searcher.K);
  // 一级粗排coarseId与其对应排名的对应关系
  vector<int> coarseIdToTopId(searcher.K);
  std::clock_t c_start = std::clock();
  for(int qid = 0; qid < queriesCount; ++qid) {
    cout << qid << "\n";
    int found = 0;
    // 计算query和各个类中心的内积，用于计算query和一级类中心的距离
    fmat_mul_full(searcher.coarseVocab, queries + qid * searcher.D,
                  searcher.K, 1, searcher.D, "TN", &(queryCoarseDistance[0]));
    fmat_mul_full(searcher.fineVocab, queries + qid * D,
                  searcher.K, 1, searcher.D, "TN", &(queryFineDistance[0]));
    // 计算query和一级类中心的距离,|x-y|/2=x^2/2-x*y+y^2/2;y^2/2都一样就不要了，作比较没用
    for(int c = 0; c < K; ++c) {
      coarseList[c].first = searcher.coarseNorms[c] - queryCoarseDistance[c];
      coarseList[c].second = c;
    }
    std::sort(coarseList.begin(), coarseList.end());
    for(int l = 0; l < searcher.L; ++l) {
      int coarseId = coarseList[l].second;
      coarseIdToTopId[coarseId] = l;
      for(int k = 0; k < searcher.K; ++k) {
        int cell_id = coarseId * searcher.K + k;
        float alphaFactor = searcher.alpha[cell_id];
        // 见公式
        scores[l*K+k].first = coarseList[l].first + searcher.fineNorms[k] * alphaFactor * alphaFactor
                              - queryFineDistance[k] * alphaFactor + searcher.coarseFineProducts[cell_id] * alphaFactor;
        scores[l*K+k].second = cell_id;
      }
      memcpy(coarseResiduals.data() + l * searcher.D, searcher.coarseVocab + searcher.D * coarseId, searcher.D * sizeof(float));
      // 记录一级码本的残差 q-S
      fvec_rev_sub(coarseResiduals.data() + l * searcher.D, queries + qid * searcher.D, searcher.D);
      // 拷贝前L个倒排链的PQ的码本
      memcpy(preallocatedVocabs + l * rerankK * D, searcher.rerankVocabs + coarseId * rerankK * D, rerankK * D * sizeof(float));
    }
    int cellsCount = neighborsCount * ((float)K * K / N);
    std::nth_element(scores.begin(), scores.begin() + cellsCount, scores.end());
    // 倒排链得分排序,取倒排链
    std::sort(scores.begin(), scores.begin() + cellsCount);
    int currentPointer = 0;
    int cellTraversed = 0;
    while(found < neighborsCount) {
      cellTraversed += 1;
      //得到一级和二级码本
      int cell_id = scores[currentPointer].second;
      int topListId = coarseIdToTopId[cell_id / K];
      ++currentPointer;
      //得到倒排链数据
      int cellStart = (cell_id == 0) ? 0 : searcher.cellEdges[cell_id - 1];
      int cellFinish = searcher.cellEdges[cell_id];
      if(cellStart == cellFinish) {
        continue;
      }
      memcpy(residual, coarseResiduals.data() + topListId * D, D * sizeof(float));
      cblas_saxpy(D, -1.0 * searcher.alpha[cell_id], searcher.fineVocab + (cell_id % K) * D, 1, residual, 1);
      float* cellVocab = preallocatedVocabs + topListId * rerankK * D;
      // 计算 residual = p - S - a* T,即query在此倒排链的残差 公式(4)
      for(int id = cellStart; id < cellFinish && found < neighborsCount; ++id) {
        result[qid][found].second = searcher.index[id].pointId;
        result[qid][found].first = 0.0;
        float diff = 0.0;
        for(int m = 0; m < M; ++m) {
          float* codeword = cellVocab + m * rerankK * subDim + searcher.index[id].bytes[m] * subDim;
          float* residualSubvector = residual + m * subDim;
          for(int d = 0; d < subDim; ++d) {
            // 实时计算查询向量和对应编码类中心的距离
            diff = residualSubvector[d] - codeword[d];
            result[qid][found].first += diff * diff;
          }
        }
        ++found;
      }
    }
    std::sort(result[qid].begin(), result[qid].end());
  }
  std::clock_t c_end = std::clock();
  std::cout << std::fixed << std::setprecision(2) << "CPU time used: "
              << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / queriesCount << " ms\n";
}
#endif

float computeRecall1At(const vector<vector<std::pair<float, int> > >& result,
                      const int* groundtruth, int R, uint64_t gt_d) {
  int limit = (R < result[0].size()) ? R : result[0].size();
  int positive = 0;
  for(int i = 0; i < result.size(); ++i) {
    bool hit = false;
    for(int j = 0; j < limit; ++j) {
      if(result[i][j].second == *(groundtruth + i * gt_d)) {
        ++positive;
        hit = true;
      }
    }
    if(!hit) {
      //LOG(INFO) << "[GT MISS] query " << i << ", want " << *(groundtruth + i * gt_d) << ", @" << limit;
    }
  }
  return (float(positive) / result.size());
}

//检索结果top R有在gt的top R中的比例
float computeRecallTopRinGT(const vector<vector<std::pair<float, int> > >& result,
                      const int* groundtruth, int R, uint64_t gt_d) {
  int limit = (R < result[0].size()) ? R : result[0].size();
  uint64_t hit_num = 0;
  // 遍历所有query
  for(int i = 0; i < result.size(); ++i) {
    // 遍历query的top limit个结果
    for(int j = 0; j < limit; ++j) {
      // 看看在不在该query的top limit中
      for(int k = 0; k < limit; ++k) {
        if(result[i][j].second == groundtruth[i*gt_d +k]) {
          hit_num++;
          break;
        }
      }
    }
  }
  return (float(hit_num) / result.size() / limit);
}
void ComputeRecall(const vector<vector<std::pair<float, int> > >& result,
                   const int* groundtruth, uint64_t gt_d) {
    int R = 1;
    std::cout << std::fixed << std::setprecision(5) << "R@" << R << ":" << 
                 computeRecall1At(result, groundtruth, R, gt_d) << "\n";
    R = 10;
    std::cout << std::fixed << std::setprecision(5) << "R@" << R << ":" << 
                 computeRecall1At(result, groundtruth, R, gt_d) << "\n";
    R = 100;
    std::cout << std::fixed << std::setprecision(5) << "R@" << R << ":" << 
                 computeRecall1At(result, groundtruth, R, gt_d) << "\n";

    std::cout << std::fixed << std::setprecision(5) << "SameInTop" << 100 << ":" << 
                 computeRecallTopRinGT(result, groundtruth, 100, gt_d) << "\n";
}

int main(int argc, char** argv) {
    gnoimi::print_elements(argv,argc);

    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true; //日志全部输出到标准错误

    if(FLAGS_nprobe <=0 || FLAGS_nprobe > FLAGS_K * FLAGS_L) {
      FLAGS_nprobe = FLAGS_K * FLAGS_L;
    }
    CHECK(M == FLAGS_M);
    CHECK(FLAGS_rerankK == 256) << "rerankK must = 256";
    CHECK(FLAGS_L >= 0 && FLAGS_L <= FLAGS_K );
    CHECK(FLAGS_neighborsCount > FLAGS_topk);
    LOG(INFO) << "sizeof Record " << sizeof(Record);

    // 初始化
    Searcher searcher(FLAGS_model_prefix,FLAGS_pq_prefix,FLAGS_index_prefix,FLAGS_lpq_file_prefix,FLAGS_D,FLAGS_K,FLAGS_M,FLAGS_rerankK,FLAGS_threadsCount);

    if( FLAGS_make_index) {
      LOG(INFO) << "making index";
      CHECK(!FLAGS_base_file.empty()) << "need set file base_file";
      LOG(INFO) << "start indexing";

      size_t file_d = 0;
      size_t file_n = 0;
      CHECK(gnoimi::bfvecs_fsize(FLAGS_base_file.c_str(),&file_d, &file_n) > 0);
      searcher.docnum = FLAGS_N == 0 ? file_n : std::min(file_n,FLAGS_N);
      if(searcher.docnum > 0) {
        //预申请空间
        for(auto & i:searcher.ivf){
          i.reserve((searcher.docnum+1000000)/(searcher.K*searcher.K));
        }
      }
      searcher.codes.resize(searcher.docnum * M);
      if(searcher.is_lopq) {
        searcher.term2.resize(searcher.docnum);
      }
      LOG(INFO) << " BASE file " << FLAGS_base_file <<", get docnum:" << searcher.docnum <<", d:" << file_d
        <<", resize pq codes:" << searcher.docnum * M;
      size_t input_D = 0;
      gnoimi::b2fvecs_read_callback(FLAGS_base_file.c_str(),
         input_D, searcher.docnum, 1000000,std::bind(&Searcher::AddIndex, &searcher, FLAGS_L, std::placeholders::_1, 
         std::placeholders::_2, std::placeholders::_3),FLAGS_normalize_before_read);
      searcher.D = input_D;
      LOG(INFO) << "end index,read " << searcher.docnum << ",want " << FLAGS_N;
      searcher.SaveIndex();
      return 0;
    }
    CHECK(!access(FLAGS_groud_truth_file.c_str(),0)  && !access(FLAGS_query_filename.c_str(),0));
    searcher.LoadIndex();
    //searcher.LoadPQWithData();

    // 读取query
    uint64_t queriesCount = FLAGS_queriesCount;
    auto queries =  gnoimi::read_bfvecs(FLAGS_query_filename.c_str(), FLAGS_D, queriesCount,FLAGS_normalize_before_read);
    LOG(INFO) << "L2 norm " << faiss::fvec_norm_L2sqr(queries.get(), FLAGS_D);
    LOG(INFO) << "read "<< queriesCount <<" doc from " << FLAGS_query_filename << ", d:" << FLAGS_D;

    // 读取groud truth
    uint64_t gt_d = 0;
    uint64_t gt_num = queriesCount;
    auto groundtruth = gnoimi::ivecs_read(FLAGS_groud_truth_file.c_str(),&gt_d,&gt_num);
    LOG(INFO) << "Groundtruth is read";
    CHECK(gt_d > 0);
    // 统计grouth truth 所在的cellid
    std::unordered_map<int,int> gts_cellid;
    if(FLAGS_print_gt_in_LK){
      FLAGS_threadsCount = 1;
      grouth_cellid.resize(queriesCount,-1); //记录每个groudth在哪个cell当中  
      for(int i = 0; i< queriesCount; i++) {
        gts_cellid.insert({groundtruth.get()[gt_d * i],-1}); 
      }
      int last_cell = 0;
      for(int i = 0; i < FLAGS_K * FLAGS_K ; i++) {
        if(searcher.cellEdges[i] - last_cell != 0) {
          for(int j = last_cell; j < searcher.cellEdges[i]; j++) {
            auto it = gts_cellid.find(searcher.index[j].pointId);
            if(it != gts_cellid.end()) {
              it->second = i;
            }
          }
        }
        last_cell = searcher.cellEdges[i];
      }
      for(int i = 0; i< queriesCount; i++) {
        grouth_cellid[i] = gts_cellid[groundtruth.get()[gt_d * i]];
        //LOG(INFO) << "FIND GT CELL query " << i << ", gt:" << groundtruth.get()[gt_d * i] << ", in cell:" << grouth_cellid[i];
      }
    }
    
    // query
    
    double t0 = elapsed();
    {
      float * z = new float[FLAGS_D * FLAGS_D];
      fmat_mul_full(searcher.rerankRotation,searcher.rerankRotation,
                  FLAGS_D, FLAGS_D, FLAGS_D, "TN", z);
      gnoimi::print_elements(z,2*FLAGS_D);
      delete[] z;
    }
    vector<vector<std::pair<float, int> > > results(queriesCount);
    int nprobe = (FLAGS_nprobe == 0) ? (FLAGS_neighborsCount * FLAGS_K * FLAGS_K)/ searcher.docnum + 100 : FLAGS_nprobe;
    #pragma omp parallel for num_threads(FLAGS_threadsCount)
    for(uint64_t i = 0 ; i< queriesCount; i++) {
      vector<vector<std::pair<float, int> > > result;
      search_stats.travel_doc += searcher.SearchNearestNeighbors(queries.get() + i * FLAGS_D, 1, FLAGS_L, nprobe ,FLAGS_neighborsCount, result, i, FLAGS_topk);
      results[i]  = result[0];
      LOG(INFO) << "query finish " << i << ", dis:"<< result[0][0].first <<",docid:" << result[0][0].second;
    }
    // debug
    //searcher.AddIndex(FLAGS_L,queriesCount,queries.get(),0);


    double t1 = elapsed();
    std::cout.setf(ios::fixed);//precision控制小数点后的位数
    std::cout.precision(2);//后2位
    std::cout << "[COST] ["<< (t1-t0) << " s] query num:" << queriesCount << ",thread_num:" << FLAGS_threadsCount << ",query cost:" << (t1-t0)*1000/queriesCount <<" ms" <<",nprobe:" << nprobe
      << ",L:"<< FLAGS_L <<",top:"<< FLAGS_neighborsCount<< ",travel_doc:"<< search_stats.travel_doc*1.0/queriesCount
      << ",table_dis_doc:"<<search_stats.table_dis_doc*1.0/queriesCount<<",inner_table_num:"<<search_stats.inner_table_num*1.0/queriesCount 
      << ",ivf cost:" << search_stats.ivf_cost*1000/queriesCount<<"ms,pq cost:"<< search_stats.pq_cost*1000/queriesCount
      << "ms,check ivfs:"<< search_stats.nprobe/queriesCount <<",opq cost:"<<search_stats.opq_cost*1000/queriesCount <<"ms\n";
    std::cout.precision(4);//后4位
    if(FLAGS_print_gt_in_LK) {
      std::cout << "GT cell hit result. query num:" << queriesCount << ",gt_in_LK:" << search_stats.gt_in_LK*1.0/queriesCount
        << ",gt_in_retrievaled:" << search_stats.gt_in_retrievaled*1.0/queriesCount << "\n";
    }
    ComputeRecall(results, groundtruth.get(),gt_d); 
    
    //保存查询结果
    {
      char query_result_ctr[2048];
      sprintf(query_result_ctr,"%s.k%d.l%d.ne%d.np%d.queryresult.ivecs",
        FLAGS_index_prefix.c_str(),FLAGS_K,FLAGS_L,
        FLAGS_neighborsCount,FLAGS_nprobe);
      string query_result(query_result_ctr);

      vector<int> qr(queriesCount * FLAGS_topk);
      for(int i = 0; i < results.size(); i++) {
        CHECK(results[i].size() >= FLAGS_topk) <<"query " << i <<", candi num:" << results[i].size() 
          << " < " << FLAGS_topk;
          for(int j = 0; j< FLAGS_topk;j++){
            qr[i * FLAGS_topk + j] = results[i][j].second;
          }
      }
      ivecs_write(query_result.c_str(), FLAGS_topk, queriesCount ,qr.data());
      LOG(INFO) << "write query result to " << query_result;
    }
    return 0;
}
