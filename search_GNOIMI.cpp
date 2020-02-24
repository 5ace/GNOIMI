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
#include <faiss/index_io.h>

using std::cout;
using std::ios;
using std::string;
using std::vector;
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
DEFINE_bool(normalize_before_read,true,"读取索引和查询是否归一化");


std::vector<int> grouth_cellid;
struct SearchStats {
  int gt_in_LK = 0; // gt在 L个倒排链中的query个数
  int gt_in_retrievaled = 0; // gt被遍历过的query 个数
  double ivf_cost = 0.0; //ms
  double pq_cost = 0.0; //ms
  double opq_cost = 0.0;
  uint64_t travel_doc = 0;
  uint64_t nprobe = 0;

  void Reset() {
    gt_in_LK = 0;
    gt_in_retrievaled = 0;
    ivf_cost = 0.0;
    pq_cost = 0.0;
    opq_cost = 0.0;
    travel_doc = 0;
    nprobe = 0;
  }
} search_stats;

const uint64_t M = 16;
struct Record {
  int pointId;
  unsigned char bytes[M];
};

struct Searcher {
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
  string model_prefix_,index_prefix_,pq_prefix_;
  uint64_t D;
  uint64_t K;
  uint64_t M;
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
  faiss::IndexPQ* pq = nullptr; // global的pq
  bool is_lopq = false; //是否每个倒排链独立o
  bool is_lpq = false; //是否每个倒排链独立pq
  std::vector<faiss::IndexPQ*> lpqs; //每个倒排链各自存储的pq 是个数组
  uint64_t compute_table_docnum_threshold;//查询时倒排链长度大于这个就用查表法

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
    LOG(INFO) << "Searcher construct ok,is_lpq:"<< is_lpq <<",is_lopq:" << is_lopq <<", compute_table_docnum_threshold:"<< compute_table_docnum_threshold;
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
  // 不能多线程调用
  void AddIndex(int L, uint64_t n, float *x, uint64_t start_id) {
    static vector<vector<int>> cellids;
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
    }
    int loopN = 10000;
    int loopNum = (n + loopN - 1)/loopN;
    cellids.resize(loopNum);
    residuals.resize(loopNum);
    recall_cell_num_for_each_querys.resize(loopNum);

    int nprobe = 1;
    #pragma omp parallel for if(FLAGS_threadsCount > 1)  num_threads(FLAGS_threadsCount)
    for(uint64_t i = 0; i < loopNum; ++i) {
      int tmp_n = (i==loopNum-1) ? n - (loopNum-1) * loopN : loopN;
      LOG(INFO) << "loopN " << i * loopN * D << ",loopNum:"<<loopNum<<",n:"<< n <<",start_id:"<<start_id <<",tmp_n:"<<tmp_n;
      SearchIvf(tmp_n, x + i * loopN * D, L, nprobe, cellids[i], residuals[i],recall_cell_num_for_each_querys[i]);  
    }
   
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
              if(is_lopq) {
                //单独o
                //根据此残差的倒排链找到对应的o旋转然后加入code
                LopqMatrix(batch_residual.data() + i * D, 1, GetCoarseIdFromCellId(cellids[j][i]));
              } 
              //pq编码
              lpqs[GetCoarseIdFromCellId(cellids[j][i])] -> pq.compute_code(batch_residual.data() + i * D,
                &codes[(idx + i) * M]);
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
  
  void LopqMatrix(float* x,uint64_t n, uint64_t k){
      thread_local vector<float> temp(n * D);
      temp.resize( n * D);
      fmat_mul_full(lopqRerankRotation + k * D * D, x,
                    D, n , D, "TN", temp.data());
      memcpy(x, temp.data(), n * D * sizeof(float));
  }
  void OpqMatrix(float* x,uint64_t n){
      thread_local vector<float> temp(n * D);
      temp.resize( n * D);
      fmat_mul_full(rerankRotation, x,
                    D, n , D, "TN", temp.data());
      memcpy(x, temp.data(), n * D * sizeof(float));
  }
  /*
    @param neighborsCount:为了减少无效残差计算传入查询想要的neighborsCount,
      query 的时候空倒排链跳过，已计算倒排链的个数>=neighborsCount 停止计算
    @residuals cellids 在query情况下，只返回倒排链不为空的cellid和其残差，并且达到neighborsCount之后不再计算,
      即residuals cellids的size虽然是queriesCount * nprobe * D 和 queriesCount * nprobe,但实际有意义的只在前几个
  */
  bool SearchIvf(uint64_t queriesCount, float *queries, uint64_t L, int &nprobe, vector<int> &cellids,vector<float> &residuals, 
      vector<int> &recall_cell_num_for_each_query, bool is_filter_empty_cell = false,
      uint64_t start_id = 0, uint64_t neighborsCount = 0xFFFFFFFFUL) {
    if(nprobe > L * K || nprobe <= 0) {
      nprobe = L * K;
    }
    residuals.resize(queriesCount * nprobe * D);
    cellids.resize(queriesCount * nprobe);
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
      // 计算query和一级类中心的距离,|x-y|/2=x^2/2-x*y+y^2/2;y^2/2都一样就不要了，作比较没用
      for(int c = 0; c < K; ++c) {
        coarseList[c].first = coarseNorms[c] - queryCoarseDistance[c];
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
          int cellId = coarseId * K + k;
          float alphaFactor = alpha[cellId];
          // 见公式
          scores[l*K+k].first = coarseList[l].first + fineNorms[k] * alphaFactor * alphaFactor
                              - queryFineDistance[k] * alphaFactor + coarseFineProducts[cellId] * alphaFactor;
          scores[l*K+k].second = cellId;
        }
        memcpy(coarseResiduals.data() + l * D, coarseVocab + D * coarseId, D * sizeof(float));
        // 记录一级码本的残差 q-S
        fvec_rev_sub(coarseResiduals.data() + l * D, queries + qid * D, D);
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
        int cellId = scores[travesered_list_num].second;

        if(is_filter_empty_cell == true) {
          int last = (cellId == 0 ? 0 : cellEdges[cellId-1]);
          //是query情形
          if(recall_doc_num >= neighborsCount) {
            break;
          }
          if(cellEdges[cellId] - last == 0) {
            continue;
          }
          recall_doc_num += (cellEdges[cellId] - last);
        }

        int topListId = coarseIdToTopId[cellId / K]; //得到这个cellid对应的粗排的顺序
        //拷贝p-S
        //double t1 = elapsed();
        memcpy(query_residual + recall_doclist_num * D , coarseResiduals.data() + topListId * D, D * sizeof(float));
        //double t2 = elapsed();
        //residual = p -S - alpha*T
        cblas_saxpy(D, -1.0 * alpha[cellId], fineVocab + (cellId % K) * D, 1, query_residual + recall_doclist_num * D, 1);
        //double t3 = elapsed();
        //z1 += (t2 - t1);
        //z2 += (t3 - t2);
        candi_cell_id[recall_doclist_num] = cellId;
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
  uint64_t SearchNearestNeighbors(float* x,
                              int n, int L, int nprobe, int neighborsCount, 
                              vector<vector<std::pair<float, int> > >& result, uint64_t start_id, uint64_t topk) {
    result.resize(n);
    //result查询结果每个query保存neighborsCount个结果
    thread_local vector<int> cellids;
    thread_local vector<float> residuals;
    thread_local vector<int> recall_cell_num_for_each_query; // query的时候对于空倒排连不计算残差了，所以返回的cell个数<=nprobe

    // LOG(INFO) << "query vec Squared L2 norm " << faiss::fvec_norm_L2sqr(x,D);

    // 根据neighborsCount预估需要遍历倒排链的个数主要是为了减少std::partial_sort(L*K)的时间
    double t0 = elapsed();
    SearchIvf(n, x, L, nprobe, cellids, residuals, recall_cell_num_for_each_query, true,start_id,neighborsCount);  
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
        result[qid].resize(neighborsCount, std::make_pair(std::numeric_limits<float>::max(), -1));
        float* query_residual = residuals.data() + qid * nprobe * D;
        int* candi_cell_id = cellids.data() + qid * nprobe;
        int found = 0;
        int check_cell_num = 0;
        int empty_cell_num = 0;
        //遍历多个倒排链
        for(int i = 0; i < recall_cell_num_for_each_query[qid]; i++) {
          //得到倒排链数据
          int cellId = candi_cell_id[i];
          int cellStart = (cellId == 0) ? 0 : cellEdges[cellId - 1];
          int cellFinish = cellEdges[cellId];

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
            if(grouth_cellid[start_id + qid] == cellId) {
              search_stats.gt_in_retrievaled++; 
              LOG(INFO) << "GT CELL query:" << start_id + qid << ", hit retrieval cell:" << cellId;
            }
          }
          #endif
          float* cell_residual = query_residual + i * D;
          if(is_lopq) {
            //LOG(INFO) << "[LPQ] lopq cellId:" << cellId <<",coarseId:" << GetCoarseIdFromCellId(cellId);
            //每个残差单独o
            #ifdef GNOIMI_QUERY_DEBUG
            double tt1 = elapsed();
            #endif
            if(FLAGS_pq_minus_mean) {
                //需要减去均值  
               minus_mu(D, 1, cell_residual, lopq_residual_mean + GetCoarseIdFromCellId(cellId) * D);
            }
            LopqMatrix(cell_residual, 1, GetCoarseIdFromCellId(cellId));
            #ifdef GNOIMI_QUERY_DEBUG
            search_stats.opq_cost += (elapsed() - tt1); 
            #endif
          } else {
            if(FLAGS_pq_minus_mean) {
                //需要减去均值  
               minus_mu(D, 1, cell_residual, gopq_residual_mean);
            }
          }

          if(is_lpq) {
            //LOG(INFO) << "[LPQ] lpq";
            //每个倒排链单独pq
            cellVocab = lpqRerankVocabs + GetCoarseIdFromCellId(cellId) * D * rerankK;
          }
          std::unique_ptr<faiss::DistanceComputer> dis_computer;
          //TODO:@zhangcunyi 长倒排链用查标累加法，短倒排链直接计算这个临界值我觉得大概是rerankK*2 = 512
          if(cellFinish - cellStart >= compute_table_docnum_threshold && neighborsCount - found >= compute_table_docnum_threshold) {
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
              }
          } else {
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
        if(topk + 1000 >= result[qid].size()) {
          std::sort(result[qid].begin(), result[qid].end());
        } else {
          std::partial_sort(result[qid].begin(), result[qid].begin() + topk , result[qid].end());
        }
        //LOG(INFO) << "query " << qid << " check_cell_num "<< check_cell_num << " empty_cell_num "
        //  << empty_cell_num << " found " << found ;
        travel_doc += found;
        search_stats.nprobe += check_cell_num; 
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
        int cellId = coarseId * searcher.K + k;
        float alphaFactor = searcher.alpha[cellId];
        // 见公式
        scores[l*K+k].first = coarseList[l].first + searcher.fineNorms[k] * alphaFactor * alphaFactor
                              - queryFineDistance[k] * alphaFactor + searcher.coarseFineProducts[cellId] * alphaFactor;
        scores[l*K+k].second = cellId;
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
      int cellId = scores[currentPointer].second;
      int topListId = coarseIdToTopId[cellId / K];
      ++currentPointer;
      //得到倒排链数据
      int cellStart = (cellId == 0) ? 0 : searcher.cellEdges[cellId - 1];
      int cellFinish = searcher.cellEdges[cellId];
      if(cellStart == cellFinish) {
        continue;
      }
      memcpy(residual, coarseResiduals.data() + topListId * D, D * sizeof(float));
      cblas_saxpy(D, -1.0 * searcher.alpha[cellId], searcher.fineVocab + (cellId % K) * D, 1, residual, 1);
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
      LOG(INFO) << " BASE file " << FLAGS_base_file <<", get docnum:" << searcher.docnum <<", d:" << file_d
        <<", resize pq codes:" << searcher.docnum * M;
      gnoimi::b2fvecs_read_callback(FLAGS_base_file.c_str(),
         searcher.D, searcher.docnum, 1000000,std::bind(&Searcher::AddIndex, &searcher, FLAGS_L, std::placeholders::_1, 
         std::placeholders::_2, std::placeholders::_3),FLAGS_normalize_before_read);
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
      LOG(INFO) << "query finish " << i << ", result:" << result[0][0].second;
    }

    double t1 = elapsed();
    std::cout.setf(ios::fixed);//precision控制小数点后的位数
    std::cout.precision(2);//后2位
    std::cout << "[COST] ["<< (t1-t0) << " s] query num:" << queriesCount << ",thread_num:" << FLAGS_threadsCount << ",query cost:" << (t1-t0)*1000/queriesCount <<" ms" <<",nprobe:" << nprobe
      << ",L:"<< FLAGS_L <<",top:"<< FLAGS_neighborsCount<< ",travel_doc:"<< search_stats.travel_doc*1.0/queriesCount 
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
