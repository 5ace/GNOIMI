#include <algorithm>
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
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/distances.h>
#include <faiss/VectorTransform.h>
#include <gflags/gflags.h>
#include <faiss/index_io.h>

using std::cout;
using std::ios;
using std::string;
using std::vector;

DEFINE_string(groud_truth_file,"","format ivec");
DEFINE_string(query_filename,"","format fvec");
DEFINE_bool(make_index,true,"制作index还是查询index, true:only make index, false: only search index");
DEFINE_string(model_prefix,"./train_gnoimi_","prefix of model files");
DEFINE_string(index_prefix,"./index_gnoimi_","prefix of index files");
DEFINE_string(base_file,"","file content vectors to index,format:bvecs or fvecs");
DEFINE_int32(K,4096,"coarse and residual centriods num");
DEFINE_uint64(D,128,"feature dim");
DEFINE_int32(M,16,"PQ's subvector num");
DEFINE_uint64(N,1000000000,"index vectors num");
DEFINE_int64(L,32,"coarse search num for GNOIMI");
DEFINE_int32(rerankK,256,"PQ number of bit per subvector index,");
DEFINE_int32(threadsCount,10,"thread num");
DEFINE_int32(nprobe,200," if not in (0,K*L] else  set K*L");
DEFINE_int32(neighborsCount,5000,"coarse search num for GNOIMI");
DEFINE_int32(queriesCount,0,"load query count");
const int M = 16;
struct Record {
  int pointId;
  unsigned char bytes[M];
};

struct Searcher {
  float* coarseVocab;
  float* coarseNorms;
  float* fineVocab;
  float* fineNorms;
  float* alpha;
  float* coarseFineProducts;
  Record* index;
  int* cellEdges; //cellEdges[0] 表示第一个倒排链的END，第二个倒排链的开头
  float* rerankRotation;
  float* rerankVocabs;
  string model_prefix_,index_prefix_;
  size_t D;
  int K;
  int M;
  int subDim;
  int rerankK;
  int threadsCount;
  faiss::IndexPQ* pq;
  faiss::IndexPQ* pq_with_data;
  vector<vector<int>> ivf; //cellis and it's docidlist

  string coarseCodebookFilename;
  string fineCodebookFilename;
  string alphaFilename, rerankRotationFilename, rerankVocabsFilename, cellEdgesFilename,rawIndexFilename,rerankPQFaissFilename,rerankPQFaissWithDataFilename;

  Searcher(string model_prefix,string index_prefix,int D,int K, int M, int rerankK,int threadsCount) {
    pq = nullptr;
    pq_with_data = nullptr;
    index = nullptr;
    model_prefix_ = model_prefix;
    index_prefix_ = index_prefix;
    this->D = D;
    this->K = K;
    this->M = M;
    this->rerankK = rerankK;
    this->threadsCount = threadsCount;
    subDim = this->D / this->M;
    alphaFilename = model_prefix_ +"alpha.fvecs";
    fineCodebookFilename = model_prefix_ + "fine.fvecs";
    coarseCodebookFilename = model_prefix_ + "coarse.fvecs";
    rerankRotationFilename = model_prefix_ + "opq_matrix.fvecs";
    rerankVocabsFilename = model_prefix_ + "pq.fvecs";
    rerankPQFaissFilename = model_prefix_ + "pq.faiss.index";
    rerankPQFaissWithDataFilename = index_prefix_ + "pq.faiss.withdata.index";
    cellEdgesFilename = index_prefix_+ "cellEdges.dat";
    rawIndexFilename = index_prefix_+ "rawIndex.dat"; //作者设计的格式
    // 加载模型数据
    CHECK(ReadAndPrecomputeVocabsData()); 
    ivf.resize(K*K);
    LOG(INFO) << "Searcher construct ok";
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
      LOG_IF(INFO,cellEdges[i] - last_cell != 0) << "LoadCellEdges cell:" << i << ",docnum:" <<cellEdges[i] - last_cell;
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
  // preare for add index => save index
  bool LoadPQ() {
    CHECK(pq == nullptr);
    LOG(INFO) << "start load Faiss PQ " << rerankPQFaissFilename;
    pq = dynamic_cast<faiss::IndexPQ*>(faiss::read_index(rerankPQFaissFilename.c_str()));
    return pq != nullptr;
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
    size_t fz = gnoimi::file_size(rawIndexFilename.c_str());
    CHECK(fz > 0 && fz % sizeof(Record) == 0) << " check file " << rawIndexFilename;
    size_t docnum = fz / sizeof(Record);
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

    coarseVocab = (float*) malloc(K * D * sizeof(float));
    fvecs_read(coarseCodebookFilename.c_str(), D, K, coarseVocab);
    fineVocab = (float*) malloc(K * D * sizeof(float));
    fvecs_read(fineCodebookFilename.c_str(), D, K, fineVocab);
    alpha = (float*) malloc(K * K * sizeof(float));
    fvecs_read(alphaFilename.c_str(), K, K, alpha);
    rerankRotation = (float*) malloc(D * D * sizeof(float));
    fvecs_read(rerankRotationFilename.c_str(), D, D, rerankRotation);
    //float* temp = (float*) malloc(K * D * sizeof(float));
    //// 旋转1级码本,searcher.coarseVocab  
    //fmat_mul_full(rerankRotation, coarseVocab,
    //              D, K, D, "TN", temp);
    //memcpy(coarseVocab, temp, K * D * sizeof(float));
    //free(temp);
    //temp = (float*) malloc(K * D * sizeof(float));
    //// 旋转2级码本
    //fmat_mul_full(rerankRotation, fineVocab,
    //              D, K, D, "TN", temp);
    //memcpy(fineVocab, temp, K * D * sizeof(float));
    //free(temp);
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
    //原作者每个一级码本类中心训练了一个PQ
    //rerankVocabs = (float*)malloc(rerankK * D * K * sizeof(float));
    //fvecs_read(rerankVocabsFilename.c_str(), D / M, K * M * rerankK, rerankVocabs);
    //这里暂时所有倒排链公用一个PQ
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
    CHECK(pq->codes.size() == FLAGS_N * FLAGS_M) << "pq code size:"
      << pq->codes.size() << ", want:" << FLAGS_N * FLAGS_M;
    LOG(INFO) << "start saving index to " << cellEdgesFilename << "," << rawIndexFilename; 
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
        memcpy((char*)&(r.bytes[0]),(char*)(pq->codes.data()+docid*M),M);
        if(docid < 100) {
          LOG(INFO) << "=====codeinfo save index docid:" << docid;
          gnoimi::print_elements((char*)(pq->codes.data()+docid*M),M);
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
    LOG(INFO) << "finsih save index total write " << cellEdge << " want " << FLAGS_N;
    return true;
  };
  // 不能多线程调用
  void AddIndex(int L, size_t n, float *x, size_t start_id) {
    static vector<vector<int>> cellids;
    static vector<vector<float>> residuals;
    int nprobe = 1;
    if(start_id == 0) {
        LOG(INFO) << "======ori id 0";
        gnoimi::print_elements(x,D);
    }
    int loopN = 2000;
    int loopNum = (n + loopN - 1)/loopN;
    cellids.resize(loopNum);
    residuals.resize(loopNum);

    #pragma omp parallel for num_threads(FLAGS_threadsCount)
    for(size_t i = 0; i < loopNum; ++i) {
      int tmp_n = (i==loopNum-1) ? n - (loopNum-1) * loopN : loopN;
      //LOG(INFO) << "loopN " << i * loopN * D << ",loopNum:"<<loopNum<<",n:"<< n <<",start_id:"<<start_id <<",tmp_n:"<<tmp_n;
      SearchIvf(tmp_n, x + i * loopN * D, L, nprobe, cellids[i], residuals[i]);  
    }
   
    auto t1 = std::thread( [this,start_id]() {
        size_t id = start_id;
        for(auto & loop_cell : cellids) {
          for(auto cellid : loop_cell) {
            ivf[cellid].push_back(id++);
          }
        }
    });
    auto t2 = std::thread( [this]() {
        for(auto & loop_residual : residuals) {
          pq->add(loop_residual.size()/D,loop_residual.data());
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
  bool SearchIvf(size_t queriesCount, float *queries, size_t L, int &nprobe, vector<int> &cellids,vector<float> &residuals) {
    if(nprobe > L * K || nprobe <= 0) {
      nprobe = L * K;
    }
    residuals.resize(queriesCount * nprobe * D);
    cellids.resize(queriesCount * nprobe);
    //LOG(INFO) << nprobe << ",residuals size:" << residuals.size()*4;
    int subDim = D / M;
    std::clock_t c_start = std::clock();
    //先OPQ
    {
      thread_local vector<float> temp(queriesCount * D);
      temp.resize(queriesCount * D);

      fmat_mul_full(rerankRotation, queries,
                    D, queriesCount, D, "TN", temp.data());
      memcpy(queries, temp.data(), queriesCount * D * sizeof(float));
    }

    for(int qid = 0; qid < queriesCount; ++qid) {
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
      std::partial_sort(coarseList.begin(),coarseList.begin() + L ,coarseList.end());

      for(int l = 0; l < L; ++l) {
        int coarseId = coarseList[l].second;
        coarseIdToTopId[coarseId] = l;
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
      // 倒排链得分排序,取倒排链
      std::partial_sort(scores.begin(), scores.begin() + nprobe, scores.end());
      int travesered_list_num = 0;
      for(int travesered_list_num = 0; travesered_list_num < nprobe; ++travesered_list_num) {
        //得到一级和二级码本
        int cellId = scores[travesered_list_num].second;
        int topListId = coarseIdToTopId[cellId / K]; //得到这个cellid对应的粗排的顺序
        //拷贝p-S
        memcpy(query_residual + travesered_list_num * D , coarseResiduals.data() + topListId * D, D * sizeof(float));
        //residual = p -S - alpha*T
        cblas_saxpy(D, -1.0 * alpha[cellId], fineVocab + (cellId % K) * D, 1, query_residual + travesered_list_num * D, 1);
        candi_cell_id[travesered_list_num] = cellId;
      }
    }
    return true;
  }
  void SearchNearestNeighbors(float* x,
                              int n, int L, int neighborsCount, 
                              vector<vector<std::pair<float, int> > >& result, size_t start_id) {
    result.resize(n);
    //result查询结果每个query保存neighborsCount=5000个结果
    thread_local vector<int> cellids;
    thread_local vector<float> residuals;
    LOG(INFO) << "query vec Squared L2 norm " << faiss::fvec_norm_L2sqr(x,D);
    int nprobe = 0;
    SearchIvf(n, x, L, nprobe, cellids, residuals);  
    for(int i = 0; i < n; i++) {
      LOG(INFO) << "QueryIndex doc:" << start_id + i << ",search to cell:" << cellids[i*nprobe] << ",nprobe:"<<nprobe;
      if((start_id + i) % 10000 == 0) {
        LOG(INFO) << "======query id:" << start_id + i << ",residual===";
        gnoimi::print_elements(residuals.data()+i*nprobe*D,D);
      }
    }
    std::clock_t c_start = std::clock();
    float* cellVocab = rerankVocabs;
    for(int qid = 0; qid < n; ++qid) {
        result[qid].resize(neighborsCount, std::make_pair(std::numeric_limits<float>::max(), -1));
        float* query_residual = residuals.data() + qid * nprobe * D;
        int* candi_cell_id = cellids.data() + qid * nprobe;
        int found = 0;
        int check_cell_num = 0;
        int empty_cell_num = 0;
        //遍历多个倒排链
        for(int i = 0; i < nprobe; i++) {
          //得到倒排链数据
          int cellId = candi_cell_id[i];
          int cellStart = (cellId == 0) ? 0 : cellEdges[cellId - 1];
          int cellFinish = cellEdges[cellId];
          check_cell_num++;
          if(cellStart == cellFinish) {
            empty_cell_num++;
            continue;
          }
          float* cell_residual = query_residual + i * D;
          std::unique_ptr<faiss::DistanceComputer> dis_computer;
          if(pq_with_data != nullptr) {
            dis_computer.reset(pq_with_data->get_distance_computer());
            dis_computer->set_query(cell_residual);
          }
          for(int id = cellStart; id < cellFinish && found < neighborsCount; ++id) {
            result[qid][found].second = index[id].pointId;
            result[qid][found].first = 0.0;
            float diff = 0.0;
            for(int m = 0; m < M; ++m) {
              float* codeword = cellVocab + m * rerankK * subDim + index[id].bytes[m] * subDim;
              float* residualSubvector = cell_residual + m * subDim;
              for(int d = 0; d < subDim; ++d) {
                // 实时计算查询向量和对应编码类中心的距离
                diff = residualSubvector[d] - codeword[d];
                result[qid][found].first += diff * diff;
              }
            }
            // 现计算距离
            if(index[id].pointId < 100) {
              LOG(INFO) << "=====codeinfo query docid:" << index[id].pointId;
              gnoimi::print_elements((char*)&(index[id].bytes[0]),FLAGS_M);
              if(dis_computer) {
                LOG(INFO) << " match docid:"<< index[id].pointId <<",local dis:" << result[qid][found].first << ",faiss diss:" << (*dis_computer)(index[id].pointId);
              }
            }
            ++found;
          }
          if(found >= neighborsCount) {
            break;
          }
        }
        std::sort(result[qid].begin(), result[qid].end());
        //LOG(INFO) << "query " << qid << " check_cell_num "<< check_cell_num << " empty_cell_num "
        //  << empty_cell_num << "found " << found;
        for(int z = 0; z < 10 && z < result[qid].size();z++) {
          LOG(INFO) << "query:" << start_id+qid << ",match "<<z <<"th:" <<result[qid][z].second << ",score:"
          <<result[qid][z].first;
        }
    }
    std::clock_t c_end = std::clock();
    LOG(INFO) << "CPU time used: "<< 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / n<< " ms\n";
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

float computeRecallAt(const vector<vector<std::pair<float, int> > >& result,
                      const int* groundtruth, int R) {
  int limit = (R < result[0].size()) ? R : result[0].size();
  int positive = 0;
  for(int i = 0; i < result.size(); ++i) {
    for(int j = 0; j < limit; ++j) {
      if(result[i][j].second == groundtruth[i]) {
        ++positive;
      }
    }
  }
  return (float(positive) / result.size());
}
void ComputeRecall(const vector<vector<std::pair<float, int> > >& result,
                   const int* groundtruth) {
    int R = 1;
    std::cout << std::fixed << std::setprecision(5) << "Recall@ " << R << ": " << 
                 computeRecallAt(result, groundtruth, R) << "\n";
    R = 10;
    std::cout << std::fixed << std::setprecision(5) << "Recall@ " << R << ": " << 
                 computeRecallAt(result, groundtruth, R) << "\n";
    R = 100;
    std::cout << std::fixed << std::setprecision(5) << "Recall@ " << R << ": " << 
                 computeRecallAt(result, groundtruth, R) << "\n";
}

int main(int argc, char** argv) {
    gnoimi::print_elements(argv,argc);

    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true; //日志全部输出到标准错误

    if(FLAGS_nprobe <=0 || FLAGS_nprobe > FLAGS_K * FLAGS_L) {
      FLAGS_nprobe = FLAGS_K * FLAGS_L;
    }
    CHECK(FLAGS_rerankK == 256) << "rerankK must = 256";
    CHECK(FLAGS_L >= 0 && FLAGS_L <= FLAGS_K );
    LOG(INFO) << "sizeof Record " << sizeof(Record);
    // 初始化
    Searcher searcher(FLAGS_model_prefix,FLAGS_index_prefix,FLAGS_D,FLAGS_K,FLAGS_M,FLAGS_rerankK,FLAGS_threadsCount);

    if( FLAGS_make_index) {
      LOG(INFO) << "making index";
      CHECK(!FLAGS_base_file.empty()) << "need set file base_file";
      CHECK(searcher.LoadPQ());
      LOG(INFO) << "start indexing";
      size_t read_num = FLAGS_N;
      if(read_num > 0) {
        //预申请空间
        for(auto & i:searcher.ivf){
          i.reserve((read_num+1000000)/(searcher.K*searcher.K));
        }
      }
      gnoimi::b2fvecs_read_callback(FLAGS_base_file.c_str(),
         searcher.D, read_num, 100000,std::bind(&Searcher::AddIndex, &searcher, FLAGS_L, std::placeholders::_1, 
         std::placeholders::_2, std::placeholders::_3));
      LOG(INFO) << "end index,read " << read_num << ",want " << FLAGS_N;
      searcher.SaveIndex();
    }
    CHECK(!access(FLAGS_groud_truth_file.c_str(),0)  && !access(FLAGS_query_filename.c_str(),0));
    searcher.LoadIndex();
    //searcher.LoadPQWithData();
    size_t queriesCount = FLAGS_queriesCount;
    // 读取query
    auto queries =  gnoimi::read_bfvecs(FLAGS_query_filename.c_str(), FLAGS_D, queriesCount,true);
    LOG(INFO) << "L2 norm " << faiss::fvec_norm_L2sqr(queries.get(), FLAGS_D);
    LOG(INFO) << "read "<< queriesCount <<" doc from " << FLAGS_query_filename << ", d:" << FLAGS_D;
    //float * z = new float[FLAGS_D * FLAGS_D];
    //fmat_mul_full(searcher.rerankRotation,searcher.rerankRotation,
    //              FLAGS_D, FLAGS_D, FLAGS_D, "TN", z);
    //gnoimi::print_elements(z,2*FLAGS_D);

    LOG(INFO) << "========ori id:0 ";
    gnoimi::print_elements(queries.get(),FLAGS_D);
    float* temp = (float*) malloc(queriesCount * FLAGS_D * sizeof(float));
    //// 旋转一下,应该是一级OPQ  ====> 改到搜索倒排链之前统一旋转
    //fmat_mul_full(searcher.rerankRotation, queries.get(),
    //              FLAGS_D, queriesCount, FLAGS_D, "TN", temp);
    //memcpy(queries.get(), temp, queriesCount * FLAGS_D * sizeof(float));
    //free(temp);  
    vector<vector<std::pair<float, int> > > results(queriesCount);
    #pragma omp parallel for num_threads(FLAGS_threadsCount)
    for(size_t i = 0 ; i<queriesCount; i++) {
      vector<vector<std::pair<float, int> > > result;
      searcher.SearchNearestNeighbors(queries.get() + i * FLAGS_D, 1, FLAGS_L,FLAGS_neighborsCount, result, i);
      results[i]  = result[0];
      LOG(INFO) << "query finish " << i << ", result:" << result[0][0].second;
    }
    //SearchNearestNeighbors(searcher,queries, neighborsCount, result);
    LOG(INFO) << "Before reading groundtruth...";
    size_t gt_d = 1;
    size_t gt_num = queriesCount;
    auto groundtruth = gnoimi::ivecs_read(FLAGS_groud_truth_file.c_str(),&gt_d,&gt_num);
    for(int i =0;i<gt_num; i++) {
      LOG(INFO) << "=======gt"<< i <<"========";
      gnoimi::print_elements(groundtruth.get()+i,gt_d);
    }
    LOG(INFO) << "Groundtruth is read";
    ComputeRecall(results, groundtruth.get()); 
    return 0;
}
