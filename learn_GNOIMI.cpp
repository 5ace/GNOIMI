#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <functional>
#include <vector>
#include <ctime>
#include <chrono>
#include <string.h>

#include <cblas.h>
#include "tool/utils.h"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>
#include <faiss/VectorTransform.h>
#include <gflags/gflags.h>
#include <faiss/index_io.h>


using std::cout;
using std::ios;
using std::string;
using std::vector;

DEFINE_int32(thread_num,10,"train S,T, alpha thread num");
DEFINE_int32(k,4096,"coarse and residual centriods num");
DEFINE_int32(d,128,"feature dim");
DEFINE_int32(n,1000000,"learn feature num, AUTO CHANGE TO  multiple of learnIterationsCount*thread_num");
DEFINE_int32(learnIterationsCount,10,"learnIterationsCount");
DEFINE_int32(l,8,"");
DEFINE_int32(M,16,"PQ's subvector num");
DEFINE_int32(KK_nbits,8,"PQ number of bit per subvector index");
DEFINE_string(initCoarseFilename,"","initCoarseFilename fvecs, centorids of coarse");
DEFINE_string(initFineFilename,"","initFineFilename fvecs, centorids of residual");
DEFINE_string(learnFilename,"","learnFilename fvecs or bvecs, must normlized");
DEFINE_string(outputFilesPrefix,"./train_gnoimi_","prefix of output file name");
DEFINE_string(pqOutputFilesPrefix,"","if empty then same as FLAGS_outputFilesPrefix");
DEFINE_int32(trainThreadChunkSize,10000,"the dim num for yael fmat_mul_full ");
DEFINE_bool(direct_train_s_t_alpha,false,"false = alreay train coarse fine and alpha, and read them from file");
DEFINE_bool(train_pq,true,"is train pq");
DEFINE_bool(train_opq,true,"分2种情况1.跟faiss一样用原始向量训练opq；2.用残差训练opq、用残差*opq训练pq；这里设置true就是方法2用残差训练opq");
DEFINE_bool(train_lopq,false,"每个一级倒排链单独训练一个pq和opq旋转矩阵,faiss每个kmeans最少需要40*k=10240,4000个倒排链最好是5000w训练数据，每个倒排链分配1万多个训练数据");
DEFINE_bool(train_lopq_global_o,false,"如果为true则所有倒排链公用一个o,每个一级倒排链只单独一个pq");
DEFINE_int32(lopq_train_coarse_start,-1,"可选值0,..K-1,-1 means all");
DEFINE_int32(lopq_train_coarse_end,-1,"可选值1,..K,-1 means all 主要为了分布式计算,[lopq_train_coarse_start,lopq_train_coarse_end)");
DEFINE_string(lpq_file_prefix,"","lpq输出的多个opq pq faisspq的文件名前缀，最好单独放到一个目录下");
DEFINE_bool(pq_minus_mean,false,"是否在pq或opq之前减去均值");
DEFINE_bool(normalize_before_train,true,"is normalize read train fvecs");

size_t D;
int K;
size_t totalLearnCount;
int learnIterationsCount;
int L;

string learnFilename; 
string initCoarseFilename;
string initFineFilename;
string outputFilesPrefix;
string pqOutputFilesPrefix;

int trainThreadChunkSize;
int threadsCount;

float* train_vecs;
float* residual_vecs;
int* coarseAssigns;
int* fineAssigns;
int64_t* assigns;
float* alphaNum;
float* alphaDen;
float* alpha;

vector<float*> alphaNumerators;
vector<float*> alphaDenominators;

float* coarseVocab;
float* fineVocab;
float* fineVocabNum;
float* fineVocabDen;
float* coarseVocabNum;
float* coarseVocabDen;

vector<uint32_t> coarse_doc_num;

vector<float*> fineVocabNumerators;
vector<float*> fineVocabDenominators;
vector<float*> coarseVocabNumerators;
vector<float*> coarseVocabDenominators;

float* coarseNorms;
float* fineNorms;
float* coarseFineProducts;

float* errors;
string alphaFilename;
string fineVocabFilename;
string coarseVocabFilename;
string pqFileName;
string pqIndexFileName;
string opq_matrix_file;
string gopq_residual_mean_file;
string lpq_file_prefix;

///////////////////////////
float* calc_mean(int d, int n, float *x) {
   float* mu = new float[d];
   memset(mu,0,sizeof(float)*d);
   for(size_t i = 0; i < n; i++) {
     for(size_t j = 0; j < d ; j++) {
        mu[j] += x[i * d + j]; 
     } 
   } 
   for(size_t j = 0; j < d ; j++) {
      mu[j] = mu[j] / n;
   }
   return mu;
}


// calc || p -( Sk + alpha * Tl) || ^2
float disOfCell(int k, int l, float* x) {
  vector<float> t(D);
  float *c = coarseVocab + k * D;
  float *f = fineVocab + l * D;
  for(int i = 0; i < D; i++) {
    t[i] = c[i] + alpha[k * K + l] * f[i];
  }
  return faiss::fvec_L2sqr(x,t.data(),D); 
}
float disOfCell(int k,float* x) {
  float *c = coarseVocab + k * D;
  return faiss::fvec_L2sqr(x,c,D); 
}
void computeOptimalAssignsSubset(int threadId) {
  long long startId = (totalLearnCount / threadsCount) * threadId;
  int pointsCount = totalLearnCount / threadsCount;
  int chunksCount = pointsCount / trainThreadChunkSize;
  float* pointsCoarseTerms = (float*)malloc(trainThreadChunkSize * K * sizeof(float));
  float* pointsFineTerms = (float*)malloc(trainThreadChunkSize * K * sizeof(float));
  errors[threadId] = 0.0;

  float* chunkPoints = train_vecs + startId * D;
  std::vector<std::pair<float, int> > coarseScores(K);
  for(int chunkId = 0; chunkId < chunksCount; ++chunkId,chunkPoints += trainThreadChunkSize * D) {
    LOG(INFO) << "[Assigns][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << chunksCount;
    LOG(INFO) << "train vec Squared L2 norm " << faiss::fvec_norm_L2sqr(chunkPoints,D);
    // 与查询的过程一致，|p-S-alpha*T| 
    // 索引这里预计算p*S和p*T
    fmat_mul_full(coarseVocab, chunkPoints, K, trainThreadChunkSize, D, "TN", pointsCoarseTerms);
    fmat_mul_full(fineVocab, chunkPoints, K, trainThreadChunkSize, D, "TN", pointsFineTerms);
    for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
      // 这里想计算|p-S|^2
      // 转换为 p*p/2-P*S +S*S/2,由于对于同一个p,S*S/2=0.5是一样的所以排序时排除即可
      
      // 这里计算的是P*S-p*p/2=tmp真正的l2距离是 -1*tmp*2+1
      cblas_saxpy(K, -1.0, coarseNorms, 1, pointsCoarseTerms + pointId * K, 1);
      for(int k = 0; k < K; ++k) {
        // 这first就是每个查询向量与所有一级类中心的近似距离,S*S/2-p*S
        coarseScores[k].first = (-1.0) * pointsCoarseTerms[pointId * K + k];
        coarseScores[k].second = k;
      }
      std::sort(coarseScores.begin(), coarseScores.end());
#if 0
      {
        for(int i = 0; i < K; i++) {
          LOG(INFO) << "after sort:" << i << ",k:"<<coarseScores[i].second<<",calc dis:"
          <<coarseScores[i].first<<","<< coarseScores[i].first * 2 + 1
          <<",real dis:"<< disOfCell(coarseScores[i].second, chunkPoints + chunkId * D);
        }
        exit(0);
      }
#endif

      float currentMinScore = 999999999.0;
      int currentMinCoarseId = -1;
      int currentMinFineId = -1;
      for(int l = 0; l < L; ++l) {
        //examine cluster l
        int currentCoarseId = coarseScores[l].second;
        float currentCoarseTerm = coarseScores[l].first;
        for(int currentFineId = 0; currentFineId < K; ++currentFineId) {
          float alphaFactor = alpha[currentCoarseId * K + currentFineId];
          // score = |p - T - alpha*T|
          float score = currentCoarseTerm + alphaFactor * coarseFineProducts[currentCoarseId * K + currentFineId] + 
                        (-1.0) * alphaFactor * pointsFineTerms[pointId * K + currentFineId] + 
                        alphaFactor * alphaFactor * fineNorms[currentFineId];
          //LOG(INFO) << "l:"<< l << ",k:"<< currentFineId << ",l2 dis:"<< score*2+1.0 <<",real dis:" << disOfCell(currentCoarseId, currentFineId, chunkPoints + chunkId * D);
          if(score < currentMinScore) {
            currentMinScore = score;
            currentMinCoarseId = currentCoarseId;
            currentMinFineId = currentFineId;
          }
        }
      }
      //保留每个训练向量的一级码本，二级码本以及与此码本的欧氏距离
      coarseAssigns[startId + chunkId * trainThreadChunkSize + pointId] = currentMinCoarseId;
      fineAssigns[startId + chunkId * trainThreadChunkSize + pointId] = currentMinFineId;
      assigns[startId + chunkId * trainThreadChunkSize + pointId] = currentMinCoarseId * K + currentMinFineId;
      errors[threadId] += currentMinScore * 2 + 1.0; // point has a norm equals 1.0
    }
  }
  free(pointsCoarseTerms);
  free(pointsFineTerms);
}

void computeOptimalAlphaSubset(int threadId) {
  memset(alphaNumerators[threadId], 0, K * K * sizeof(float));
  memset(alphaDenominators[threadId], 0, K * K * sizeof(float));
  long long startId = (totalLearnCount / threadsCount) * threadId;
  int pointsCount = totalLearnCount / threadsCount;
  int chunksCount = pointsCount / trainThreadChunkSize;
  float* residual = (float*)malloc(D * sizeof(float));
  float* chunkPoints = train_vecs + startId * D;
  for(int chunkId = 0; chunkId < chunksCount; ++chunkId,chunkPoints += trainThreadChunkSize * D) {
    std::cout << "[Alpha][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << chunksCount << "\n";
    for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
      int coarseAssign = coarseAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      int fineAssign = fineAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      memcpy(residual, chunkPoints + pointId * D, D * sizeof(float));
      cblas_saxpy(D, -1.0, coarseVocab + coarseAssign * D, 1, residual, 1);
      alphaNumerators[threadId][coarseAssign * K + fineAssign] += 
           cblas_sdot(D, residual, 1, fineVocab + fineAssign * D, 1);
      alphaDenominators[threadId][coarseAssign * K + fineAssign] += fineNorms[fineAssign] * 2; // we keep halves of norms 
    }
  }
  free(residual);
}

void computeOptimalFineVocabSubset(int threadId) {
  memset(fineVocabNumerators[threadId], 0, K * D * sizeof(float));
  memset(fineVocabDenominators[threadId], 0, K * sizeof(float));
  long long startId = (totalLearnCount / threadsCount) * threadId;
  int pointsCount = totalLearnCount / threadsCount;
  int chunksCount = pointsCount / trainThreadChunkSize;
  float* residual = (float*)malloc(D * sizeof(float));
  float* chunkPoints = train_vecs + startId * D;
  for(int chunkId = 0; chunkId < chunksCount; ++chunkId,chunkPoints += trainThreadChunkSize * D) {
    std::cout << "[Fine vocabs][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << chunksCount << "\n";
    for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
      int coarseAssign = coarseAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      int fineAssign = fineAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      float alphaFactor = alpha[coarseAssign * K + fineAssign];
      memcpy(residual, chunkPoints + pointId * D, D * sizeof(float));
      cblas_saxpy(D, -1.0, coarseVocab + coarseAssign * D, 1, residual, 1);
      cblas_saxpy(D, alphaFactor, residual, 1, fineVocabNumerators[threadId] + fineAssign * D, 1);
      fineVocabDenominators[threadId][fineAssign] += alphaFactor * alphaFactor;
    }
  }
  free(residual);
}

void computeOptimalCoarseVocabSubset(int threadId) {
  memset(coarseVocabNumerators[threadId], 0, K * D * sizeof(float));
  memset(coarseVocabDenominators[threadId], 0, K * sizeof(float));
  long long startId = (totalLearnCount / threadsCount) * threadId;
  int pointsCount = totalLearnCount / threadsCount;
  int chunksCount = pointsCount / trainThreadChunkSize;
  float* residual = (float*)malloc(D * sizeof(float));
  float* chunkPoints = train_vecs + startId * D;
  for(int chunkId = 0; chunkId < chunksCount; ++chunkId,chunkPoints += trainThreadChunkSize * D) {
    std::cout << "[Coarse vocabs][Thread " << threadId << "] " << "processing chunk " <<  chunkId << " of " << chunksCount;
    for(int pointId = 0; pointId < trainThreadChunkSize; ++pointId) {
      int coarseAssign = coarseAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      int fineAssign = fineAssigns[startId + chunkId * trainThreadChunkSize + pointId];
      float alphaFactor = alpha[coarseAssign * K + fineAssign];
      memcpy(residual, chunkPoints + pointId * D, D * sizeof(float));
      cblas_saxpy(D, -1.0 * alphaFactor, fineVocab + fineAssign * D, 1, residual, 1);
      cblas_saxpy(D, 1, residual, 1, coarseVocabNumerators[threadId] + coarseAssign * D, 1);
      coarseVocabDenominators[threadId][coarseAssign] += 1.0;
    }
  }
  free(residual);
}
void init_global_varibles() {
    K = FLAGS_k;
    learnIterationsCount = FLAGS_learnIterationsCount;
    L = FLAGS_l;
    initCoarseFilename = FLAGS_initCoarseFilename;
    initFineFilename = FLAGS_initFineFilename;
    outputFilesPrefix = FLAGS_outputFilesPrefix;
    pqOutputFilesPrefix = FLAGS_pqOutputFilesPrefix.empty() ? outputFilesPrefix : FLAGS_pqOutputFilesPrefix;
    trainThreadChunkSize = FLAGS_trainThreadChunkSize; //矩阵相乘的大小
    threadsCount = FLAGS_thread_num;

    //这里因为原作者的设计多线程和每次矩阵相乘大小的要求totalLearnCount需规整一下
    int totalLearnCount_t = totalLearnCount;
    totalLearnCount = totalLearnCount/threadsCount/trainThreadChunkSize*trainThreadChunkSize*threadsCount;
    LOG(INFO) << "totalLearnCount(FLAGS_n) change from " << totalLearnCount_t <<" to " << totalLearnCount;
    residual_vecs = (float*)malloc(totalLearnCount * D * sizeof(float));
    coarseAssigns = (int*)malloc(totalLearnCount * sizeof(int));
    fineAssigns = (int*)malloc(totalLearnCount * sizeof(int));
    assigns = (int64_t*)malloc(totalLearnCount * sizeof(int64_t));
    alphaNum = (float*)malloc(K * K * sizeof(float));
    alphaDen = (float*)malloc(K * K * sizeof(float));
    alpha = (float*)malloc(K * K * sizeof(float));
    
    alphaNumerators.resize(threadsCount);
    alphaDenominators.resize(threadsCount);
    
    coarseVocab = (float*)malloc(D * K * sizeof(float));
    fineVocab = (float*)malloc(D * K * sizeof(float));
    fineVocabNum = (float*)malloc(D * K * sizeof(float));
    fineVocabDen = (float*)malloc(K * sizeof(float));
    coarseVocabNum = (float*)malloc(D * K * sizeof(float));
    coarseVocabDen = (float*)malloc(K * sizeof(float));
    
    fineVocabNumerators.resize(threadsCount);
    fineVocabDenominators.resize(threadsCount);
    coarseVocabNumerators.resize(threadsCount);
    coarseVocabDenominators.resize(threadsCount);
    
    coarseNorms = (float*)malloc(K * sizeof(float));
    fineNorms = (float*)malloc(K * sizeof(float));
    coarseFineProducts = (float*)malloc(K * K * sizeof(float));
    
    errors = (float*)malloc(threadsCount * sizeof(float));
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      alphaNumerators[threadId] = (float*)malloc(K * K * sizeof(float*));
      alphaDenominators[threadId] = (float*)malloc(K * K * sizeof(float*));
    }
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      fineVocabNumerators[threadId] = (float*)malloc(K * D * sizeof(float*));
      fineVocabDenominators[threadId] = (float*)malloc(K * sizeof(float*));
    }  
    for(int threadId = 0; threadId < threadsCount; ++threadId) {
      coarseVocabNumerators[threadId] = (float*)malloc(K * D * sizeof(float));
      coarseVocabDenominators[threadId] = (float*)malloc(K * sizeof(float));
    }
    alphaFilename = outputFilesPrefix +"alpha.fvecs";
    fineVocabFilename = outputFilesPrefix + "fine.fvecs";
    coarseVocabFilename = outputFilesPrefix + "coarse.fvecs";
    pqFileName = pqOutputFilesPrefix + "pq.fvecs";
    pqIndexFileName = pqOutputFilesPrefix + "pq.faiss.index";
    opq_matrix_file = pqOutputFilesPrefix + "opq_matrix.fvecs";
    lpq_file_prefix = FLAGS_lpq_file_prefix;
    gopq_residual_mean_file = pqOutputFilesPrefix + "gopq_residual_mean.fvecs";
}
void update_precompute() {
      //计算每个类中心的内积,用于计算距离使用
      for(int k = 0; k < K; ++k) {
        coarseNorms[k] = cblas_sdot(D, coarseVocab + k * D, 1, coarseVocab + k * D, 1) / 2;
        fineNorms[k] = cblas_sdot(D, fineVocab + k * D, 1, fineVocab + k * D, 1) / 2;
      }
      //矩阵相乘 coarseFineProducts=fineVocab*transpose(coarseVocab) 得到K*K的矩阵
      fmat_mul_full(fineVocab, coarseVocab, K, K, D, "TN", coarseFineProducts);
}

int update_assigns() {

      vector<std::thread> workers;
      memset(errors, 0, threadsCount * sizeof(float));
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        workers.push_back(std::thread(computeOptimalAssignsSubset, threadId));
      }
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        workers[threadId].join();
      }
      float totalError = 0.0;
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        totalError += errors[threadId];
      }
      LOG(INFO) << "update_assigns finish, Current reconstruction error... " << totalError / totalLearnCount 
      << ",imbalance:"<<faiss::imbalance_factor(totalLearnCount,K*K,assigns);
      return 0;
}
void train_alpha() {
    // init vocabs
    fvecs_read(initCoarseFilename.c_str(), D, K, coarseVocab);
    fvecs_read(initFineFilename.c_str(), D, K, fineVocab);
    // init alpha
    for(int i = 0; i < K * K; ++i) {
      alpha[i] = 1.0;
    }
    // learn iterations
    std::cout << "Start learning iterations...\n";
    for(int it = 0; it < learnIterationsCount; ++it) {
      // 预计算
      update_precompute();

      // https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Babenko_Efficient_Indexing_of_CVPR_2016_paper.pdf
      // 见这里迭代 优化4个参数

      // update Assigns
      update_assigns();

      // update alpha
      vector<std::thread> workers;
      workers.clear();
      LOG(INFO) << "update alpha";
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        workers.push_back(std::thread(computeOptimalAlphaSubset, threadId));
      }
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        workers[threadId].join();
      }
      workers.clear();
      memset(alphaNum, 0, K * K * sizeof(float));
      memset(alphaDen, 0, K * K * sizeof(float));
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        cblas_saxpy(K * K, 1, alphaNumerators[threadId], 1, alphaNum, 1);
        cblas_saxpy(K * K, 1, alphaDenominators[threadId], 1, alphaDen, 1);
      }
      for(int i = 0; i < K * K; ++i) {
        alpha[i] = (alphaDen[i] == 0) ? 1.0 : alphaNum[i] / alphaDen[i];
      }
      LOG(INFO) << "update fine Vocabs";
      // update fine Vocabs
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        workers.push_back(std::thread(computeOptimalFineVocabSubset, threadId));
      }
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        workers[threadId].join();
      }
      workers.clear();
      memset(fineVocabNum, 0, K * D * sizeof(float));
      memset(fineVocabDen, 0, K * sizeof(float));
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        cblas_saxpy(K * D, 1, fineVocabNumerators[threadId], 1, fineVocabNum, 1);
        cblas_saxpy(K, 1, fineVocabDenominators[threadId], 1, fineVocabDen, 1);
      }
      for(int i = 0; i < K * D; ++i) {
        fineVocab[i] = (fineVocabDen[i / D] == 0) ? 0 : fineVocabNum[i] / fineVocabDen[i / D];
      }
      LOG(INFO) << "update coarse Vocabs";
      // update coarse Vocabs
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        workers.push_back(std::thread(computeOptimalCoarseVocabSubset, threadId));
      }
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        workers[threadId].join();
      }
      workers.clear();
      memset(coarseVocabNum, 0, K * D * sizeof(float));
      memset(coarseVocabDen, 0, K * sizeof(float));
      for(int threadId = 0; threadId < threadsCount; ++threadId) {
        cblas_saxpy(K * D, 1, coarseVocabNumerators[threadId], 1, coarseVocabNum, 1);
        cblas_saxpy(K, 1, coarseVocabDenominators[threadId], 1, coarseVocabDen, 1);
      }
      for(int i = 0; i < K * D; ++i) {
        coarseVocab[i] = (coarseVocabDen[i / D] == 0) ? 0 : coarseVocabNum[i] / coarseVocabDen[i / D];
      }
      // save current alpha and vocabs
      //string alphaFilename = outputFilesPrefix +"alpha_" + std::to_string(it) + ".fvecs";
      //fvecs_write(alphaFilename.c_str(),K,K,alpha);
      //string fineVocabFilename = outputFilesPrefix + "fine_" + std::to_string(it) + ".fvecs";
      //fvecs_write(fineVocabFilename.c_str(),D,K,fineVocab);
      //string coarseVocabFilename = outputFilesPrefix + "coarse_" + std::to_string(it) + ".fvecs";
      //fvecs_write(coarseVocabFilename.c_str(),D,K,coarseVocab);
    }
    fvecs_write(alphaFilename.c_str(),K,K,alpha);
    fvecs_write(fineVocabFilename.c_str(),D,K,fineVocab);
    fvecs_write(coarseVocabFilename.c_str(),D,K,coarseVocab);
    LOG(INFO) << "finish train alpha and write them to file";
}
// 在已知alpha S T 的情况下计算残差
void update_residuls() {
    LOG(INFO) << "[Residual] precomput";
    update_precompute();
    update_assigns();

    coarse_doc_num.resize(K,0);
    memset(coarse_doc_num.data(),0,sizeof(coarse_doc_num[0])*coarse_doc_num.size());

    LOG(INFO) << "[Residual] start miuns";
    #pragma omp parallel for num_threads(threadsCount)
    for(int i = 0 ; i < totalLearnCount;i++) {
      int coarseAssign = coarseAssigns[i];
      #pragma omp critical
      {
        coarse_doc_num[coarseAssign]++;
      }
      int fineAssign = fineAssigns[i];
      float* coarse = coarseVocab + coarseAssign * D;
      float* fine = fineVocab + fineAssign * D;
      float* residual = residual_vecs + i * D;
      float* train = train_vecs + i * D;
      float alpha_i = alpha[coarseAssign*K + fineAssign]; 
      for(int j = 0; j < D; j++) {
        residual[j] =  train[j] - coarse[j] - fine[j] * alpha_i ;
      }
    }
    LOG(INFO) << "=====train ====";
    gnoimi::print_elements(train_vecs+3*D,D);
    LOG(INFO) << "=====coarse ====";
    gnoimi::print_elements(coarseVocab+coarseAssigns[3]*D,D);
    LOG(INFO) << "=====fine ====";
    gnoimi::print_elements(fineVocab+fineAssigns[3]*D,D);
    LOG(INFO) << "=====alpha ====" << alpha[coarseAssigns[3]*K + fineAssigns[3]];
    LOG(INFO) << "=====residual ====";
    gnoimi::print_elements(residual_vecs+3*D,D);
    int sum = 0;
    for(int i = 0; i < K; i++){
      LOG(INFO) << "coarse " << i <<", has " << coarse_doc_num[i] << " elements";
      sum += coarse_doc_num[i];
    }
    LOG(INFO) << "[Residual] finish miuns " << sum ;
}
//训练opq，保存到save_filename，并且改变训练向量x=R*x
void train_opq(uint64_t n, float* x, string save_filename, string mean_file) {
      //计算残差均值,并输出到文件
      std::unique_ptr<float[]> mu(calc_mean(D,n,x));
      fvecs_write(mean_file.c_str(),D,1,mu.get());

      faiss::OPQMatrix opq(D,FLAGS_M);
      opq.max_train_points = opq.max_train_points * 5;
      opq.niter = 70;
      opq.niter_pq = 6;

      opq.verbose = true;
      opq.train(n,x);
      fvecs_write(save_filename.c_str(),D,D,opq.A.data());

      if(FLAGS_pq_minus_mean) {
        LOG(INFO) << "start minus mu";
        minus_mu(D,n,x,mu.get());
        LOG(INFO) << "end minus mu";
      }
      // 旋转残差
      size_t batch_size = 1000000;
      size_t batch_num = (n + batch_size - 1) / batch_size;
      for(size_t i = 0; i < batch_num; i++) {
         size_t real_size = ((i == batch_num - 1) ? (n - batch_size * (batch_num - 1)) : batch_size);
         float* r = new float[real_size * D];
         fmat_mul_full(opq.A.data(), x + i * batch_size * D , D, real_size, D, "TN", r);
         memcpy(x + i * batch_size * D, r , sizeof(float) * real_size * D);
         delete[] r;
      }
}
void train_pq(uint64_t n, float* x, string save_matrix_filename, string save_faiss_filename) {
      //开始训练PQ
      faiss::IndexPQ pq(D,FLAGS_M,FLAGS_KK_nbits);
      pq.do_polysemous_training = true;
      pq.verbose = true;
      pq.pq.cp.max_points_per_centroid = 2000;
      pq.pq.cp.niter = 40;
      pq.pq.verbose = true;
      pq.train(n,x);
      fvecs_write(save_matrix_filename.c_str(),pq.pq.dsub,pq.pq.ksub*pq.pq.M,pq.pq.centroids.data());
      faiss::write_index(&pq,save_faiss_filename.c_str());

      float error_sum = 0.0f;
      #pragma omp parallel for reduction(+:error_sum)
      for(size_t i = 0; i < n; i++) {
        vector<uint8_t> code(pq.sa_code_size());
        vector<float> decode_v(D);
        pq.sa_encode(1,x + i * D, code.data()); 
        pq.sa_decode(1,code.data(),decode_v.data());
        error_sum += faiss::fvec_L2sqr(decode_v.data(),x + i * D,D);
      }
      LOG(INFO) << "pq error for n:"<< n <<",minus_mu:"<<FLAGS_pq_minus_mean<<",train_lopq:"<<FLAGS_train_lopq
      << ",train_lopq_global_o:"<<FLAGS_train_lopq_global_o <<",train_opq:" << FLAGS_train_opq
      << ",error:" << error_sum / n <<",total_error:" << error_sum;
}
int main(int argc, char** argv) {
    gnoimi::print_elements(argv,argc);

    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    // 初始化全局变量 =====
    FLAGS_logtostderr = true; //日志全部输出到标准错误

    learnFilename = FLAGS_learnFilename; 
    D = FLAGS_d;
    totalLearnCount = FLAGS_n;
    std::shared_ptr<float> features = gnoimi::read_bfvecs(learnFilename.c_str(), D,  totalLearnCount, FLAGS_normalize_before_train); 
    // 初始化全局变量
    init_global_varibles();
    // 读取训练向量
    train_vecs = features.get();

    LOG(INFO) << initCoarseFilename << " "<< initFineFilename << " " <<
        learnFilename << " " << outputFilesPrefix;

    CHECK(!learnFilename.empty() && !outputFilesPrefix.empty());
    if(FLAGS_direct_train_s_t_alpha == true) {
      CHECK(!initCoarseFilename.empty() && !initFineFilename.empty());
      //需要训练alpha
      train_alpha();
    } else {
      //读取alpha 
      LOG(INFO) << "loading " << alphaFilename <<"," << fineVocabFilename <<"," << coarseVocabFilename;
      CHECK(fvecs_read(alphaFilename.c_str(), K, K, alpha) == K);
      CHECK(fvecs_read(fineVocabFilename.c_str(), D, K, fineVocab) == K);
      CHECK(fvecs_read(coarseVocabFilename.c_str(), D, K, coarseVocab) == K);
    }
    
    // 计算残差
    update_residuls();

    if(FLAGS_train_lopq) {
      CHECK(FLAGS_train_opq == false && FLAGS_train_pq == false);
    }

    if(FLAGS_train_opq) {
      LOG(INFO) << "train opq start ==== ";
      train_opq(totalLearnCount, residual_vecs, opq_matrix_file,gopq_residual_mean_file);
      LOG(INFO) << "train opq end ==== ";
    }

    if(FLAGS_train_pq) {
      //开始训练PQ
      LOG(INFO) << "train PQ start";
      train_pq(totalLearnCount, residual_vecs, pqFileName, pqIndexFileName);
      LOG(INFO) << "train PQ finish";
    }

    if(FLAGS_train_lopq) {
      CHECK(!lpq_file_prefix.empty());
      LOG(INFO) << "train LOPQ start FLAGS_lopq_train_coarse_start:" << FLAGS_lopq_train_coarse_start
        <<",FLAGS_lopq_train_coarse_end:" << FLAGS_lopq_train_coarse_end;
      CHECK(FLAGS_lopq_train_coarse_start >=-1 && FLAGS_lopq_train_coarse_start <= FLAGS_k
        && FLAGS_lopq_train_coarse_end >=-1 && FLAGS_lopq_train_coarse_end <= FLAGS_k);
      std::vector<int> coarse_centers;
      if(FLAGS_lopq_train_coarse_start == -1 || FLAGS_lopq_train_coarse_end == -1) {
        for(int i = 0; i < FLAGS_k; i++)
          coarse_centers.push_back(i);
      } else {
          for(int i = FLAGS_lopq_train_coarse_start; i < FLAGS_lopq_train_coarse_end && i < FLAGS_k; ++i)
            coarse_centers.push_back(i);
      }
      if(coarse_centers.empty()) {
        LOG(INFO) << "no coarse_centers";
        return 0;
      }
      std::shared_ptr<float> global_o;
      if(FLAGS_train_lopq_global_o == true) {
        LOG(INFO) << "load global o from " << opq_matrix_file;
        uint64_t d = 0;
        uint64_t size = 0;

        global_o.reset(gnoimi::fvecs_read(opq_matrix_file.c_str(),&d,&size));
        LOG(INFO) << "load global o from " << opq_matrix_file << ", d:"<<d <<",size:"<<size;
      }
      // 相同一级聚类的residual拷贝到一起
      vector<float> train_residual; 
      for(auto coarse_center : coarse_centers) {
        train_residual.resize(coarse_doc_num[coarse_center] * D);
        int idx = 0;
        for(int i = 0 ; i < totalLearnCount;i++) {
          int coarseAssign = coarseAssigns[i];
          if(coarseAssign == coarse_center) {
            memcpy(train_residual.data() + idx * D, residual_vecs + i * D , sizeof(float) * D); 
            idx++;
          }
        }
        CHECK(idx == coarse_doc_num[coarse_center]) << "coarse_center:" <<coarse_center <<",idx:"<<idx
          <<",coarse_doc_num[coarse_center]:" << coarse_doc_num[coarse_center];
        LOG(INFO) << "start train lopq of " << coarse_center <<", docnum:" << coarse_doc_num[coarse_center];
        if(coarse_doc_num[coarse_center] < 256) {
          LOG(ERROR) << "docnum:"<< coarse_doc_num[coarse_center] << " < 256,for coarse:"<<coarse_center;
          continue;
        }
        if(FLAGS_train_lopq_global_o ==false) {

          //每个倒排链单独训练opq,然后旋转残差
          LOG(INFO) << "start train lopq of " << coarse_center <<",doc_num:" << coarse_doc_num[coarse_center];
          string lopq_residual_mean_file = lpq_file_prefix+"_"+std::to_string(coarse_center)+".lopq_residual_mean.fvecs";
          train_opq(coarse_doc_num[coarse_center], train_residual.data(), lpq_file_prefix+"_"+std::to_string(coarse_center)+".opq_matrix.fvecs",lopq_residual_mean_file);
          LOG(INFO) << "finish train lopq of " << coarse_center;
        } else {
          //大家公用一个已读取到的opq，直接旋转计算残差
          LOG(INFO) << "no need train lopq use global one,R*r directly";
          if(FLAGS_pq_minus_mean) {
            vector<float> mu(D);
            CHECK(fvecs_read(gopq_residual_mean_file.c_str(), D, 1, mu.data()) == 1);
            minus_mu(D,coarse_doc_num[coarse_center],train_residual.data(),mu.data());
          }
          float * r = new float[coarse_doc_num[coarse_center] * D];
          fmat_mul_full(global_o.get(), train_residual.data(), D, coarse_doc_num[coarse_center], D, "TN", r);
          memcpy(train_residual.data(),r,sizeof(float)*coarse_doc_num[coarse_center]*D);
          delete[] r;
        }

        LOG(INFO) << "start train pq of " << coarse_center <<", doc_num:" << coarse_doc_num[coarse_center];
        train_pq(coarse_doc_num[coarse_center], train_residual.data(), 
          lpq_file_prefix+"_"+std::to_string(coarse_center)+".pq_matrix.fvecs",
          lpq_file_prefix+"_"+std::to_string(coarse_center)+".pq.faiss.index");
        LOG(INFO) << "end train pq of " << coarse_center;
      }
    }
    

    free(coarseAssigns);
    free(fineAssigns);
    free(alphaNum);
    free(alphaDen);
    free(alpha);
    free(coarseVocab);
    free(coarseVocabNum);
    free(coarseVocabDen);
    free(fineVocab);
    free(fineVocabNum);
    free(fineVocabDen);
    free(coarseNorms);
    free(fineNorms);
    free(coarseFineProducts);
    free(errors);
    return 0;
}

