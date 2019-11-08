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
#include <faiss/utils/distances.h>
#include <gflags/gflags.h>

#ifdef __cplusplus
extern "C"{
#endif

#include <yael/kmeans.h>
#include <yael/vector.h>
#include <yael/matrix.h>

int fvecs_read(const char *fname, int d, int n, float *v);
int ivecs_new_read(const char *fname, int *d_out, int **vi);
void fmat_mul_full(const float *left, const float *right,
                   int m, int n, int k,
                   const char *transp,
                   float *result);
int fvec_read(const char *fname, int d, float *a, int o_f);
int* ivec_new_read(const char *fname, int *d_out);
void fmat_rev_subtract_from_columns(int d,int n,float *m,const float *avg);
void fvec_sub(float * v1, const float * v2, long n);
int b2fvecs_read(const char *fname, int d, int n, float *v);
void fvec_add(float * v1, const float * v2, long n);
float* fmat_new_transp(const float *a, int ncol, int nrow);

#ifdef __cplusplus
}
#endif

using std::cout;
using std::ios;
using std::string;
using std::vector;

DEFINE_int32(thread_num,10,"train S,T, alpha thread num");
DEFINE_int32(k,4096,"coarse and residual centriods num");
DEFINE_int32(d,128,"feature dim");
DEFINE_int32(n,1000000,"learn feature num, each thread_num train n/thread_num");
DEFINE_int32(learnIterationsCount,10,"learnIterationsCount");
DEFINE_int32(l,8,"");
DEFINE_string(initCoarseFilename,"","initCoarseFilename fvecs, centorids of coarse");
DEFINE_string(initFineFilename,"","initFineFilename fvecs, centorids of residual");
DEFINE_string(learnFilename,"","learnFilename fvecs or bvecs, must normlized");
DEFINE_string(outputFilesPrefix,"./train_gnoimi_","prefix of output file name");
DEFINE_int32(trainThreadChunkSize,10000,"trainThreadChunkSize");

size_t D;
int K;
size_t totalLearnCount;
int learnIterationsCount;
int L;
string learnFilename; 
string initCoarseFilename;
string initFineFilename;
string outputFilesPrefix;
int trainThreadChunkSize;
int threadsCount;

float* train_vecs;
int* coarseAssigns;
int* fineAssigns;
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

vector<float*> fineVocabNumerators;
vector<float*> fineVocabDenominators;
vector<float*> coarseVocabNumerators;
vector<float*> coarseVocabDenominators;

float* coarseNorms;
float* fineNorms;
float* coarseFineProducts;

float* errors;

///////////////////////////
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
      // 这里想计算|p-S|
      // 转换为 p*p/2-P*S +S*S,由于对于同一个p,p*p/2=0.5是一样的所以排序时排除即可
      cblas_saxpy(K, -1.0, coarseNorms, 1, pointsCoarseTerms + pointId * K, 1);
      for(int k = 0; k < K; ++k) {
        // 这first就是每个查询向量与所有一级类中心的近似距离,S*S/2-p*S
        coarseScores[k].first = (-1.0) * pointsCoarseTerms[pointId * K + k];
        coarseScores[k].second = k;
      }
      std::sort(coarseScores.begin(), coarseScores.end());
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

    D = FLAGS_d;
    K = FLAGS_k;
    totalLearnCount = FLAGS_n;
    learnIterationsCount = FLAGS_learnIterationsCount;
    L = FLAGS_l;
    learnFilename = FLAGS_learnFilename; 
    initCoarseFilename = FLAGS_initCoarseFilename;
    initFineFilename = FLAGS_initFineFilename;
    outputFilesPrefix = FLAGS_outputFilesPrefix;
    trainThreadChunkSize = FLAGS_trainThreadChunkSize;
    threadsCount = FLAGS_thread_num;

    coarseAssigns = (int*)malloc(totalLearnCount * sizeof(int));
    fineAssigns = (int*)malloc(totalLearnCount * sizeof(int));
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
}
int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    //初始化全局变量 =====
    FLAGS_logtostderr = true; //日志全部输出到标准错误
    init_global_varibles();
    std::shared_ptr<float> features = gnoimi::read_bfvecs(learnFilename.c_str(), D,  totalLearnCount); 
    train_vecs = features.get();

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
  LOG(INFO) << initCoarseFilename << " "<< initFineFilename << " " <<
      learnFilename << " " << outputFilesPrefix;

  CHECK(!initCoarseFilename.empty() && !initFineFilename.empty() && !learnFilename.empty() && !outputFilesPrefix.empty());
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
    //计算每个类中心的内积,用于计算距离使用
    for(int k = 0; k < K; ++k) {
      coarseNorms[k] = cblas_sdot(D, coarseVocab + k * D, 1, coarseVocab + k * D, 1) / 2;
      fineNorms[k] = cblas_sdot(D, fineVocab + k * D, 1, fineVocab + k * D, 1) / 2;
    }
    //矩阵相乘 coarseFineProducts=fineVocab*transpose(coarseVocab) 得到K*K的矩阵
    fmat_mul_full(fineVocab, coarseVocab, K, K, D, "TN", coarseFineProducts);
    // https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Babenko_Efficient_Indexing_of_CVPR_2016_paper.pdf
    // 见这里迭代 优化4个参数
    // update Assigns
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
    LOG(INFO) << "Current reconstruction error... " << totalError / totalLearnCount << "\n";

    workers.clear();
    // update alpha
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
    string alphaFilename = outputFilesPrefix +"alpha_" + std::to_string(it) + ".fvecs";
    gnoimi::fvecs_write(alphaFilename.c_str(),K,K,alpha);
    string fineVocabFilename = outputFilesPrefix + "fine_" + std::to_string(it) + ".fvecs";
    gnoimi::fvecs_write(fineVocabFilename.c_str(),D,K,fineVocab);
    string coarseVocabFilename = outputFilesPrefix + "coarse_" + std::to_string(it) + ".fvecs";
    gnoimi::fvecs_write(coarseVocabFilename.c_str(),D,K,coarseVocab);
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











