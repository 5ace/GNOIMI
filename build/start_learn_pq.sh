#./learnGNOIMI --n=1000000 --train_opq=false --direct_train_s_t_alpha=true --train_pq=true --thread_num=50 --k=4096 --learnIterationsCount=30 --outputFilesPrefix=sift1B/gnoimi_train_data/learn_from_yael_k4096 --initCoarseFilename=./sift1B/gnoimi_train_data/coarse_centroids_init_d128_k4096_yael.fvecs --initFineFilename=./sift1B/gnoimi_train_data/fine_centroids_init_d128_k4096_yael.fvecs --learnFilename=./sift1B/gnoimi_train_data/sift_learn_opq_M16i_n100w.fvecs &>log.learn_yael_opq_k4096
#./learnGNOIMI --n=1000000 --train_opq=false --direct_train_s_t_alpha=true --train_pq=true --thread_num=50 --k=512 --learnIterationsCount=30 --outputFilesPrefix=sift1B/gnoimi_train_data/learn_from_yael_k512 --initCoarseFilename=./sift1B/gnoimi_train_data/coarse_centroids_init_d128_k512_yael.fvecs --initFineFilename=./sift1B/gnoimi_train_data/fine_centroids_init_d128_k512_yael.fvecs --learnFilename=./sift1B/gnoimi_train_data/sift_learn_opq_M16i_n100w.fvecs &>log.learn_yael_opq_k512



#训练残差opq
./learnGNOIMI --n=1000000 --train_opq=true --direct_train_s_t_alpha=true --train_pq=true --thread_num=50 --k=4096 --learnIterationsCount=30 --outputFilesPrefix=sift1B/gnoimi_train_data/learn_from_yael_k4096_residual_opq_ --initCoarseFilename=./sift1B/gnoimi_train_data/coarse_centroids_init_d128_k4096_residual_opq_yael.fvecs --initFineFilename=./sift1B/gnoimi_train_data/fine_centroids_init_d128_k4096_residual_opq_yael.fvecs --learnFilename=./sift1B/sift_learn.bvecs &>log.learn_yael_opq_k4096_residual_opq

