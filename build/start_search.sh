#./searchGNOIMI -threadsCount=10 --model_prefix=./sift1B/gnoimi_train_data/learn_from_yael_k4096 --index_prefix=./sift1B/gnoimi_train_data/index_k4096_20w_ --query_filename=./sift1B/sift_base.bvecs --base_file=./sift1B/sift_base.bvecs --groud_truth_file=sift1B/sift_groundtruth_0_1w.ivecs --N=200000 --neighborsCount=100 --queriesCount=10000 --make_index=false &>log.search &
./searchGNOIMI -threadsCount=10 --model_prefix=./sift1B/gnoimi_train_data/learn_from_yael_k4096 --index_prefix=./sift1B/gnoimi_train_data/index_k4096_20w_ --query_filename=./sift1B/sift_query.bvecs --base_file=./sift1B/sift_base.bvecs --groud_truth_file=sift1B/sift_groundtruth_20w.ivecs --N=200000 --neighborsCount=100 --queriesCount=10000 --make_index=true &>log.search &