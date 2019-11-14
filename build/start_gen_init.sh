#nohup ./gen_init_data --train_file=sift1B/sift_learn.bvecs --n=500000 --use_yael=true --iter_num=100 --thread_num=10 --d=128 --k=4096 --coarse_centriods_file=sift1B/gnoimi_train_data/coarse_centroids_init_d128_k4096_yael.fvecs --residual_centriods_file=sift1B/gnoimi_train_data/fine_centroids_init_d128_k4096_yael.fvecs &>log.init.yael &
nohup ./gen_init_data --train_file=./sift1B/gnoimi_train_data/sift_learn_opq_M16i_n100w.fvecs --n=1000000 --use_yael=true --iter_num=20 --thread_num=50 --d=128 --k=4096 --coarse_centriods_file=sift1B/gnoimi_train_data/coarse_centroids_init_d128_k4096_yael.fvecs --residual_centriods_file=sift1B/gnoimi_train_data/fine_centroids_init_d128_k4096_yael.fvecs &>log.init.yael &
