#nohup ./gen_init_data --train_file=./sift1B/gnoimi_train_data/sift_learn_opq_M16i_n100w.fvecs --n=1000000 --use_yael=true --iter_num=20 --thread_num=50 --d=128 --k=4096 --coarse_centriods_file=sift1B/gnoimi_train_data/coarse_centroids_init_d128_k4096_yael.fvecs --residual_centriods_file=sift1B/gnoimi_train_data/fine_centroids_init_d128_k4096_yael.fvecs &>log.init.yael.k4096 &
#nohup ./gen_init_data --train_file=./sift1B/gnoimi_train_data/sift_learn_opq_M16i_n100w.fvecs --n=1000000 --use_yael=true --iter_num=20 --thread_num=50 --d=128 --k=512 --coarse_centriods_file=sift1B/gnoimi_train_data/coarse_centroids_init_d128_k512_yael.fvecs --residual_centriods_file=sift1B/gnoimi_train_data/fine_centroids_init_d128_k512_yael.fvecs &>log.init.yael.k512 &

nohup ./gen_init_data --train_file=./sift1B/sift_learn.bvecs --n=1000000 --use_yael=true --iter_num=30 --thread_num=50 --d=128 --k=4096 --coarse_centriods_file=sift1B/gnoimi_train_data/coarse_centroids_init_d128_k4096_residual_opq_yael.fvecs --residual_centriods_file=sift1B/gnoimi_train_data/fine_centroids_init_d128_k4096_residual_opq_yael.fvecs &>log.init.yael.k4096_residual_opq.sift1b &


#mmu0.5b
#./gen_init_data --train_file=mmu0.5b/mmu0.5b_learn.99w.fvecs  --n=0 --use_yael=true --iter_num=30 --thread_num=50 --d=128 --k=4096 --coarse_centriods_file=mmu0.5b/coarse_centroids_init_d128_k4096_residual_opq_yael.fvecs --residual_centriods_file=mmu0.5b/fine_centroids_init_d128_k4096_residual_opq_yael.fvecs &>log.init.yael.k4096_residual_opq.mmu0.5b
