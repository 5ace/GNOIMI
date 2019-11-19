#这里一般用于从原始数据训练opq，但论文中应该是对残差训练opq
./train_opq -M=16 -d=128 -n=1000000 -train_file=./sift1B/sift_learn.bvecs -opq_train_file=./sift1B/gnoimi_train_data/sift_learn_opq_M16i_n100w.fvecs -opq_matrix_file=./sift1B/gnoimi_train_data/opq_matrix.fvecs &>log.opq &
