#./gen_groudth --N=10000000 -base_file=./sift1B/sift_base.bvecs --query_filename=./sift1B/sift_query.bvecs --query_N=10000 -groud_truth_file=sift1B/sift_groundtruth_1kw.ivecs &>log.gen_groudtruth &
./gen_groudth --N=500000000 -base_file=./sift1B/sift_base.bvecs --query_filename=./sift1B/sift_query.bvecs --query_N=10000 -groud_truth_file=sift1B/sift_groundtruth_1kw.ivecs &>log.gen_groudtruth &
./gen_groudth --N=10000000 -base_file=./mmu0.5b/mmu0.5b_base.fvecs --query_filename=./mmu0.5b/mmu0.5b_query.99w.fvecs --query_N=50000 -groud_truth_file=./mmu0.5b/mmu0.5b_groudtruth &>log.gen_groudtruth &
