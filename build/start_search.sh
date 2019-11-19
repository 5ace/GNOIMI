#./searchGNOIMI -threadsCount=10 --model_prefix=./sift1B/gnoimi_train_data/learn_from_yael_k4096 --index_prefix=./sift1B/gnoimi_train_data/index_k4096_20w_ --query_filename=./sift1B/sift_base.bvecs --base_file=./sift1B/sift_base.bvecs --groud_truth_file=sift1B/sift_groundtruth_0_1w.ivecs --N=200000 --neighborsCount=100 --queriesCount=10000 --make_index=false &>log.search &
#./searchGNOIMI -threadsCount=10 --L=60 --model_prefix=./sift1B/gnoimi_train_data/learn_from_yael_k4096 --index_prefix=./sift1B/gnoimi_train_data/index_k4096_20w_ --query_filename=./sift1B/sift_query.bvecs --base_file=./sift1B/sift_base.bvecs --groud_truth_file=sift1B/sift_groundtruth_20w.ivecs --N=200000 --neighborsCount=1000 --queriesCount=10000 --make_index=false &>log.search &
#./searchGNOIMI -threadsCount=10 --model_prefix=./sift1B/gnoimi_train_data/learn_from_yael_k512 --index_prefix=./sift1B/gnoimi_train_data/index_k512_20w_ --query_filename=./sift1B/sift_query.bvecs --base_file=./sift1B/sift_base.bvecs --groud_truth_file=sift1B/sift_groundtruth_20w.ivecs --N=200000 --neighborsCount=200 --queriesCount=10000 --make_index=true &>log.search &
#./searchGNOIMI -threadsCount=10 --residual_opq=true --L=32 --model_prefix=./sift1B/gnoimi_train_data/learn_from_yael_k4096_residual_opq_ --index_prefix=./sift1B/gnoimi_train_data/index_k4096_20w_residual_opq_ --query_filename=./sift1B/sift_query.bvecs --base_file=./sift1B/sift_base.bvecs --groud_truth_file=sift1B/sift_groundtruth_20w.ivecs --N=200000 --neighborsCount=200 --queriesCount=10000 --make_index=false &>log.search &
./searchGNOIMI -threadsCount=1 --residual_opq=true --L=32 --model_prefix=./sift1B/gnoimi_train_data/learn_from_yael_k4096_residual_opq_ --index_prefix=./sift1B/gnoimi_train_data/index_k4096_1kw_residual_opq_ --query_filename=./sift1B/sift_query.bvecs --base_file=./sift1B/sift_base.bvecs --groud_truth_file=sift1B/sift_groundtruth_1kw.ivecs --N=10000000 --neighborsCount=1000 --queriesCount=10000 --make_index=false &>log.search.sift1b.1kw.k4096 &




#faiss index
#./faiss_search_index --make_index=true --N=0 --query_N=10000 --base_file=mmu0.5b/mmu0.5b_base.fvecs --learn_N=0 --learn_file=mmu0.5b/mmu0.5b_learn.99w.fvecs --index_factory_str="OPQ16,IVF8192,PQ16" --index_file=./mmu0.5b/OPQ16,IVF8192,PQ16.0.5b.faiss.index  --groundtruth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --search_args="nprobe=10" --query_file=mmu0.5b/mmu0.5b_query.99w.fvecs &>log.index.mmu0.5b.faiss.0.5b

#./faiss_search_index --make_index=true --N=10000000 --base_file=./sift1B/sift_base.bvecs --learn_N=1000000 --learn_file=./sift1B/sift_learn.bvecs --index_factory_str="OPQ16,IVF8192,PQ16" --index_file=./sift1B/OPQ16,IVF8192,PQ16.1kw.faiss.index  --groundtruth_file=./sift1B/sift_groundtruth_1kw.ivecs  --search_args="nprobe=10" --query_file=./sift1B/sift_query.bvecs
#./faiss_search_index --N=10000000 --make_index=true --base_file=./mmu0.5b/mmu0.5b_base.fvecs --learn_N=0 --learn_file=./mmu0.5b/mmu0.5b_learn.99w.fvecs --index_factory_str="OPQ16,IVF8192,PQ16" --index_file=./mmu0.5b/OPQ16,IVF8192,PQ16.1kw.faiss.index  --groundtruth_file=./sift1B/sift_groundtruth.ivecs  --search_args="nprobe=10" 



#faiss search
#./faiss_search_index --make_index=false --N=10000000 --base_file=./sift1B/sift_base.bvecs --learn_N=1000000 --learn_file=./sift1B/sift_learn.bvecs --index_factory_str="OPQ16,IVF8192,PQ16" --index_file=./sift1B/OPQ16,IVF8192,PQ16.1kw.faiss.index  --groundtruth_file=./sift1B/sift_groundtruth_1kw.ivecs  --search_args="nprobe=10" --query_file=./sift1B/sift_query.bvecs
