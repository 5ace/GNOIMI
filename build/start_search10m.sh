#SV 64

#gen init

#./gen_init_data --train_file=./mmu10m/dimreduce.spreadvector.learn.d64.n5000000.fvecs --n=1000000 --use_yael=true --iter_num=30 --thread_num=50 --d=64 --k=2048 --coarse_centriods_file=./mmu10m/gnoimi2048/coarse_centroids_init_d64_k2048_residual_opq_yael.fvecs --residual_centriods_file=./mmu10m/gnoimi2048/fine_centroids_init_d64_k2048_residual_opq_yael.fvecs &>log.init.yael.sv.k2048.d64_residual_opq.mmu10m 
#./learnGNOIMI --n=1000000 --train_opq=true --direct_train_s_t_alpha=true --train_pq=true --train_lopq=false --thread_num=50 --k=2048 --learnIterationsCount=30 --outputFilesPrefix=./mmu10m/gnoimi2048/learn_from_yael_d64_sv_k2048_residual_opq_ --initCoarseFilename=./mmu10m/gnoimi2048/coarse_centroids_init_d64_k2048_residual_opq_yael.fvecs --initFineFilename=./mmu10m/gnoimi2048/fine_centroids_init_d64_k2048_residual_opq_yael.fvecs --learnFilename=./mmu10m/dimreduce.spreadvector.learn.d64.n5000000.fvecs --d=64 &>log.learn_yael_opq_d64_k2048_residual_opq.mmu10m.sv
#k2048 gopq  1kw index
./searchGNOIMI -threadsCount=42 --K=2048 --residual_opq=true --L=32 --index_prefix=./mmu10m/gnoimi2048/index_k2048_d64_sv_1kw_residual_opq_ --model_prefix=./mmu10m/gnoimi2048/learn_from_yael_d64_sv_k2048_residual_opq_  --query_filename=./mmu10m/dimreduce.spreadvector.query.d64.n10000.fvecs --base_file=./mmu10m/dimreduce.spreadvector.base.d64.n10000000.fvecs --groud_truth_file=./mmu10m/groundtruth1024.ivecs --N=0 --neighborsCount=200 --queriesCount=10000 --make_index=true --D=64 &>log.index.mmu10m.gnoimi.1kw.sv.d64.k2048.gopq

#pca 128
./searchGNOIMI -threadsCount=42 --K=2048 --residual_opq=true --L=32 --index_prefix=./mmu10m/gnoimi2048/index_k2048_1kw_residual_pca128_1kw_opq_ --model_prefix=./mmu0.5b/learn_from_yael_k2048_residual_opq_  --query_filename=mmu10m/query128_new.fvecs --base_file=mmu10m/base128_new.fvecs --groud_truth_file=./mmu10m/groundtruth1024.ivecs --N=0 --neighborsCount=200 --queriesCount=10000 --make_index=true &>log.index.mmu10m.gnoimi.pca128.1kw.k2048.gopq


#faiss search sv64
./faiss_search_index --D=64 --search_index=true --make_index=true --N=0 --base_file=./mmu10m/dimreduce.spreadvector.base.d64.n10000000.fvecs --learn_N=1000000 --learn_file=./mmu10m/dimreduce.spreadvector.learn.d64.n5000000.fvecs --index_factory_str="OPQ16,IVF8192,PQ16" --index_file=./mmu10m/spreadvector.d64.OPQ16,IVF8192,PQ16.index  --groundtruth_file=./mmu10m/groundtruth1024.ivecs --search_args="nprobe=10|nprobe=20|nprobe=30|nprobe=50" --query_file=./mmu10m/dimreduce.spreadvector.query.d64.n10000.fvecs &>log.mmu10m.sv.OPQ16,IVF8192,PQ16 &
#pca 128
./faiss_search_index --D=128 --search_index=true --make_index=true --N=0 --base_file=./mmu10m/base128_new.fvecs --learn_N=1000000 --learn_file=./mmu0.5b/mmu0.5b_learn.198w.fvecs --index_factory_str="OPQ16,IVF8192,PQ16" --index_file=./mmu10m/pca128.OPQ16,IVF8192,PQ16.index  --groundtruth_file=./mmu10m/groundtruth1024.ivecs --search_args="nprobe=10|nprobe=20|nprobe=30|nprobe=50" --query_file=./mmu10m/query128_new.fvecs &>log.mmu10m.pca128.OPQ16,IVF8192,PQ16 &
