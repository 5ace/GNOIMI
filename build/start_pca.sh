./faiss_search_index --D=1024 --make_index=true --search_index=true --N=0 --query_N=10000 --base_file=/home/zhangcunyi/data_set/mmu10m/base1024_new.fvecs --learn_N=1000000 --learn_file=/home/zhangcunyi/data_set/mmu10m/learn1024_new.fvecs --index_factory_str="PCA128,OPQ16,PQ16" --index_file=/home/zhangcunyi/data_set/mmu10m/PCA128,OPQ16,PQ16.1024.faiss.index  --groundtruth_file=/home/zhangcunyi/data_set/mmu10m/groundtruth1024.ivecs --search_args="nprobe=10" --query_file=/home/zhangcunyi/data_set/mmu10m/query1024_new.fvecs &>log.PCA128,OPQ16,PQ16.1024.faiss

./faiss_search_index --D=1024 --make_index=true --search_index=true --N=0 --query_N=10000 --base_file=/home/zhangcunyi/data_set/mmu10m/base1024_new.fvecs --learn_N=1000000 --learn_file=/home/zhangcunyi/data_set/mmu10m/learn1024_new.fvecs --index_factory_str="PCA64,OPQ16,PQ16" --index_file=/home/zhangcunyi/data_set/mmu10m/PCA64,OPQ16,PQ16.1024.faiss.index  --groundtruth_file=/home/zhangcunyi/data_set/mmu10m/groundtruth1024.ivecs --search_args="nprobe=10" --query_file=/home/zhangcunyi/data_set/mmu10m/query1024_new.fvecs &>log.PCA64,OPQ16,PQ16.1024.faiss