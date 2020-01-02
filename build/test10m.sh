
cd /kml_zhangcunyi/code/GNOIMI/build/
export LD_LIBRARY_PATH="../../../lib/openblas/;../../../lib/gflags;../../glog;../../yael_v438/yael"



#L=(40)
#nprobe=(2000)
#neighborcount=(5000)

L=( 30 40 40 50 60)
nprobe=(200 300 500 1000 2000)
neighborcount=(10000 30000 50000 100000 150000)
[ ${#L[*]} -eq ${#nprobe[*]} ] && [ ${#L[*]} -eq ${#neighborcount[*]} ] || ( echo "array num error ${#L[*]} ${#nprobe[*]} ${#neighborcount[*]} " && exit -1 )
:<<EOF
#sv64 gnoimi
log=/home/web_server/zhangcunyi/testGNOIMI/log.gnoimi.mmu10m.diffargs.sv64.k2048.gopq
>$log
i=0
while [ $i -lt ${#L[*]} ]
do
  numactl -C 0 -m 0 ./searchGNOIMI -threadsCount=1 --K=2048 --residual_opq=true --L=${L[$i]} --index_prefix=./mmu10m/gnoimi2048/index_k2048_d64_sv_1kw_residual_opq_ --model_prefix=./mmu10m/gnoimi2048/learn_from_yael_d64_sv_k2048_residual_opq_  --query_filename=./mmu10m/dimreduce.spreadvector.query.d64.n10000.fvecs --base_file=./mmu10m/dimreduce.spreadvector.base.d64.n10000000.fvecs --groud_truth_file=./mmu10m/groundtruth1024.ivecs --N=0 --neighborsCount=${neighborcount[$i]} --queriesCount=10000  --nprobe=${nprobe[$i]} --make_index=false --D=64 &>>$log 
  let i++
done

#pca128
L=( 30 40 40 50 60)
nprobe=(200 300 500 1000 2000)
neighborcount=(10000 30000 50000 100000 150000)

log=/home/web_server/zhangcunyi/testGNOIMI/log.gnoimi.mmu10m.diffargs.pca128.k2048.gopq
>$log
i=0
while [ $i -lt ${#L[*]} ]
do
  numactl -C 0 -m 0 ./searchGNOIMI -threadsCount=1 --K=2048 --residual_opq=true --L=${L[$i]} --index_prefix=./mmu10m/gnoimi2048/index_k2048_1kw_residual_pca128_1kw_opq_ --model_prefix=./mmu0.5b/learn_from_yael_k2048_residual_opq_  --query_filename=mmu10m/query128_new.fvecs --base_file=mmu10m/base128_new.fvecs --groud_truth_file=./mmu10m/groundtruth1024.ivecs --N=0 --neighborsCount=${neighborcount[$i]} --queriesCount=10000 --make_index=false  --nprobe=${nprobe[$i]} &>>$log 
  let i++
done
EOF
#numactl -C 0 -m 0 ./faiss_search_index --D=64 --search_index=true --make_index=false --N=0 --base_file=./mmu10m/dimreduce.spreadvector.base.d64.n10000000.fvecs --learn_N=1000000 --learn_file=./mmu10m/dimreduce.spreadvector.learn.d64.n5000000.fvecs --index_factory_str="OPQ16,IVF8192,PQ16" --index_file=./mmu10m/spreadvector.d64.OPQ16,IVF8192,PQ16.index  --groundtruth_file=./mmu10m/groundtruth1024.ivecs --search_args="nprobe=10|nprobe=20|nprobe=30|nprobe=50" --query_file=./mmu10m/dimreduce.spreadvector.query.d64.n10000.fvecs &>/home/web_server/zhangcunyi/testGNOIMI/log.mmu10m.sv64.OPQ16,IVF8192,PQ16
numactl -C 0 -m 0 ./faiss_search_index --D=128 --search_index=true --make_index=false --N=0 --base_file=./mmu10m/base128_new.fvecs --learn_N=1000000 --learn_file=./mmu0.5b/mmu0.5b_learn.198w.fvecs --index_factory_str="OPQ16,IVF8192,PQ16" --index_file=./mmu10m/pca128.OPQ16,IVF8192,PQ16.index  --groundtruth_file=./mmu10m/groundtruth1024.ivecs --search_args="nprobe=10|nprobe=20|nprobe=30|nprobe=40|nprobe=50" --query_file=./mmu10m/query128_new.fvecs &>/home/web_server/zhangcunyi/testGNOIMI/log.mmu10m.pca128.OPQ16,IVF8192,PQ16 
