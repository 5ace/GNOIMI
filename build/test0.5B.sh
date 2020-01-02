
cd /kml_zhangcunyi/code/GNOIMI/build/
export LD_LIBRARY_PATH="../../../lib/openblas/;../../../lib/gflags;../../glog;../../yael_v438/yael"


L=( 40 40 40 50 60)
nprobe=(200 300 500 1000 2000)
neighborcount=(50000 100000 130000 200000 300000)

#L=(40)
#nprobe=(2000)
#neighborcount=(5000)

[ ${#L[*]} -eq ${#nprobe[*]} ] && [ ${#L[*]} -eq ${#neighborcount[*]} ] || ( echo "array num error ${#L[*]} ${#nprobe[*]} ${#neighborcount[*]} " && exit -1 )
L=( 32 40 40 40 50 60 60 60 )
nprobe=(150 200 300 500 1000 2000 2000 2200)
neighborcount=(30000 50000 100000 130000 200000 300000 400000 500000)
L=(32 40 40 40 50 60 60 60 70 80 90)
nprobe=(150 200 300 500 800 1300 1600 2000 3000 4000 5000)
neighborcount=(30000 50000 60000 100000 150000 200000 250000 300000 360000 500000 600000)
#neighborcount=(20000 40000 50000 80000 120000 150000 180000 220000 250000)

log=/home/web_server/zhangcunyi/testGNOIMI/log.gnoimi.mmu0.5b.diffargs.k2048.lopq.lpq.minus.decom
>$log
i=0
while [ $i -lt ${#L[*]} ]
do
  #0.5b lopq lpq k3072
  #numactl -C 0 -m 0 ./searchGNOIMI -threadsCount=1 --K=3072 --residual_opq=true --L=${L[$i]} --lpq_file_prefix=mmu0.5b/lpq_k3072/ --model_prefix=./mmu0.5b/learn_from_yael_k3072_residual_opq_ --index_prefix=./mmu0.5b/index_k3072_0.5b_residual_opq_lopq_lpq_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --N=0 --neighborsCount=${neighborcount[$i]} --queriesCount=10000 --make_index=false --nprobe=${nprobe[$i]}  &>>$log 
  #0.5b lopq lpq minus k3072
  #numactl -C 0 -m 0 ./searchGNOIMI -threadsCount=1 --pq_minus_mean=true --K=3072 --residual_opq=true --lpq_file_prefix=mmu0.5b/lpq_k3072_minus/ --pq_prefix=mmu0.5b/learn_from_yael_k3072_residual_opq_minus_mu_ --model_prefix=mmu0.5b/learn_from_yael_k3072_residual_opq_   --index_prefix=./mmu0.5b/index_k3072_0.5b_residual_opq_lopq_lpq_minus_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --N=0 --queriesCount=10000 --make_index=false --nprobe=${nprobe[$i]} --neighborsCount=${neighborcount[$i]} --L=${L[$i]} &>>$log 
  #0.5b lopq lpq minus decompostion
  numactl -C 0 -m 0 ./search_GNOIMI_deco -threadsCount=1 --pq_minus_mean=true --K=2048 --residual_opq=true --lpq_file_prefix=mmu0.5b/lpq_k2048_minus/ --pq_prefix=mmu0.5b/learn_from_yael_k2048_residual_opq_minus_mu_ --model_prefix=mmu0.5b/learn_from_yael_k2048_residual_opq_   --index_prefix=./mmu0.5b/index_k2048_0.5b_residual_opq_lopq_lpq_minus_decom_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --N=0  --queriesCount=10000 --nprobe=${nprobe[$i]} --neighborsCount=${neighborcount[$i]} --L=${L[$i]} --make_index=false &>>$log
  #0.5b lopq lpq minus
  #numactl -C 0 -m 0 ./searchGNOIMI -threadsCount=1 --pq_minus_mean=true --K=2048 --residual_opq=true --L=32 --lpq_file_prefix=mmu0.5b/lpq_k2048_minus/ --pq_prefix=mmu0.5b/learn_from_yael_k2048_residual_opq_minus_mu_ --model_prefix=mmu0.5b/learn_from_yael_k2048_residual_opq_   --index_prefix=./mmu0.5b/index_k2048_0.5b_residual_opq_lopq_lpq_minus_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --N=0 --queriesCount=10000 --make_index=false --nprobe=${nprobe[$i]} --neighborsCount=${neighborcount[$i]} --L=${L[$i]} &>>$log 
  #0.5b lopq lpq
  #numactl -C 0 -m 0 ./searchGNOIMI -threadsCount=1 --K=2048 --residual_opq=true --L=${L[$i]} --lpq_file_prefix=mmu0.5b/lpq_k2048/ --model_prefix=./mmu0.5b/learn_from_yael_k2048_residual_opq_ --index_prefix=./mmu0.5b/index_k2048_0.5b_residual_opq_lopq_lpq_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --N=0 --neighborsCount=${neighborcount[$i]} --queriesCount=10000 --make_index=false --nprobe=${nprobe[$i]}  &>>$log 
  #1kw lopq lpq
  #numactl -C 0 -m 0 ./searchGNOIMI -threadsCount=1 --K=2048 --residual_opq=true --L=${L[$i]} --lpq_file_prefix=mmu0.5b/lpq_k2048/ --model_prefix=./mmu0.5b/learn_from_yael_k2048_residual_opq_ --index_prefix=./mmu0.5b/index_k2048_1kw_residual_opq_lopq_lpq_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_1kw.ivecs --N=10000000 --neighborsCount=${neighborcount[$i]} --queriesCount=10000 --make_index=false --nprobe=${nprobe[$i]}  &>>$log 
  #1kw gopq lpq
  #numactl -C 0 -m 0 ./searchGNOIMI -threadsCount=1 --K=2048 --residual_opq=true --L=${L[$i]} --lpq_file_prefix=mmu0.5b/lpq_k2048_global_o/ --model_prefix=./mmu0.5b/learn_from_yael_k2048_residual_opq_ --index_prefix=./mmu0.5b/index_k2048_1kw_residual_opq_gopq_lpq_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_1kw.ivecs --N=10000000 --neighborsCount=${neighborcount[$i]} --queriesCount=10000 --make_index=false --nprobe=${nprobe[$i]}  &>>$log 
  #1kw gopq
  #numactl -C 0 -m 0 ./searchGNOIMI -threadsCount=1 --K=2048 --residual_opq=true --L=${L[$i]} --model_prefix=./mmu0.5b/learn_from_yael_k2048_residual_opq_ --index_prefix=./mmu0.5b/index_k2048_1kw_residual_opq_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_1kw.ivecs --N=10000000 --neighborsCount=${neighborcount[$i]} --queriesCount=10000 --make_index=false --nprobe=${nprobe[$i]}  &>>$log 
  let i++
done
:<<EOF
log=/home/web_server/zhangcunyi/testGNOIMI/log.gnoimi.mmu0.5b.diffargs.k2048.gopq.lpq
>$log
i=0
while [ $i -lt ${#L[*]} ]
do
  #0.5b gopq lpq
  numactl -C 0 -m 0 ./searchGNOIMI -threadsCount=1 --K=2048 --residual_opq=true --L=${L[$i]} --lpq_file_prefix=mmu0.5b/lpq_k2048_global_o/ --model_prefix=./mmu0.5b/learn_from_yael_k2048_residual_opq_ --index_prefix=./mmu0.5b/index_k2048_0.5b_gopq_lpq_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --N=0 --neighborsCount=${neighborcount[$i]} --queriesCount=10000 --make_index=false --nprobe=${nprobe[$i]}  &>>$log 
  let i++
done
L=(60)
nprobe=(2000)
neighborcount=(300000)
log=/home/web_server/zhangcunyi/testGNOIMI/log.gnoimi.mmu0.5b.diffargs.k2048.gopq
>$log
i=0
while [ $i -lt ${#L[*]} ]
do
  #0.5b gopq
  numactl -C 0 -m 0 ./searchGNOIMI --K=2048 -threadsCount=1 --queriesCount=10000  --residual_opq=true  --model_prefix=./mmu0.5b/learn_from_yael_k2048_residual_opq_ --index_prefix=./mmu0.5b/index_k2048_0.5b_residual_opq_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --N=0 --make_index=false --L=${L[$i]} --neighborsCount=${neighborcount[$i]} --nprobe=${nprobe[$i]} &>>$log
  let i++
done
#########K4096######
L=(32 40 40 45 50 60 65)
nprobe=(2000 2000 3000 4000 5000 7000 10000)
neighborcount=(30000 50000 100000 150000 200000 300000 400000)
L=(65 )
nprobe=(10000)
neighborcount=(400000)

log=/home/web_server/zhangcunyi/testGNOIMI/log.gnoimi.mmu0.5b.diffargs.k4096.gopq
>$log
i=0
while [ $i -lt ${#L[*]} ]
do
  #0.5b gopq
  numactl -C 0 -m 0 ./searchGNOIMI --K=4096 -threadsCount=1 --queriesCount=10000  --residual_opq=true  --model_prefix=./mmu0.5b/learn_from_yael_k4096_residual_opq_ --index_prefix=./mmu0.5b/index_k4096_0.5b_residual_opq_ --query_filename=mmu0.5b/mmu0.5b_query.99w.fvecs --base_file=mmu0.5b/mmu0.5b_base.fvecs --groud_truth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --N=0 --make_index=false --L=${L[$i]} --neighborsCount=${neighborcount[$i]} --nprobe=${nprobe[$i]} &>>$log
  let i++
done


log=/home/web_server/zhangcunyi/testGNOIMI/log.faiss.OPQ16,IVF8192,PQ16.mmu0.5b.diffargs
numactl -C 0 -m 0 ./faiss_search_index --make_index=false --N=0 --query_N=10000 --base_file=mmu0.5b/mmu0.5b_base.fvecs --learn_N=0 --learn_file=mmu0.5b/mmu0.5b_learn.198w.fvecs --index_factory_str="OPQ16,IVF8192,PQ16" --index_file=./mmu0.5b/OPQ16,IVF8192,PQ16.0.5b.faiss.index  --groundtruth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --search_args="nprobe=4|nprobe=8|nprobe=20|nprobe=40|nprobe=60|nprobe=80" --query_file=mmu0.5b/mmu0.5b_query.99w.fvecs &>$log 
log=/home/web_server/zhangcunyi/testGNOIMI/log.faiss.OPQ16,IMI2x12,PQ16.mmu0.5b.diffargs
numactl -C 0 -m 0 ./faiss_search_index --make_index=false --N=0 --query_N=10000 --base_file=mmu0.5b/mmu0.5b_base.fvecs --learn_N=0 --learn_file=mmu0.5b/mmu0.5b_learn.198w.fvecs --index_factory_str="OPQ16,IMI2x12,PQ16" --index_file=./mmu0.5b/OPQ16,IMI2x12,PQ16.0.5b.faiss.index  --groundtruth_file=mmu0.5b/mmu0.5b_groudtruth_0.5b.ivecs --search_args="nprobe=8000|nprobe=10000" --query_file=mmu0.5b/mmu0.5b_query.99w.fvecs &>$log 
nprobe=(150 200 300 500 800 1300 1600 2000 3000)
neighborcount=(30000 50000 100000 130000 200000 300000 400000 500000)
log=/home/web_server/zhangcunyi/testGNOIMI/log.ivf-hnsw-group-prunin.0.5b
cd /kml_zhangcunyi/code/ivf-hnsw/examples/
>$log
i=0
while [ $i -lt ${#nprobe[*]} ]
do
 sh  run_mmu0.5b_grouping_OPQ.sh.numa ${nprobe[$i]} ${neighborcount[$i]} "on" &>>$log 
 let i++
done

i=0
while [ $i -lt ${#nprobe[*]} ]
do
 sh run_mmu0.5b_grouping_OPQ.sh.numa ${nprobe[$i]} ${neighborcount[$i]} "off" &>>$log 
 let i++
done
EOF
