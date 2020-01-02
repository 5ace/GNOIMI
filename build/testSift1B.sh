cd /kml_zhangcunyi/code/GNOIMI/build/
export LD_LIBRARY_PATH="../../../lib/openblas/;../../../lib/gflags;../../glog;../../yael_v438/yael"

L=(32 40 40 40 50 60 60 60 70 80 90)
nprobe=(150 200 300 500 800 1300 1600 2000 3000 4000 5000)
neighborcount=(30000 50000 60000 100000 150000 200000 250000 300000 360000 500000 600000)
[ ${#L[*]} -eq ${#nprobe[*]} ] && [ ${#L[*]} -eq ${#neighborcount[*]} ] || ( echo "array num error ${#L[*]} ${#nprobe[*]} ${#neighborcount[*]} " && exit -1 )
:<<EOF
log=/home/web_server/zhangcunyi/testGNOIMI/log.gnoimi.sift1b.diffargs.k2048.lopq.mu.deco
>$log
i=0
while [ $i -lt ${#L[*]} ]
do
  numactl -C 0 -m 0 ./search_GNOIMI_deco -threadsCount=1 --pq_minus_mean=true --K=2048 --residual_opq=true --lpq_file_prefix=sift1B/lpq_k2048_minus/ --pq_prefix=sift1B/lpq_k2048_minus/learn_from_yael_k2048_residual_opq_minus_mu_ --model_prefix=sift1B/lpq_k2048_minus/learn_from_yael_k2048_residual_opq_ --index_prefix=./sift1B/lpq_k2048_minus/index_k2048_0.5b_residual_opq_lopq_lpq_minus_decom_ --query_filename=./sift1B/sift_query.bvecs --base_file=./sift1B/sift_base.bvecs --groud_truth_file=sift1B/sift_groundtruth.ivecs --N=0  --queriesCount=10000  --nprobe=${nprobe[$i]} --neighborsCount=${neighborcount[$i]} --L=${L[$i]}  --make_index=false &>>$log 
  let i++
done
EOF
#ivf adc
log=/home/web_server/zhangcunyi/testGNOIMI/log.ivfadc.sift1b.diffargs.OPQ16,IVF8192,PQ16
numactl -C 0 -m 0 ./faiss_search_index --make_index=false --N=0 --query_N=10000 --base_file=./sift1B/sift_base.bvecs --learn_N=0 --learn_file=./sift1B/sift_learn.bvecs --index_factory_str="OPQ16,IVF8192,PQ16" --index_file=./sift1B/OPQ16,IVF8192,PQ16.index  --groundtruth_file=./sift1B/sift_groundtruth.ivecs --search_args="nprobe=2|nprobe=4|nprobe=8|nprobe=10|nprobe=15|nprobe=20" --query_file=./sift1B/sift_query.bvecs &>$log 
#imi
#log=/home/web_server/zhangcunyi/testGNOIMI/log.ivfadc.sift1b.diffargs.OPQ16,IMI2x12,PQ16
#numactl -C 0 -m 0 ./faiss_search_index --make_index=false --N=0 --query_N=10000 --base_file=./sift1B/sift_base.bvecs --learn_N=0 --learn_file=./sift1B/sift_learn.bvecs --index_factory_str="OPQ16,IMI2x12,PQ16" --index_file=./sift1B/OPQ16,IMI2x12,PQ16.index  --groundtruth_file=./sift1B/sift_groundtruth.ivecs --search_args="nprobe=100|nprobe=250|nprobe=500|nprobe=1000|nprobe=2000|nprobe=4000|nprobe=8000" --query_file=./sift1B/sift_query.bvecs &>$log 
#log=/home/web_server/zhangcunyi/testGNOIMI/log.ivfadc.sift1b.diffargs.OPQ16,IMI2x12,PQ16.ht
#numactl -C 0 -m 0 ./faiss_search_index --make_index=false --N=0 --query_N=10000 --base_file=./sift1B/sift_base.bvecs --learn_N=0 --learn_file=./sift1B/sift_learn.bvecs --index_factory_str="OPQ16,IMI2x12,PQ16" --index_file=./sift1B/OPQ16,IMI2x12,PQ16.index  --groundtruth_file=./sift1B/sift_groundtruth.ivecs --search_args="nprobe=100,ht=55|nprobe=500,ht=55|nprobe=1000,ht=55|nprobe=2000,ht=55|nprobe=4000,ht=55|nprobe=8000,ht=55" --query_file=./sift1B/sift_query.bvecs &>$log 
