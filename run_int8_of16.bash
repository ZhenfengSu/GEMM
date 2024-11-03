# python gemm_int8_of16.py --shape_row 4096 --shape_col 4096 --output shape_4096_info.txt
# python gemm_int8_of16.py --shape_row 4096 --shape_col 4096 --data_type FP16 --plot_mode True --output shape_4096_fp16_info.txt
# # 8192 方阵
# python gemm_int8_of16.py --shape_row 8192 --shape_col 8192 --plot_mode True --output shape_8192_int8_info.txt
# python gemm_int8_of16.py --shape_row 4096 --shape_col 16384 --plot_mode True --output shape_4096_16384_int8_info.txt
# python gemm_int8_of16.py --shape_row 16384 --shape_col 4096 --plot_mode True --output shape_16384_4096_int8_info.txt

log_file="./txt/output.txt"
# 创建txt文件夹（如果不存在的话）
if [ ! -d "./txt" ]; then
  mkdir ./txt
fi
# 创建output.txt文件（如果不存在的话）
if [ ! -f $log_file ]; then
  touch $log_file
fi


python gemm_int8_of16.py --shape_row 512 --shape_col 512 --plot_mode True --output shape_512_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 768 --shape_col 768 --plot_mode True --output shape_768_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 1024 --shape_col 1024 --plot_mode True --output shape_1024_.txt >> $log_file
python gemm_int8_of16.py --shape_row 1280 --shape_col 1280 --plot_mode True --output shape_1280_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 1664 --shape_col 1664 --plot_mode True --output shape_1664_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 1792 --shape_col 1792 --plot_mode True --output shape_1792_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 1920 --shape_col 1920 --plot_mode True --output shape_1920_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 2048 --shape_col 2048 --plot_mode True --output shape_2048_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 2432 --shape_col 2432 --plot_mode True --output shape_2432_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 2560 --shape_col 2560 --plot_mode True --output shape_2560_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 2816 --shape_col 2816 --plot_mode True --output shape_2816_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 3072 --shape_col 3072 --plot_mode True --output shape_3072_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 3328 --shape_col 3328 --plot_mode True --output shape_3328_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 3712 --shape_col 3712 --plot_mode True --output shape_3712_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 3840 --shape_col 3840 --plot_mode True --output shape_3840_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 3968 --shape_col 3968 --plot_mode True --output shape_3968_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 4096 --shape_col 4096 --plot_mode True --output shape_4096_int8_info.txt >> $log_file
python gemm_int8_of16.py --shape_row 8192 --shape_col 8192 --plot_mode True --output shape_8192_int8_info.txt >> $log_file

