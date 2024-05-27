#!/bin/bash

# Array of GPU indices
gpus=(0 1 2 3 4 5 6 7)
#!/bin/bash

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs found: $num_gpus"

gpus=($(seq 0 $((num_gpus - 1))))
scripts=(
    "python eval_model.py --model_name model_fake_fake_task_1-fuse_config_8192_0.0_512_0.001_1_1024_1_1-hidden_channels_256-layer_type_SAGEConv-learned_embedding_dim_256-lr_0.001-num_layers_3-pretrained_method_3-sample_0-train_all_emd_0_4"
    "python eval_model.py --model_name model_fake_fake_task_1-fuse_config_8192_0.0_512_0.001_1_256_0_0-hidden_channels_256-layer_type_SAGEConv-learned_embedding_dim_256-lr_0.001-num_layers_3-pretrained_method_2-sample_0-train_all_emd_0_4"
    "python eval_model.py --model_name model_fake_fake_task_1-fuse_config_8192_0.0_512_0.001_1_1024_1_1-hidden_channels_256-layer_type_SAGEConv-learned_embedding_dim_256-lr_0.001-num_layers_3-pretrained_method_0-sample_0-train_all_emd_1_4"
    "python eval_model.py --model_name model_fake_fake_task_1-fuse_config_8192_0.0_512_0.001_1_256_1_1-hidden_channels_256-layer_type_SAGEConv-learned_embedding_dim_256-lr_0.001-num_layers_3-pretrained_method_2-sample_0-train_all_emd_0_4"
    "python eval_model.py --model_name model_fake_fake_task_1-fuse_config_8192_0.0_512_0.001_1_1024_1_1-hidden_channels_256-layer_type_SAGEConv-learned_embedding_dim_256-lr_0.001-num_layers_3-pretrained_method_0-sample_0-train_all_emd_0_4"
    "python eval_model.py --model_name model_fake_fake_task_1-fuse_config_8192_0.0_512_0.001_1_1024_1_1-hidden_channels_256-layer_type_SAGEConv-learned_embedding_dim_256-lr_0.001-num_layers_3-pretrained_method_1-sample_0-train_all_emd_0_4"
    "python eval_model.py --model_name model_fake_fake_task_1-fuse_config_8192_0.0_512_0.001_1_1024_1_0-hidden_channels_256-layer_type_SAGEConv-learned_embedding_dim_256-lr_0.001-num_layers_3-pretrained_method_2-sample_0-train_all_emd_0_4"
)

for i in ${!scripts[@]}; do
    gpu_index=$((i % num_gpus))
    CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} ${scripts[$i]} &
    # If all GPUs are in use, wait for any to finish before continuing
    if (( (i + 1) % num_gpus == 0 )); then
        wait
    fi
done
wait

