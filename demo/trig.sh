export CUDA_VISIBLE_DEVICES=2

python -m demo.text_to_image_sdxl \
    --num_step 1 \
    --precision float16 \
    --conditioning_timestep 399