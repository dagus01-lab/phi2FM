echo "Running training with config: args/newgpu.yml"

torchrun --nproc_per_node=1 training_script.py -r args/newgpu.yml 2>&1 | tee nano_nosqrt.log