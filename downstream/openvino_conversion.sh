sudo podman run -it --rm \
    --privileged \
    --user root \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --device-cgroup-rule='c 189:* rmw' \
    --platform linux/amd64 \
    -v /home/gdaga/phi2FM/downstream/onnx:/home/mnt \
    -w /home/mnt \
    openvino/ubuntu18_dev:2020.3 \
    python3 /opt/intel/openvino_2020.3.194/deployment_tools/model_optimizer/mo.py \
    --input_model phisat2net_geoaware_best_model.onnx \
    --data_type FP16 \
    --input_shape "[1,8,224,224]" \
    --progress \
    --stream_output \
    --output_dir /home/mnt


# sudo podman run -it --rm \
#     --privileged \
#     --user root \
#     --security-opt seccomp=unconfined \
#     --security-opt apparmor=unconfined \
#     --device-cgroup-rule='c 189:* rmw' \
#     --platform linux/amd64 \
#     -v /home/gdaga/phi2FM/downstream/onnx:/home/mnt \
#     -w /home/mnt \
#     openvino/ubuntu18_dev:2020.3 \
#     python3 /opt/intel/openvino_2020.3.194/deployment_tools/model_optimizer/mo.py \
#     --input_model phisat2net_geoaware_best_encoder.onnx \
#     --data_type FP16 \
#     --input_shape "[1,8,224,224]" \
#     --progress \
#     --stream_output \
#     --output_dir /home/mnt

# dummy_bottom_feats = torch.randn(1, 640, 14, 14)  # Adjust size based on your encoder output
# # skips: skip connections from encoder at different resolutions
# dummy_skip4 = torch.randn(1, 640, 28, 28)
# dummy_skip3 = torch.randn(1, 320, 56, 56)
# dummy_skip2 = torch.randn(1, 160, 112, 112) 
# dummy_skip1 = torch.randn(1, 80, 224, 224)

# # Convert decoder
# sudo podman run -it --rm \
#     --privileged \
#     --user root \
#     --security-opt seccomp=unconfined \
#     --security-opt apparmor=unconfined \
#     --device-cgroup-rule='c 189:* rmw' \
#     --platform linux/amd64 \
#     -v /home/gdaga/phi2FM/downstream/onnx:/home/mnt \
#     -w /home/mnt \
#     openvino/ubuntu18_dev:2020.3 \
#     python3 /opt/intel/openvino_2020.3.194/deployment_tools/model_optimizer/mo.py \
#     --input_model phisat2net_geoaware_best_decoder.onnx \
#     --data_type FP16 \
#     --input_shape "[1, 640, 14, 14],[1,80,224,224],[1, 160, 112, 112],[1, 320, 56, 56],[1, 640, 28, 28]" \
#     --input "bottom_feats","skip4","skip3","skip2","skip1" \
#     --progress \
#     --stream_output \
#     --output_dir /home/mnt