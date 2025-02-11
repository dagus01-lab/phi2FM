# Define the base image and tag
IMAGE_NAME := openvino/ubuntu18_dev
IMAGE_TAG := 2020.3
FULL_IMAGE_NAME := $(IMAGE_NAME):$(IMAGE_TAG)

MOUNT := /home/nvidia/Desktop/carloscollado/mount_folder
MYRIAD_UDEV_RULES := /home/nvidia/Desktop/carloscollado/output_folder/97-myriad-usbboot.rules


# Pull the Docker image
pull:
	docker pull $(FULL_IMAGE_NAME)

# Run the Docker container
run:
	docker run --platform linux/amd64 -it --rm \
		--privileged \
		--device-cgroup-rule='c 189:* rmw' \
		-v /dev/bus/usb:/dev/bus/usb \
		-v $(MOUNT):/home/mount/ \
		-u root \
		-w /home/mount/ \
		$(FULL_IMAGE_NAME) /bin/bash

