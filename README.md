# 3D Gaussian Splatting on Graphcore IPUs

> **Current stable branch: `cleanup`** — this is the fixed, working branch. Use it as the basis for any new work.
> Latest commit: `fe0ef6f` (flip scene option, 2026-04-14).

Experimental implementation of an alternatice to neural radiance fields: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting) B. Kerbl and G. Kopanas, T. Leimk{\"u}hler and G. Drettakis, ACM Transactions on Graphics, July 2023.

## Which bits are a good fit for IPU?:
- Should be able to perform efficient Gaussian transformations and splats using matrix and convolution unit.
- Tile based renderer is IPU friendly (each tile of framebuffer can stay pinned in SRAM of each IPU tile).
- Can hold millions of gaussians in SRAM.
  - i.e. scene and framebuffer can all stay in SRAM.

## Which bits will be difficult?:
- After transformation and sorting Gaussians need to be dynamically moved to the destination tile(s) for splatting.
- Load imbalances: depending on scene and viewpoint the number of Gaussians splatted per tile could vary significantly.
 - I.e. any one gaussian might need to be splatted on 0 or all tiles!
- We will try to solve this using experimental dynamic/jitted exchange code.

The implementation is highly experimental work in progress. Currently implementation is as follows:

- Point rasterisation on CPU.
- Remote viewer.
- Ability to switch render device between CPU/IPU in remote viewer.
- Currently when IPU is selected as the render device it only transforms the points using the AMP unit: they are then splatted on the CPU.

### License

Unless stated otherwise all the content in this repository is available under [this license](./LICENSE). Submodules are all licensed under their own terms: please check the [relevant subfolder](/external).

## Instructions

The demo consists of two parts: a cloud based render server and a local remote viewer. The instructions below are a quick start guide to build and run these components. These instructions are tested on Graphcloud but should work on other cloud services where IPUs are available and you are free to tunnel/forward ports from the container.

### Build a Docker image for Poplar SDK 3.3.0

Setup will be much easier if you build and launch the tested docker image:

```
git clone https://github.com/markp-gc/docker-files.git

export CONTAINER_SSH_PORT=2023

docker build -t $USER/poplar_3.3_dev --build-arg UNAME=$USER  \
--build-arg UID=$(id -u) --build-arg GID=$(id -g) \
--build-arg CUSTOM_SSH_PORT=$CONTAINER_SSH_PORT  \
docker-files/graphcore/poplar_dev
```

### Launch the container

Source your Poplar SDK so that you can use gc-docker command to launch your container (within the container you will be using its SDK which is fixed at 3.3.0 to ensure compatibility if someone reconfigures the shared host system). The SDK should be pre-installed in /opt but check the path for your system:

```
source /opt/gc/poplar_sdk-ubuntu_20_04-3.2.0+1277-7cd8ade3cd/enable
gc-docker -- --detach -it --rm --name "$USER"_docker -v/nethome/$USER:/home/$USER --tmpfs /tmp/exec $USER/poplar_3.3_dev
```

You should now see the running container listed when you run the command docker ps and you should be able to attach to it to get a shell in the container: `docker attach "$USER"_docker`

:warning: Note that the gc-docker command above has mounted your home directory as the home directory in your container. This could break your home in the base system in theory (but in practice the convenience outweighs the small risk). Just be aware that changes to your home directory in the container will be reflected in the base home directory (which might have a different Ubuntu version e.g. 18 instead of 22).

### Clone and build the splatting render server

```
git clone --recursive https://github.com/graphcore-research/gaussian_splat_ipu.git
mkdir gaussian_splat_ipu/build
cd gaussian_splat_ipu/build/
cmake -GNinja ..
 ninja -j100
```

This should configure and build successfully as the container already has everything you need installed.

### Build the remote user interface application

The remote-UI runs on your local laptop or workstation. (The setup here will be less straightforward than for the server depending on your machine’s configuration). There is a list of dependencies in that repo's [README](https://github.com/markp-gc/remote_render_ui#dependencies)

It has been tested on Ubuntu 18 and Mac OSX 11.6 it should be possible to build on other systems if you can satisfy the dependencies.

Clone the repo (we need a specific branch that is compatible with the splatting server):

```
git clone --recursive --branch splat_viewer git@github.com:markp-gc/remote_render_ui.git
mkdir -p remote_render_ui/build
cd remote_render_ui/build/
cmake -GNinja ..
ninja -j100
```

Unless your machine happened to be configured perfectly already then this will probably not configure or build first time so please carefully review all the dependencies, especially for the videolib submodule as the FFMPEG version must be 4.4: [videolib instructions](https://github.com/markp-gc/videolib#installing-dependencies).

### Create a tunnel into your container

The most reliable way to achieve this is to use VS Code to forward the port for you. Setup VS code for remote development in your container as follows:
- Install VS Code on your local laptop/workstation: [get VS Code](https://code.visualstudio.com)
- Install the remote-development pack (follow this link and click install) [get the remote development extension pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
- Connect VS Code to your cloud server via SSH following the instructions here: [connect VS code via SSH](https://code.visualstudio.com/docs/remote/ssh)

The VS Code remote-development package supports development in containers: once you are connected to the remote machine via SSH you can see and attach to running containers by clicking the Remote Explorer icon in the left panel, then select Containers from the drop down list at the top. You should be able to see the container that you created is in the list and in the running state. Click the arrow icon to attach to your container in the current window:

![image](https://github.com/markp-gc/ipu_path_trace/assets/65598182/d7813be1-d72a-4482-ba51-b338a77276c8)

Once you are in the container find the ports panel and forward a port number of your choice (> 1024):

![image](https://github.com/markp-gc/ipu_path_trace/assets/65598182/7405e3bc-5f39-4775-9b12-f1fdf7f031df)

### Run the demo

On the cloud machine in your docker container launch the server:

```
./src/main/splat --input ../data/horse_statue_01.xyz --ui-port 5000
```

On your local laptop or workstation launch the remote user interface:

```
./remote-ui -w 1600 -h 1100 --port 5000
```

:warning: The above commands assume that you set up your SSH tunnel so that localhost is directed into the container, and that the locally forwarded port is the same as the remote port (VS Code might choose another local port if the one you asked for was in use).

The user interface should launch and connect to render the server. You should be able to rotate the poijnt cloud using the control wheel in the top right of the control panel and zoom using the FOV slider. You can also switch the accelerator device using the drop down box.

