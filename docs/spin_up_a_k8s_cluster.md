# Spin up a K8s Cluster
> This is a draft!

Mainly follow the official [documentation](https://v1-31.docs.kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/).


## Steps
1. `cd` into `pkgs`, extract all Debian packages, and use `dpkg -i` to install them.

2. Run the following command to disable swap:
```bash
swapoff -a

# swapon shows nothing
swapon --show
```

3. Install a CRI-compatible container runtime (take `contaienrd` as an example):
```bash
...
```

We assume that `contaierd` has been installed. Then, run the following command to check the service:
```bash
systemctl status containerd
```

4. Configure cgroup drivers
Check if `systemd` is the selected init system:
```
ps -p 1
```

Then, set `systemd` for both `containerd` and `kubelet`. Let's handle `containerd` as follows:
```
mkdir -p /etc/containerd
containerd config default | sed 's/SystemCgroup = false/SystemCgroup = true/' | tee /etc/containerd/config.toml
systemctl restart containerd
```
You can also see this [guide](https://v1-31.docs.kubernetes.io/docs/setup/production-environment/container-runtimes/#containerd-systemd).

As shown [here](https://v1-31.docs.kubernetes.io/docs/tasks/administer-cluster/kubeadm/configure-cgroup-driver/#configuring-the-kubelet-cgroup-driver):

> **Note**: In v1.22 and later, if the user does not set the `cgroupDriver` field under `KubeletConfiguration`, kubeadm defaults it to `systemd`.

Please skip the cgroup driver setup for `kubelet` now.

5. Run the following command to initialize a control plane node:
```bash
kubeadm init --apiserver-advertise-address <ip-address> --pod-network-cidr "10.244.0.0/16" --upload-certs
```

6. Configure CNI


## Issues
### Can't Pull Container Images from Public Networks
This is the most tricky part when we run on-premise deployment.

1. Pre-pull and pack the required control plane images (please refer to [here](https://kubernetes.io/docs/reference/setup-tools/kubeadm/kubeadm-init/#without-internet-connection)):
```bash
kubeadm config images list
kubeadm config images pull
```

You can prepare all images wherever you have an Internet connection.

> We use `docker pull` to pull all images from the private Docker harbor in our company.

2. Transfer the image pack to the target environment without an Internet connection.

(Even trickier here) For us, we tag all pulled images by `docker image tag <hash> <name>` based on the output names of `kubeadm config images list`, tar them with `docker save -o <image-name>.tar <n>:<v>`, and import them using `ctr -n k8s.io image import <image-name>.tar`.

Use `ctr -n k8s.io images ls` or `crictl images` to check imported images. Finally, `kubeadm init` can successfully find those images.

### Fail to Start `pause`
Run `journalctl -u containerd -f` for troubleshooting. It turns out to be an image version mismatch. Hence, we modify `sandbox_image` field in `/etc/containerd/config.toml` to `registry.k8s.io/pause:3.10`.

### Specific Ports Are Occupied or Configuration Files Already Exist
```
kubeadm reset
```

### Fail to Find `/proc/sys/net/bridge/...iptables`
```
modprobe br_netfilter
```

## Plus
* Use K9s for monitoring