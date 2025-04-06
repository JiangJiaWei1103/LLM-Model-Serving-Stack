# LLM-Model-Serving-Stack

> Planned software stack: K8s (v1.31), NginX, LiteLLM, SGLang, Ollama, infinity, TEI, EFK, nvitop, Prometheus, Grafana...

Deploy a multi-tenancy LLM model serving stack in an air-gapped environment within minutes, but still requiring some effort for now...

* K8s suite (under `./pkgs`): `conntrack`, `cri-tools`, `ethtool`, `kubernetes-cni`, `kubeadm`, `kubelet`, `kubectl`


## An Overly Simple Arch
![](https://github.com/JiangJiaWei1103/LLM-Model-Serving-Stack/tree/main/assets/model_serving_arch.png)


## Todos
* [ ] Modularize and clean up the ugly `nginx.conf` for readability and maintainability
