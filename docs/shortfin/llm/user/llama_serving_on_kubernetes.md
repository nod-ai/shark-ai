# Llama 8b GPU instructions on Kubernetes

## Setup

We will use an example with `llama_8b_f16` in order to describe the
process of exporting a model and deploying four instances of a shortfin llm server
behind a load balancer on MI300X GPU.

### Pre-Requisites

- Kubernetes cluster available to use
- kubectl installed on system and configured for cluster of interest
    - To install kubectl, please check out [kubectl install](https://kubernetes.io/docs/tasks/tools/#kubectl)
    and make sure to set the `KUBECONFIG` environment variable to point to your kube config file to authorize
    connection to the cluster.

### Deploy shortfin llama app service

To generate the artifacts required for this k8s deployment, please follow [llama_serving.md](./llama_serving.md) until you have have all of the files that we need to run the shortfin LLM server.
Please upload your artifacts to a storage option that you can pull from in your k8s cluster (NFS, S3, CSP).
Save [llama-app-deployment.yaml](../../../../shortfin/deployment/shortfin_apps/llm/k8s/llama-app-deployment.yaml) locally and edit it to include your artifacts you just stored and change flags to intended configuration.

To deploy llama app:

```
kubectl apply -f llama-app-deployment.yaml
```

To retrieve external IP for targetting the llama app load balancer:

```
kubectl get service amdshark-llama-app-service
```

Now, you can use the external IP for sglang integration or just sending text generation requests.

### Delete shortfin llama app service

After done using, make sure to delete:

```
kubectl delete deployment amdshark-llama-app-deployment
kubectl delete service amdshark-llama-app-service
```
