#!/usr/bin/env bash

CLUSTER_ID=$1
gcloud container clusters get-credentials $CLUSTER_ID

CLUSTER_NAME=`kubectl config get-clusters | grep $CLUSTER_ID`

kubectl config set-context $CLUSTER_NAME
kubectl get nodes
