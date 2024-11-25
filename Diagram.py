from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.vcs import Github
from diagrams.onprem.ci import Jenkins
from diagrams.onprem.mlops import Mlflow
from diagrams.aws.storage import S3
from diagrams.onprem.container import Docker
from diagrams.k8s.compute import Pod
from diagrams.onprem.monitoring import Prometheus, Grafana
from diagrams.programming.framework import Flask
from diagrams.generic.blank import Blank

with Diagram("ML Infra Serving Framework", filename="ml_infra_serving_framework", show=False):
    # Repositories
    github = Github("GitHub\n(Model Code + YAML Configs)")
    mlflow = Mlflow("MLFlow\n(Model Weights)")
    s3 = S3("S3 Storage")
    mlflow >> Edge(label="Stores weights in") >> s3

    # CICD Pipeline
    jenkins = Jenkins("Jenkins\n(CICD Pipeline)")

    # Build Process
    mlinfra_lib = Flask("MLInfra Library\n(Shared Code)")
    docker_image = Docker("Docker Image\n(Model + MLInfra Lib)")
    docker_image - Edge(label="Includes") - mlinfra_lib

    # Environments
    with Cluster("Kubernetes Cluster"):
        with Cluster("Dev Environment"):
            dev_pod = Pod("Dev Pod")
        with Cluster("Test Environment"):
            test_pod = Pod("Test Pod")
        with Cluster("Prod Environment"):
            prod_pod = Pod("Prod Pod")

    # Monitoring
    prometheus = Prometheus("Prometheus")
    grafana = Grafana("Grafana")
    prometheus >> Edge(label="6. Feeds into") >> grafana

    # Deployment Steps
    github >> Edge(label="1. Pull code & configs") >> jenkins
    jenkins >> Edge(label="2. Retrieve weights from") >> mlflow
    jenkins >> Edge(label="3. Build image with") >> docker_image
    jenkins >> Edge(label="4. Deploy to K8s") >> [dev_pod, test_pod, prod_pod]
    [dev_pod, test_pod, prod_pod] >> Edge(label="5. Send metrics to") >> prometheus
