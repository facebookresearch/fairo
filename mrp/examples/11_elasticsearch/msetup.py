import mrp
import os

class ElasticSearchConst:
    DOCKER_IMAGE = "elasticsearch:8.0.0"
    DOCKER_DASHBOARD_IMAGE = "kibana:8.0.0"
    CONTAINER_DATA_PATH = "/usr/share/elasticsearch/data"
    HOST_DATA_PATH = os.path.abspath("./es_data")

class OpenSearchConst:
    DOCKER_IMAGE = "opensearchproject/opensearch:1.3.2"
    DOCKER_DASHBOARD_IMAGE = "opensearchproject/opensearch-dashboards:1.3.2"
    CONTAINER_DATA_PATH = "/usr/share/opensearch/data"
    HOST_DATA_PATH = os.path.abspath("./os_data")

DB = OpenSearchConst

A0_LOG_PATH = os.path.abspath("./a0_data")

if not os.path.exists(A0_LOG_PATH):
    os.makedirs(A0_LOG_PATH)

if not os.path.exists(DB.HOST_DATA_PATH):
    os.makedirs(DB.HOST_DATA_PATH)

# Local ElasticSearch database.
# Saves data in ./es_data
# http://0.0.0.0:9200
mrp.process(
    name="database",
    runtime=mrp.Docker(
        image=DB.DOCKER_IMAGE,
        mount=[f"{DB.HOST_DATA_PATH}:{DB.CONTAINER_DATA_PATH}"],
    ),
    env={
        "discovery.type": "single-node",
        # "xpack.security.enabled": "false",
        "plugins.security.disabled": "true",
    },
)

# Kibana visualizes the ElasticSearch database.
# http://0.0.0.0:5601
mrp.process(
    name="dashboard",
    runtime=mrp.Docker(image=DB.DOCKER_DASHBOARD_IMAGE),
    env={
        # "plugins.security.disabled": "true",
        "OPENSEARCH_HOSTS": '["http://localhost:9200"]',
        "DISABLE_SECURITY_DASHBOARDS_PLUGIN": "true",
    },
)

# Simple process that generates data.
mrp.process(
    name="datagen",
    runtime=mrp.Conda(
        dependencies=["python=3.8", {"pip": ["alephzero"]}],
        run_command=["python3", "datagen.py"],
    ),
)

# AlephZero logger process.
# Saves data from "datagen" into ./a0_data
# Files are generated every 5s for demo purposes only.
mrp.process(
    name="log",
    runtime=mrp.Docker(
        image="ghcr.io/alephzero/log:latest",
        mount=[f"{A0_LOG_PATH}:{A0_LOG_PATH}"],
    ),
    cfg={
        "savepath": A0_LOG_PATH,
        "default_max_logfile_duration": "5s",
        "rules": [
            {
                "protocol": "pubsub",
                "topic": "**/*",
                "policies": [{"type": "save_all"}],
            },
        ],
    },
)

# Converts completed AlephZero logfiles into the ElasticSearch database.
mrp.process(
    name="a02es_indexer",
    runtime=mrp.Conda(
        dependencies=[
            "python=3.8",
            {
                "pip": [
                    "alephzero",
                    "elasticsearch",
                    "opensearch-py",
                ],
            },
        ],
        run_command=["python3", "a02es_indexer.py"],
    ),
)

mrp.main()
