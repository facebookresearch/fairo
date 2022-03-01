import fbrp
import os

A0_LOG_PATH = os.path.abspath("./a0_data")
ES_DATA_PATH = os.path.abspath("./es_data")

if not os.path.exists(A0_LOG_PATH):
    os.makedirs(A0_LOG_PATH)

if not os.path.exists(ES_DATA_PATH):
    os.makedirs(ES_DATA_PATH)

# Local ElasticSearch database.
# Saves data in ./es_data
# http://0.0.0.0:9200
fbrp.process(
    name="elasticsearch",
    runtime=fbrp.Docker(
        image="elasticsearch:8.0.0",
        mount=[f"{ES_DATA_PATH}:/usr/share/elasticsearch/data"],
    ),
    env={
        "discovery.type": "single-node",
        "xpack.security.enabled": "false",
    },
)

# Kibana visualizes the ElasticSearch database.
# http://0.0.0.0:5601
fbrp.process(
    name="kibana",
    runtime=fbrp.Docker(image="kibana:8.0.0"),
)

# Simple process that generates data.
fbrp.process(
    name="datagen",
    runtime=fbrp.Conda(
        dependencies=["python=3.8", {"pip": ["alephzero"]}],
        run_command=["python3", "datagen.py"],
    ),
)

# AlephZero logger process.
# Saves data from "datagen" into ./a0_data
# Files are generated every 5s for demo purposes only.
fbrp.process(
    name="log",
    runtime=fbrp.Docker(
        image="ghcr.io/alephzero/log:latest",
        mount=[f"{A0_LOG_PATH}:{A0_LOG_PATH}"],
    ),
    cfg={
        "savepath": A0_LOG_PATH,
        "default_max_logfile_duration": "5s",
        "rules": [
            {
                "protocol": "pubsub",
                "topic": "data",
                "policies": [{"type": "save_all"}],
            },
        ],
    },
)

# Converts completed AlephZero logfiles into the ElasticSearch database.
fbrp.process(
    name="a02es_indexer",
    runtime=fbrp.Conda(
        dependencies=["python=3.8", {"pip": ["alephzero", "elasticsearch"]}],
        run_command=["python3", "a02es_indexer.py"],
    ),
)

fbrp.main()
