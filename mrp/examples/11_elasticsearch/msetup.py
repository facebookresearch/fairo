import mrp
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
mrp.process(
    name="elasticsearch",
    runtime=mrp.Docker(
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
mrp.process(
    name="kibana",
    runtime=mrp.Docker(image="kibana:8.0.0"),
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
        dependencies=["python=3.8", {"pip": ["alephzero", "elasticsearch"]}],
        run_command=["python3", "a02es_indexer.py"],
    ),
)

if __name__ == "__main__":
    mrp.main()
