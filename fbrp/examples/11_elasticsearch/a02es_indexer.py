"""Demo AlephZero logs -> ElasticSearch."""

import a0
import elasticsearch
import json
import signal


class A02ES_Indexer:
    def __init__(self):
        # Connect to the local elasticsearch engine.
        self._es = elasticsearch.Elasticsearch(
            "http://localhost:9200", request_timeout=10
        )
        # Connect to the alephzero logger.
        self._a0 = a0.Subscriber("log/announce", self.on_log_announce)

    def on_log_announce(self, pkt):
        # AlephZero logger made an announcement.
        # We only care if that announcement is that a log file has been completed.
        info = json.loads(pkt.payload)
        if info["action"] != "closed":
            return

        # Grab path data from the announcement.
        abspath = info["write_abspath"]
        relpath = info["write_relpath"]
        original_path = info["read_relpath"]

        # Per-packet callback.
        def read_handle(tlk, fpkt):
            try:
                # Index the fields of interest.
                # TODO(lshamis): Add header flags to index user desired values, such as mime.
                # TODO(lshamis): Can we batch the operation across multiple packets?
                self._es.index(
                    index="myindex",
                    document={
                        "id": fpkt.id,
                        "abspath": abspath,
                        "relpath": relpath,
                        "original_path": original_path,
                        "offset": tlk.frame().off,
                        "walltime": dict(fpkt.headers).get("a0_time_wall"),
                        "deps": [v for k, v in fpkt.headers if k == a0.DEP],
                    },
                )
            except ConnectionError as err:
                # TODO(lshamis): Maybe retry.
                print(f"skipping pkt: {err}")

        # Iterate through each packet of the closed log file.
        fileopts = a0.File.Options.DEFAULT
        fileopts.open_options.arena_mode = a0.Arena.Mode.READONLY
        r = a0.ReaderSyncZeroCopy(
            a0.File(info["write_abspath"], fileopts), a0.INIT_OLDEST
        )
        while r.can_read():
            r.read(read_handle)


def main():
    indexer = A02ES_Indexer()
    signal.pause()


if __name__ == "__main__":
    main()
