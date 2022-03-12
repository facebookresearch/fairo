"""Demo AlephZero logs -> OpenSearch."""

import a0
import base64
import json
import opensearchpy
import os
import signal


class A02OS_Indexer:
    def __init__(self):
        # Connect to the local opensearch engine.
        self._os = opensearchpy.OpenSearch(
            request_timeout=10,

            hosts = [{'host': os.environ["host"], 'port': int(os.environ["port"])}],
            http_compress = True, # enables gzip compression for request bodies
            http_auth = json.loads(os.environ["auth"]),
            # client_cert = client_cert_path,
            # client_key = client_key_path,
            use_ssl = True,
            verify_certs = True,
            ssl_assert_hostname = False,
            ssl_show_warn = False,

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
            # Add standard fields.
            data = {
                "id": fpkt.id,
                "abspath": abspath,
                "relpath": relpath,
                "original_path": original_path,
                "payload": base64.b64encode(fpkt.payload_view).decode(),
            }

            # Add all headers.
            # TODO(lshamis): Add directive headers like "_index.payload".
            for k, v in fpkt.headers:
                data.setdefault(k, []).append(v)

            try:
                # Index the data into OS.
                # TODO(lshamis): Can we batch the operation across multiple packets?
                self._os.index(index="myindex", body=data)
            except Exception as err:
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
    indexer = A02OS_Indexer()
    signal.pause()


if __name__ == "__main__":
    main()
