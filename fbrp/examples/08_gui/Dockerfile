FROM ubuntu:focal

RUN apt update && \
    apt install -y \
        x11-apps && \
    rm -rf /var/lib/apt/lists/*

CMD ["xclock"]