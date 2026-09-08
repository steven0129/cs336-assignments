FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
          /etc/apt/sources.list.d/nvidia*.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl && \
    curl -fL \
      "https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_4/NsightSystems-linux-cli-public-2026.4.1.191-3860507.deb" \
      -o /tmp/nsight-systems-cli.deb && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      /tmp/nsight-systems-cli.deb && \
    rm -f /tmp/nsight-systems-cli.deb && \
    rm -rf /var/lib/apt/lists/*