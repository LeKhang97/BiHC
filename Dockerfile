# BiHC in a container. STRIDE and Voronota are not part of the repository: the first stage downloads and compiles
# them (STRIDE is free for academic use only, see README), the second holds Python, BiHC and the two executables.
#   docker build -t bihc .
#   docker run --rm -v "$PWD":/data bihc -i examples/1a45.pdb -o out
FROM debian:bookworm-slim AS programs
RUN apt-get update && apt-get install -y --no-install-recommends g++ make curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ARG VORONOTA_VERSION=1.29.4602
ARG STRIDE_COMMIT=867a5eb0f2479cb16615512a53ee472c54649505
WORKDIR /build
RUN curl -fsSL https://github.com/kliment-olechnovic/voronota/releases/download/v${VORONOTA_VERSION}/voronota_${VORONOTA_VERSION}.tar.gz | tar xz \
 && cd voronota_${VORONOTA_VERSION} && g++ -O3 -std=c++11 -o /build/voronota $(find ./src/ -name '*.cpp')
RUN curl -fsSL https://github.com/MDAnalysis/stride/archive/${STRIDE_COMMIT}.tar.gz | tar xz \
 && make -C stride-${STRIDE_COMMIT}/src && cp stride-${STRIDE_COMMIT}/src/stride /build/stride

FROM python:3.11-slim-bookworm
COPY --from=programs /build/voronota /build/stride /usr/local/bin/
WORKDIR /opt/bihc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest
COPY . .

WORKDIR /data
ENTRYPOINT ["python", "/opt/bihc/BiHC.py"]
