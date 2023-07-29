FROM mambaorg/micromamba:1.4.9

USER root
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN apt-get update && \
    apt-get install -y git gcc g++

RUN mkdir -p /opt/app /input /output /opt/app/ground-truth \
    && chown user:user /opt/app /input /output /opt/app/ground-truth

RUN mkdir -p /opt/conda/pkgs && chown user:user /opt/conda/pkgs
RUN touch /opt/conda/pkgs/urls.txt && chown user:user /opt/conda/pkgs/urls.txt

USER user

COPY --chown=user:user challenge_eval /opt/app/challenge_eval
COPY --chown=user:user grand_challenge /opt/app/grand_challenge

RUN micromamba install -y -n base -f /opt/app/grand_challenge/environment.yml && \
    micromamba clean --all --yes

ARG GROUND_TRUTH_PATH
COPY --chown=user:user ${GROUND_TRUTH_PATH} /opt/app/ground-truth/test-labels.p

WORKDIR /opt/app

ENTRYPOINT [ "/usr/local/bin/_entrypoint.sh", "python", "-m", "grand_challenge.evaluation" ]
