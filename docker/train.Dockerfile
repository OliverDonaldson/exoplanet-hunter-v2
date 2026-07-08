# GPU-burst training image. Run on a rented GPU only when the validation gate
# says the dataset changed materially; never on an always-on instance.
# Build from the repository root:
#   docker build -f docker/train.Dockerfile -t exoplanet-hunter-train .
FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /srv
COPY pipeline/ pipeline/
RUN pip install --no-cache-dir ./pipeline

# The training entry point arrives with feat/tfdata-pipeline
# (tf.data + TFRecord streaming from R2 + mixed_float16).
CMD ["python", "-c", "import exoplanet_hunter; print('training entry point lands in feat/tfdata-pipeline')"]
