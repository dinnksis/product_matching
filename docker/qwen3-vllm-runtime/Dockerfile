FROM vllm/vllm-openai:v0.14.0

# The competition appends metadata.entry_point as the container command.
# Replace vLLM's inherited ["vllm", "serve"] entrypoint. The competition
# appends `bash run.sh ...`, producing `/usr/bin/env bash run.sh ...`.
ENTRYPOINT ["/usr/bin/env"]
CMD ["bash"]
