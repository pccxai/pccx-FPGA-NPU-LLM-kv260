"""pccx v002 NPU runtime — user-space Python bindings for KV260 deployment.

Modules:
- isa     : 64-bit high-level command encoders (LOAD_WEIGHT/LOAD_PROMPT/NEXT_TOKEN)
- uio     : /dev/uioN mmap helpers (write ADDR_INST, push KICK, poll status)
- npu     : higher-level submit-and-wait orchestration around isa + uio
"""
