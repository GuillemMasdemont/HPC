# 1. Start with an image that ALREADY has Python 3.9
FROM python:3.9-slim

# 2. Install ONLY git (Python and Pip are already there!)
RUN apt-get update && apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. Install pandas
RUN pip install --no-cache-dir jupyter pandas matplotlib

# 4. Clone your code
WORKDIR /opt
RUN git clone https://github.com/demsarjure/hpc_hello_world.git

# 5. Set the starting point (The "CMD" we talked about!)
# Assuming the script is inside the cloned folder:
CMD ["python", "/opt/hpc_hello_world/hpc_hello_world.py"]