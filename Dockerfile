FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Set working directory and copy app files
WORKDIR /app
COPY . .

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install Python dependencies with uv tool
RUN uv pip install --system -r requirements.txt

CMD ["sh", "-c", "python3 mcp_server.py & exec python3 api_server.py"]
