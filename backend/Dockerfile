# Stage 1: Build Tools and Dependencies
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
# Install dependencies into a local directory
RUN pip install --user -r requirements.txt

# Stage 2: The Runtime Environment
FROM python:3.11-slim AS runtime
WORKDIR /app
# Only copy the installed packages from the builder stage
COPY --from=builder /root/.local /root/.local
# Copy your actual application code
COPY . .

# Update path to include the installed packages
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

# Listen on all network interfaces (0.0.0.0) so traffic can reach the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]