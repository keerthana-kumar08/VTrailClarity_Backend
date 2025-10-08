import multiprocessing

wsgi_app = "deploy.main:app" # Or your app's location

bind = "0.0.0.0:9001"

# bind = "unix:/run/gunicorn.sock"

# backlog = 2048 # Max pending connections

# Worker Processes

workers = (multiprocessing.cpu_count() * 2) + 1

worker_class = "uvicorn.workers.UvicornWorker"

preload_app = True

# threads = 1 # Usually 1 for UvicornWorker

# Security

# user = "your_app_user" # Drop privileges if running as root initially

# group = "your_app_group"

# umask = 007

# Logging

loglevel = "info"

accesslog = "-" # Log to stdout

errorlog = "-"  # Log to stderr

# access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process Naming

proc_name = "my_fastapi_app"

# Server Mechanics

keepalive = 75 # Sync with Nginx keepalive_timeout

timeout = 200 # Worker timeout

graceful_timeout = 30 # Timeout for graceful shutdown

# worker_tmp_dir = "/dev/shm" # Use tmpfs for worker temp files if available

# Reloading (for development, disable in production)

# reload = False

