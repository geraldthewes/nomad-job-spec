FROM postgres:15-alpine

# PostgreSQL runs as user postgres (uid 999)
ENV POSTGRES_USER=myapp
ENV POSTGRES_DB=myapp_production

# Default port
EXPOSE 5432

# Data directory
VOLUME ["/var/lib/postgresql/data"]
