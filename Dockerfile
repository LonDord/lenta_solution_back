FROM openjdk:11-jre-slim

WORKDIR /app

COPY build/libs/lenta_solution_back-*.jar app.jar

RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 4444

ENTRYPOINT ["java", "-jar", "app.jar"]