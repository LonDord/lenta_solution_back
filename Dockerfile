FROM amazoncorretto:11-alpine

RUN apk add --no-cache curl

WORKDIR /app

COPY build/libs/lenta_solution_back-1.0-SNAPSHOT.jar app.jar

RUN addgroup -S appuser && adduser -S appuser -G appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 4444

ENTRYPOINT ["java", "-jar", "app.jar"]