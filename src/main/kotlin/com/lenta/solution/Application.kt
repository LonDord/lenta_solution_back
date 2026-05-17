package com.lenta.solution

import com.lenta.solution.routes.configureRoutes
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.serialization.jackson.*

fun main() {
    val port = System.getenv("SERVER_PORT")?.toIntOrNull() ?: 4444
    val host = System.getenv("SERVER_HOST") ?: "192.168.0.47"

    embeddedServer(
        Netty,
        port = port,
        host = host,
        module = Application::module
    ).start(wait = true)
}

fun Application.module() {
    install(ContentNegotiation) {
        jackson()
    }

    configureRoutes()
}