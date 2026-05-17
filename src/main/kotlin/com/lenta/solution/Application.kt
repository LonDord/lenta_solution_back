package com.lenta.solution

import com.lenta.solution.routes.configureRoutes
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.serialization.jackson.*

fun main() {
    embeddedServer(
        Netty,
        port = 4444,
        host = "192.168.0.47",
        module = Application::module
    ).start(wait = true)
}

fun Application.module() {
    install(ContentNegotiation) {
        jackson()
    }

    configureRoutes()
}