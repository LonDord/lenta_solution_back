package com.lenta.solution.routes

import com.lenta.solution.services.VideoProcessorFactory
import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*

fun Application.configureRoutes() {
    val processorFactory = VideoProcessorFactory()
    val requestHandler = processorFactory.getRequestHandler()

    routing {
        route("/api") {
            get("/health") {
                call.respond(mapOf("status" to "ok", "timestamp" to System.currentTimeMillis()))
            }

            post("/processing1") {
                val processor = processorFactory.getProcessor("processing1")
                if (processor != null) {
                    processVideoRequest(call, processor, requestHandler)
                } else {
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        mapOf("error" to "Processor not found")
                    )
                }
            }

            post("/processing2") {
                val processor = processorFactory.getProcessor("processing2")
                if (processor != null) {
                    processVideoRequest(call, processor, requestHandler)
                } else {
                    call.respond(
                        HttpStatusCode.InternalServerError,
                        mapOf("error" to "Processor not found")
                    )
                }
            }
        }
    }
}

/**
 * Обрабатывает входящий запрос на обработку видео
 *
 * @param call - объект вызова, содержащий HTTP запрос и ответ
 * @param processor - конкретная реализация обработчика видео
 * @param requestHandler - утилита для извлечения видео из multipart запроса и логирования
 */
private suspend fun processVideoRequest(
    call: ApplicationCall,
    processor: com.lenta.solution.interfaces.VideoProcessor,
    requestHandler: com.lenta.solution.services.VideoRequestHandler
) {
    try {
        // получение multipart данных
        val multipart = call.receiveMultipart()

        // извлечение видео из запроса
        val videoData = requestHandler.extractVideoData(multipart)

        // проверка наличия видео
        if (videoData == null) {
            call.respond(
                HttpStatusCode.BadRequest,
                mapOf("error" to "No video file provided or file is empty")
            )
            return
        }

        // логирование информации о запросе
        requestHandler.logVideoInfo(
            call.application.environment.log,
            processor.getProcessorName(),
            videoData
        )

        val csvResult = processor.process(videoData.bytes)

        // отправка csv файла в ответ
        call.response.header(
            HttpHeaders.ContentDisposition,
            "attachment; filename=\"result.csv\""
        )

        // отправка CSV данных как ответ
        call.respondText(
            contentType = ContentType.Text.CSV,
            text = csvResult
        )

    } catch (e: Exception) {
        call.application.environment.log.error("Processing error", e)
        call.respond(
            HttpStatusCode.InternalServerError,
            mapOf("error" to "Processing failed: ${e.message}")
        )
    }
}