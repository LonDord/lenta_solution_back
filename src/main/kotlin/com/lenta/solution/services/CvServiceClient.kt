package com.lenta.solution.services

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.request.*
import io.ktor.client.request.forms.*
import io.ktor.client.statement.*
import io.ktor.http.*

data class DetectionResult(
    val track_id: Int,
    val frame_id: Int,
    val x1: Double,
    val y1: Double,
    val x2: Double,
    val y2: Double,
    val det_conf: Double,
    val decoded_text: String,
)

object CvServiceClient {
    private val mapper = jacksonObjectMapper()
    private val baseUrl = System.getenv("CV_SERVICE_URL") ?: "http://localhost:8000"

    suspend fun process(videoData: ByteArray, model: String): List<DetectionResult> {
        val client = HttpClient(CIO)
        return try {
            val response = client.post("$baseUrl/process?model=$model") {
                setBody(
                    MultiPartFormDataContent(formData {
                        append(
                            "video",
                            videoData,
                            Headers.build {
                                append(HttpHeaders.ContentDisposition, "filename=\"input.mp4\"")
                            },
                        )
                    })
                )
            }
            mapper.readValue(response.bodyAsText())
        } finally {
            client.close()
        }
    }

    fun toCsv(results: List<DetectionResult>): String {
        val csv = StringBuilder()
        csv.appendLine("track_id,frame_id,x1,y1,x2,y2,det_conf,decoded_text")
        for (r in results) {
            csv.appendLine("${r.track_id},${r.frame_id},${r.x1},${r.y1},${r.x2},${r.y2},${r.det_conf},${r.decoded_text}")
        }
        return csv.toString()
    }
}
