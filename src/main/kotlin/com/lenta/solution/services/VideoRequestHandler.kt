package com.lenta.solution.services

import io.ktor.http.content.*

class VideoRequestHandler {

    data class VideoData(
        val bytes: ByteArray,
        val fileName: String,
        val size: Int
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is VideoData) return false
            return bytes.contentEquals(other.bytes) &&
                    fileName == other.fileName &&
                    size == other.size
        }

        override fun hashCode(): Int {
            var result = bytes.contentHashCode()
            result = 31 * result + fileName.hashCode()
            result = 31 * result + size
            return result
        }
    }

    suspend fun extractVideoData(multipart: MultiPartData): VideoData? {
        var videoBytes: ByteArray? = null
        var fileName = "video.mp4"

        multipart.forEachPart { part ->
            when (part) {
                is PartData.FileItem -> {
                    if (part.name == "video") {
                        videoBytes = part.streamProvider().readBytes()
                        fileName = part.originalFileName ?: "video.mp4"
                    }
                    part.dispose()
                }
                else -> part.dispose()
            }
        }

        return videoBytes?.let { bytes ->
            VideoData(
                bytes = bytes,
                fileName = fileName,
                size = bytes.size
            )
        }
    }

    fun logVideoInfo(log: org.slf4j.Logger, processorName: String, videoData: VideoData) {
        log.info(
            "Processing video with $processorName: ${videoData.fileName}, size: ${videoData.size} bytes"
        )
    }
}