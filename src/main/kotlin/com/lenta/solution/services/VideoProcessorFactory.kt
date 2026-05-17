package com.lenta.solution.services

import com.lenta.solution.interfaces.VideoProcessor

class VideoProcessorFactory {
    private val requestHandler = VideoRequestHandler()

    private val processors = mapOf(
        "processing1" to ProcessingType1(),
        "processing2" to ProcessingType2()
    )

    fun getProcessor(type: String): VideoProcessor? {
        return processors[type]
    }

    fun getRequestHandler(): VideoRequestHandler {
        return requestHandler
    }

    fun getAllProcessors(): List<VideoProcessor> {
        return processors.values.toList()
    }
}