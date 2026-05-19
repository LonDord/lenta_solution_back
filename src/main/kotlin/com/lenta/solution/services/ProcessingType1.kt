package com.lenta.solution.services

import com.lenta.solution.interfaces.VideoProcessor

class ProcessingType1 : VideoProcessor {

    override fun getProcessorName(): String = "processing1"

    override suspend fun process(videoData: ByteArray): String {
        val results = CvServiceClient.process(videoData, model = "low")
        return CvServiceClient.toCsv(results)
    }
}