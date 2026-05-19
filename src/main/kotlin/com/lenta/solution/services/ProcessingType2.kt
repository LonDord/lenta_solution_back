package com.lenta.solution.services

import com.lenta.solution.interfaces.VideoProcessor

class ProcessingType2 : VideoProcessor {

    override fun getProcessorName(): String = "processing2"

    override suspend fun process(videoData: ByteArray): String {
        val results = CvServiceClient.process(videoData, model = "high")
        return CvServiceClient.toCsv(results)
    }
}
