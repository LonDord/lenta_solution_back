package com.lenta.solution.services

import com.lenta.solution.interfaces.VideoProcessor

class ProcessingType1 : VideoProcessor {

    override fun getProcessorName(): String = "processing1"

    override suspend fun process(videoData: ByteArray): String {
        // TODO: Реализовать алгоритм 1

        val csv = StringBuilder()

        return csv.toString()
    }
}