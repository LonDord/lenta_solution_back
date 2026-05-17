package com.lenta.solution.interfaces

interface VideoProcessor {
    suspend fun process(videoData: ByteArray): String

    fun getProcessorName(): String
}