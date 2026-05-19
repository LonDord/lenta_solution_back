plugins {
    kotlin("jvm") version "2.0.20"
    application
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    // Ktor Server
    val ktorVersion = "2.3.7"
    implementation("io.ktor:ktor-server-core:$ktorVersion")
    implementation("io.ktor:ktor-server-netty:$ktorVersion")
    implementation("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-serialization-jackson:$ktorVersion")
    implementation("io.ktor:ktor-server-cors:$ktorVersion")

    // Ktor Client (для вызова Python CV-сервиса)
    implementation("io.ktor:ktor-client-core:$ktorVersion")
    implementation("io.ktor:ktor-client-cio:$ktorVersion")

    // Jackson Kotlin-модуль (парсинг JSON-ответа от CV-сервиса)
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.16.2")

    // Logging
    implementation("ch.qos.logback:logback-classic:1.4.14")

    // CSV Processing
    implementation("com.opencsv:opencsv:5.9")

}

application {
    mainClass.set("com.lenta.solution.ApplicationKt")
}

tasks.jar {
    manifest {
        attributes["Main-Class"] = "com.lenta.solution.ApplicationKt"
    }
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    from(configurations.runtimeClasspath.get().map { if (it.isDirectory) it else zipTree(it) })
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(11)
}