// sova_audio_helper.swift
//
// Captures system audio via ScreenCaptureKit and writes raw PCM
// (float32, mono, 16 kHz) to stdout for Python to consume.
//
// Compile:
//   swiftc sova_audio_helper.swift -o sova_audio_helper \
//     -framework ScreenCaptureKit \
//     -framework CoreAudio \
//     -framework AVFoundation
//
// Requires macOS 13.0+

import Foundation
import ScreenCaptureKit
import CoreAudio
import AVFoundation

// ── Target sample rate ────────────────────────────────────────────────────────
let TARGET_SAMPLE_RATE: Double = 16000

// ── Resampler ─────────────────────────────────────────────────────────────────
// Converts AVAudioPCMBuffer from the capture rate (usually 44100/48000)
// down to 16 kHz so Whisper receives exactly what it expects.

func resample(_ buffer: AVAudioPCMBuffer, to targetRate: Double) -> [Float]? {
    guard let inputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: buffer.format.sampleRate,
        channels: 1,
        interleaved: false
    ) else { return nil }

    guard let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: targetRate,
        channels: 1,
        interleaved: false
    ) else { return nil }

    guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
        return nil
    }

    let inputFrames  = AVAudioFrameCount(buffer.frameLength)
    let outputFrames = AVAudioFrameCount(
        Double(inputFrames) * targetRate / buffer.format.sampleRate
    )

    guard let outputBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: outputFrames
    ) else { return nil }

    var error: NSError?
    var inputConsumed = false

    converter.convert(to: outputBuffer, error: &error) { _, outStatus in
        if inputConsumed {
            outStatus.pointee = .noDataNow
            return nil
        }
        // Mix down to mono if the capture buffer has multiple channels
        if buffer.format.channelCount > 1,
           let srcL = buffer.floatChannelData?[0],
           let srcR = buffer.floatChannelData?[1],
           let dst  = inputFormat.channelCount == 1
                        ? AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: inputFrames)
                        : nil {
            dst.frameLength = inputFrames
            if let dstPtr = dst.floatChannelData?[0] {
                for i in 0..<Int(inputFrames) {
                    dstPtr[i] = (srcL[i] + srcR[i]) * 0.5
                }
            }
            inputConsumed = true
            outStatus.pointee = .haveData
            return dst
        }
        inputConsumed = true
        outStatus.pointee = .haveData
        return buffer
    }

    if let error { fputs("[sova_audio_helper] Resample error: \(error)\n", stderr); return nil }

    guard let channelData = outputBuffer.floatChannelData?[0] else { return nil }
    return Array(UnsafeBufferPointer(start: channelData, count: Int(outputBuffer.frameLength)))
}

// ── Stream delegate ───────────────────────────────────────────────────────────

@available(macOS 13.0, *)
class AudioDelegate: NSObject, SCStreamOutput {

    // Accumulate samples until we have CHUNK_SEC seconds worth, then flush
    private let chunkSamples = Int(TARGET_SAMPLE_RATE) * 5   // 5-second chunks
    private var accumulator: [Float] = []
    private let stdout = FileHandle.standardOutput

    func stream(_ stream: SCStream,
                didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio else { return }

        // Convert CMSampleBuffer → AVAudioPCMBuffer
        guard let desc = sampleBuffer.formatDescription,
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(desc)?.pointee
        else { return }

        var asbdCopy = asbd as AudioStreamBasicDescription
        let format = AVAudioFormat(streamDescription: &asbdCopy)
                    ?? AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                     sampleRate: asbd.mSampleRate,
                                     channels: UInt32(asbd.mChannelsPerFrame),
                                     interleaved: false)!

        var blockBuffer: CMBlockBuffer?
        var audioBufferList = AudioBufferList()
        var bufferListSize: Int = 0

        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer, bufferListSizeNeededOut: &bufferListSize,
            bufferListOut: nil, bufferListSize: 0,
            blockBufferAllocator: nil, blockBufferMemoryAllocator: nil,
            flags: 0, blockBufferOut: nil
        )

        let rawPtr = UnsafeMutableRawPointer.allocate(byteCount: bufferListSize, alignment: 8)
        defer { rawPtr.deallocate() }
        let abl = rawPtr.bindMemory(to: AudioBufferList.self, capacity: 1)

        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer, bufferListSizeNeededOut: nil,
            bufferListOut: abl, bufferListSize: bufferListSize,
            blockBufferAllocator: nil, blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &blockBuffer
        )

        guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: abl) else {
            return
        }

        // Resample to 16 kHz mono
        guard let samples = resample(pcmBuffer, to: TARGET_SAMPLE_RATE) else { return }
        accumulator.append(contentsOf: samples)

        // When we have a full chunk, write it to stdout as raw bytes
        if accumulator.count >= chunkSamples {
            let chunk = Array(accumulator.prefix(chunkSamples))
            accumulator.removeFirst(chunkSamples)

            // Write length header (4 bytes, little-endian) then raw float32 data
            // Python reads the header to know how many bytes follow
            var length = UInt32(chunk.count * 4).littleEndian
            withUnsafeBytes(of: &length) { stdout.write(Data($0)) }
            chunk.withUnsafeBytes { stdout.write(Data($0)) }
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

@available(macOS 13.0, *)
func run() async {
    // Request permission
    do {
        _ = try await SCShareableContent.excludingDesktopWindows(false,
                                                                  onScreenWindowsOnly: true)
    } catch {
        fputs("[sova_audio_helper] Permission denied or error: \(error)\n", stderr)
        exit(1)
    }

    // Build stream config — audio only, no video
    let config = SCStreamConfiguration()
    config.capturesAudio          = true
    config.excludesCurrentProcessAudio = false

    // Exclude nothing — capture all system audio
    let filter: SCContentFilter
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(false,
                                                                            onScreenWindowsOnly: true)
        guard let display = content.displays.first else {
            fputs("[sova_audio_helper] No display found\n", stderr)
            exit(1)
        }
        filter = SCContentFilter(display: display, excludingWindows: [])
    } catch {
        fputs("[sova_audio_helper] Could not get content filter: \(error)\n", stderr)
        exit(1)
    }

    let delegate = AudioDelegate()
    let stream   = SCStream(filter: filter, configuration: config, delegate: nil)

    do {
        try stream.addStreamOutput(delegate, type: .audio,
                                   sampleHandlerQueue: .global(qos: .userInteractive))
        try await stream.startCapture()
    } catch {
        fputs("[sova_audio_helper] Stream start failed: \(error)\n", stderr)
        exit(1)
    }

    fputs("[sova_audio_helper] Capturing system audio...\n", stderr)

    // Run until parent process closes stdin (SOVA shutting down)
    do {
        for try await _ in FileHandle.standardInput.bytes { break }
    } catch {
        // stdin closed or error — either way, time to stop
    }

    try? await stream.stopCapture()
}

if #available(macOS 13.0, *) {
    Task { await run() }
    RunLoop.main.run()
} else {
    fputs("[sova_audio_helper] Requires macOS 13+\n", stderr)
    exit(1)
}