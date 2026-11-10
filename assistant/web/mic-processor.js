// AudioWorklet: collects mono mic samples into fixed 512-sample frames
// (32 ms at 16 kHz, the frame size Silero VAD expects) and posts them out.
const FRAME = 512;

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(FRAME);
    this.offset = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    let i = 0;
    while (i < channel.length) {
      const n = Math.min(FRAME - this.offset, channel.length - i);
      this.buffer.set(channel.subarray(i, i + n), this.offset);
      this.offset += n;
      i += n;
      if (this.offset === FRAME) {
        this.port.postMessage(this.buffer.slice());
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
