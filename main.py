import argparse, os, json, torch, torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torchaudio.transforms as transforms


def process_audio(input_file, output_file, speed=1.0, volume=0.0):
    waveform, sample_rate = torchaudio.load(input_file)

    if speed != 1.0:
        waveform, _ = transforms.Speed(sample_rate, speed)(waveform)

    if volume != 0.0:
        waveform = transforms.Vol(gain=volume, gain_type="db")(waveform)

    torchaudio.save(output_file, waveform, sample_rate)
    print(f"Audio processed and saved to {output_file}")

def transcribe_audio(input_file, lang):
    # Load model from HF
    device = 'cpu'
    torch_dtype = torch.float32

    model_id = "openai/whisper-medium"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id, return_attention_mask=True)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(input_file, generate_kwargs={"language": lang})
    result['wav'] = input_file
    # Write results to JSON file
    output_file = f"{os.path.splitext(input_file)[0]}_transcription.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Transcription saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Audio processing and transcription tool")
    subparsers = parser.add_subparsers(dest="command")
    
    modify_parser = subparsers.add_parser("modify", help="Modify audio file")
    modify_parser.add_argument("input_file", type=str, help="Path to input WAV file")
    modify_parser.add_argument("output_file", type=str, help="Path to output modified WAV file")
    modify_parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (default: 1.0)")
    modify_parser.add_argument("--volume", type=float, default=0.0, help="Volume change in dB (default: 0.0)")
    
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio file to text")
    transcribe_parser.add_argument("input_file", type=str, help="Path to input WAV file")
    transcribe_parser.add_argument("--lang", type=str, choices=["russian", "english"], default="english", help="Language of the model (default: english)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file: {args.input_file} is not exists.")
        return

    if args.command == "modify":
        process_audio(args.input_file, args.output_file, args.speed, args.volume)
    elif args.command == "transcribe":
        transcribe_audio(args.input_file, args.lang)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
