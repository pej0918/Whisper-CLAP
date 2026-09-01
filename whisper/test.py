import whisper

# 모델 로드 (tiny, base, small, medium, large 중 선택)
model = whisper.load_model("medium")

print(model)

# 오디오 파일 변환
# result = model.transcribe("audio_file.mp3")
# print(result["text"])