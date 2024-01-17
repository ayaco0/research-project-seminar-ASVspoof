cd ./data/wav

# すべてのwavファイルを16000Hzに変換
for file in *.wav; do
  sox "$file" -r 16000 "${file%.wav}_16k.wav"
done

# すべての変換後のwavファイルの周波数を表示
for file in *_16k.wav; do
  echo "$file: $(soxi -r "$file") Hz"
done

mkdir wav_16k
mv *_16k.wav wav_16k/