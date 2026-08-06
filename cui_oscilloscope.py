#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cui_oscilloscope.py
--------------------
複数のWAVファイルを格子状(縦x横)に並べて、波形をオシロスコープ風に
描画した動画(mp4)を書き出すCUIツール。

- 白い波形 / 黒背景 (色は変更可)
- 各セルの区切りに線を引く
- 装飾要素(軸・目盛り・文字)は一切なし
- オプションでゼロクロストリガーによる波形の安定化
- オプションで入力WAVをミックスして音声として動画に付与

使用例:
    python cui_oscilloscope.py a.wav b.wav c.wav d.wav \
        --cols 2 --rows 2 --output out.mp4

依存: numpy, opencv-python, ffmpeg(PATHに存在すること)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import os

import numpy as np
import cv2


# --------------------------------------------------------------------------
# WAV読み込み
# --------------------------------------------------------------------------
def load_wav_mono_float(path: str):
    """WAVファイルをモノラル・float32([-1,1]目安)で読み込む。

    scipy.io.wavfile を使うことで外部ライブラリ(libsndfile等)への依存を避ける。
    """
    from scipy.io import wavfile

    sr, data = wavfile.read(path)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        data = data.astype(np.float32)
    else:
        raise ValueError(f"未対応のWAVサンプル形式です: {data.dtype} ({path})")

    if data.ndim == 2:
        # 複数チャンネルはモノラルにダウンミックス
        data = data.mean(axis=1).astype(np.float32)

    return sr, data


def peak_normalize(samples: np.ndarray, headroom: float = 0.9) -> np.ndarray:
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak < 1e-9:
        return samples
    return samples * (headroom / peak)


# --------------------------------------------------------------------------
# トリガー(ゼロクロス安定化)
# --------------------------------------------------------------------------
def find_trigger(samples: np.ndarray, center: int, search_radius: int) -> int:
    """center付近で負→正のゼロクロス点を探し、波形表示の揺れを抑える。
    見つからない場合はcenterをそのまま返す。
    """
    lo = max(0, center - search_radius)
    hi = min(len(samples), center + search_radius)
    if hi - lo < 2:
        return center

    segment = samples[lo:hi]
    signs = segment >= 0
    crossings = np.where(np.diff(signs.astype(np.int8)) == 1)[0]
    if crossings.size == 0:
        return center

    crossings_abs = crossings + lo
    idx = crossings_abs[np.argmin(np.abs(crossings_abs - center))]
    return int(idx)


def get_window(samples: np.ndarray, center: int, window_len: int) -> np.ndarray:
    """centerを中心にwindow_len個のサンプルを取り出す。範囲外は0埋め。"""
    half = window_len // 2
    start = center - half
    end = start + window_len

    if start >= 0 and end <= len(samples):
        return samples[start:end]

    buf = np.zeros(window_len, dtype=np.float32)
    src_start = max(0, start)
    src_end = min(len(samples), end)
    if src_end > src_start:
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        buf[dst_start:dst_end] = samples[src_start:src_end]
    return buf


# --------------------------------------------------------------------------
# 色パース
# --------------------------------------------------------------------------
def parse_color(s: str):
    """'R,G,B' 形式の文字列を (B,G,R) のBGRタプルに変換 (OpenCV用)"""
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("色は 'R,G,B' の形式で指定してください (例: 255,255,255)")
    r, g, b = parts
    return (b, g, r)


# --------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="複数のWAVを格子状に並べて波形動画(mp4)を書き出すCUIオシロスコープ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("wavs", nargs="+", help="入力WAVファイル (格子を埋める順に指定)")
    parser.add_argument("--cols", type=int, required=True, help="横方向の並び数")
    parser.add_argument("--rows", type=int, required=True, help="縦方向の並び数")
    parser.add_argument("--output", "-o", default="output.mp4", help="出力mp4ファイルパス")

    parser.add_argument("--width", type=int, default=1280, help="出力動画の横幅(px)")
    parser.add_argument("--height", type=int, default=720, help="出力動画の縦幅(px)")
    parser.add_argument("--fps", type=float, default=60.0, help="フレームレート")

    parser.add_argument("--window-ms", type=float, default=30.0,
                         help="1フレームに表示する波形の時間幅(ミリ秒)")
    parser.add_argument("--duration", type=float, default=None,
                         help="出力動画の長さ(秒)。省略時は最長のWAVに合わせる")

    parser.add_argument("--line-thickness", type=int, default=2, help="波形線の太さ(px)")
    parser.add_argument("--line-color", type=parse_color, default="255,255,255",
                         help="波形の色 'R,G,B'")
    parser.add_argument("--bg-color", type=parse_color, default="0,0,0",
                         help="背景色 'R,G,B'")
    parser.add_argument("--grid-color", type=parse_color, default="90,90,90",
                         help="セル区切り線の色 'R,G,B'")
    parser.add_argument("--grid-thickness", type=int, default=1, help="区切り線の太さ(px)")

    parser.add_argument("--no-trigger", action="store_true",
                         help="ゼロクロストリガーによる波形安定化を無効化する")
    parser.add_argument("--no-normalize", action="store_true",
                         help="各WAVごとのピーク正規化を無効化する")

    parser.add_argument("--no-audio", action="store_true",
                         help="出力動画に音声(全WAVのミックス)を付与しない")

    parser.add_argument("--crf", type=int, default=18, help="libx264のCRF値(小さいほど高画質)")
    parser.add_argument("--preset", default="veryfast", help="libx264のpreset")

    args = parser.parse_args()

    n_cells = args.rows * args.cols
    if len(args.wavs) > n_cells:
        parser.error(f"WAVの数({len(args.wavs)})が格子のセル数({n_cells} = "
                     f"{args.rows}行 x {args.cols}列)を超えています")

    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg が見つかりません。インストールしてPATHを通してください")

    # --- WAV読み込み ---
    print("[1/4] WAV読み込み中...")
    tracks = []  # list of (sr, samples)
    for path in args.wavs:
        sr, samples = load_wav_mono_float(path)
        if not args.no_normalize:
            samples = peak_normalize(samples)
        tracks.append((sr, samples))
        dur = len(samples) / sr
        print(f"  - {path}: sr={sr}Hz, dur={dur:.2f}s")

    if args.duration is not None:
        total_duration = args.duration
    else:
        total_duration = max(len(s) / sr for sr, s in tracks)

    n_frames = max(1, int(round(total_duration * args.fps)))

    # --- セルサイズ計算(割り切れるように実サイズを調整) ---
    cell_w = args.width // args.cols
    cell_h = args.height // args.rows
    real_w = cell_w * args.cols
    real_h = cell_h * args.rows
    if (real_w, real_h) != (args.width, args.height):
        print(f"[info] 出力解像度を {real_w}x{real_h} に調整しました "
              f"({args.cols}列 x {args.rows}行に割り切れるサイズ)")

    amp_scale = (cell_h / 2.0) * 0.9  # 上下9割を波形の振幅範囲として使う

    # --- ffmpeg (映像のみ) をパイプで起動 ---
    video_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    video_tmp_path = video_tmp.name
    video_tmp.close()

    ffmpeg_video_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{real_w}x{real_h}",
        "-r", str(args.fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(args.crf),
        "-preset", args.preset,
        video_tmp_path,
    ]

    print("[2/4] 動画フレームを描画中...")
    proc = subprocess.Popen(ffmpeg_video_cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    window_samples_cache = {}
    trigger_pos = [0] * len(tracks)  # 各トラックの前回トリガー位置(検索の起点)

    try:
        for frame_idx in range(n_frames):
            t = frame_idx / args.fps
            frame = np.empty((real_h, real_w, 3), dtype=np.uint8)
            frame[:] = args.bg_color

            for i in range(n_cells):
                row, col = divmod(i, args.cols)
                y0, y1 = row * cell_h, (row + 1) * cell_h
                x0, x1 = col * cell_w, (col + 1) * cell_w

                if i >= len(tracks):
                    continue  # WAVが足りない分は空セル(背景のみ)

                sr, samples = tracks[i]
                window_len = max(2, int(round(sr * args.window_ms / 1000.0)))
                center = int(round(t * sr))

                if not args.no_trigger:
                    search_radius = window_len // 2
                    center = find_trigger(samples, center, search_radius)

                window = get_window(samples, center, window_len)

                # cell_w ピクセルに合わせて線形補間
                if window_len == cell_w:
                    line_vals = window
                else:
                    xp = np.arange(window_len, dtype=np.float32)
                    x_new = np.linspace(0, window_len - 1, cell_w, dtype=np.float32)
                    line_vals = np.interp(x_new, xp, window)

                ys = (cell_h / 2.0 - line_vals * amp_scale).astype(np.int32)
                ys = np.clip(ys, 0, cell_h - 1)
                xs = np.arange(cell_w, dtype=np.int32)
                points = np.column_stack((xs, ys)).reshape(-1, 1, 2)

                cell_view = frame[y0:y1, x0:x1]
                cv2.polylines(cell_view, [points], isClosed=False,
                               color=args.line_color, thickness=args.line_thickness,
                               lineType=cv2.LINE_AA)

            # --- セル区切り線を最後に上書き描画 ---
            for c in range(1, args.cols):
                x = c * cell_w
                cv2.line(frame, (x, 0), (x, real_h - 1), args.grid_color,
                          args.grid_thickness, lineType=cv2.LINE_AA)
            for r in range(1, args.rows):
                y = r * cell_h
                cv2.line(frame, (0, y), (real_w - 1, y), args.grid_color,
                          args.grid_thickness, lineType=cv2.LINE_AA)

            proc.stdin.write(frame.tobytes())

            if frame_idx % max(1, n_frames // 20) == 0:
                pct = 100.0 * frame_idx / n_frames
                print(f"\r  {pct:5.1f}% ({frame_idx}/{n_frames})", end="", flush=True)

        print(f"\r  100.0% ({n_frames}/{n_frames})")
    finally:
        proc.stdin.close()
        stderr_out = proc.stderr.read().decode(errors="ignore")
        ret = proc.wait()
        if ret != 0:
            print(stderr_out, file=sys.stderr)
            os.unlink(video_tmp_path)
            sys.exit("ffmpeg (映像エンコード) が失敗しました")

    # --- 音声ミックスダウン & mux ---
    if args.no_audio:
        print("[3/4] 音声処理をスキップ")
        print("[4/4] 出力ファイルを配置中...")
        shutil.move(video_tmp_path, args.output)
    else:
        print("[3/4] 音声をミックスダウン中...")
        target_sr = max(sr for sr, _ in tracks)
        n_samples_out = int(round(total_duration * target_sr))
        mix = np.zeros(n_samples_out, dtype=np.float32)

        for sr, samples in tracks:
            if sr != target_sr:
                # 簡易リサンプリング(線形補間)
                old_n = len(samples)
                new_n = int(round(old_n * target_sr / sr))
                xp = np.arange(old_n, dtype=np.float32)
                x_new = np.linspace(0, old_n - 1, new_n, dtype=np.float32)
                samples = np.interp(x_new, xp, samples).astype(np.float32)
            n = min(len(samples), n_samples_out)
            mix[:n] += samples[:n]

        peak = np.max(np.abs(mix)) if mix.size else 0.0
        if peak > 1.0:
            mix = mix / peak * 0.98

        audio_tmp_path = video_tmp_path + ".wav"
        from scipy.io import wavfile
        wavfile.write(audio_tmp_path, target_sr, mix.astype(np.float32))

        print("[4/4] 映像と音声を結合中...")
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", video_tmp_path,
            "-i", audio_tmp_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            args.output,
        ]
        ret = subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.unlink(video_tmp_path)
        os.unlink(audio_tmp_path)
        if ret.returncode != 0:
            print(ret.stderr.decode(errors="ignore"), file=sys.stderr)
            sys.exit("ffmpeg (音声結合) が失敗しました")

    print(f"完了: {args.output}")


if __name__ == "__main__":
    main()
