# concat_guard_gui_progress_v2.py (parcheado N/A-safe)

import os
import sys
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    USE_DND = True
except Exception:
    USE_DND = False

try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

def _ensure_bundled_ffmpeg_on_path():
    base = getattr(sys, "_MEIPASS", None)
    if base and os.path.isdir(base):
        os.environ["PATH"] = base + os.pathsep + os.environ.get("PATH", "")
_ensure_bundled_ffmpeg_on_path()

APP_TITLE = "Concat Guard GUI"
VERSION = "2.0"

def get_config_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        cfg_dir = base / "ConcatGuard"
    else:
        cfg_dir = Path.home() / ".concat_guard"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "config.json"

def load_config() -> dict:
    p = get_config_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_config(cfg: dict):
    p = get_config_path()
    try:
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass

def default_output_dir() -> Path:
    if os.name == "nt":
        vids = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Videos"
        out = vids / "ConcatGuard Output"
    elif sys.platform == "darwin":
        out = Path.home() / "Movies" / "ConcatGuard Output"
    else:
        out = Path.home() / "Videos" / "ConcatGuard-Output"
    out.mkdir(parents=True, exist_ok=True)
    return out

def which_path(bin_name: str) -> str | None:
    from shutil import which
    return which(bin_name)

def ensure_local_onedrive(path: Path):
    if os.name == "nt":
        try:
            subprocess.run(["attrib", "-P", "-U", str(path)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def natural_key(name: str):
    tokens = re.findall(r'\d+|\D+', name)
    key = []
    for t in tokens:
        if t.isdigit():
            key.append((1, int(t)))
        else:
            key.append((0, t.lower()))
    return key

def collect_files(folder: Path, exts: List[str]) -> List[Path]:
    exts_lower = {e.lower().lstrip('.') for e in exts}
    files = []
    for p in folder.glob("*"):
        if p.is_file() and p.suffix.lower().lstrip('.') in exts_lower:
            files.append(p.resolve())
    return files

def write_concat_list(paths: List[Path], list_path: Path) -> None:
    lines = []
    for p in paths:
        s = str(p).replace("'", "''")
        lines.append(f"file '{s}'")
    list_path.write_text("\n".join(lines), encoding="ascii")

# ----------------------------
# PARCHE ROBUSTO ffprobe (N/A)
# ----------------------------

def _safe_int(val, default=0):
    try:
        if isinstance(val, str) and val.strip().upper() == "N/A":
            return default
        return int(float(val))
    except Exception:
        return default

def _safe_float(val, default=0.0):
    try:
        if isinstance(val, str) and val.strip().upper() == "N/A":
            return default
        return float(val)
    except Exception:
        return default

def _parse_fps(fr):
    """Convierte '30000/1001', '30/1', '29.97' o 'N/A' a float fps seguro."""
    try:
        if not fr or str(fr).upper() == "N/A":
            return 0.0
        if "/" in fr:
            n, d = fr.split("/", 1)
            n = _safe_float(n, 0.0)
            d = _safe_float(d, 0.0)
            return (n / d) if d else 0.0
        return _safe_float(fr, 0.0)
    except Exception:
        return 0.0

def probe_video_stats(path: Path):
    """
    Devuelve (duration_s, fps, frames_aprox).
    - duration_s: format.duration (float, 0.0 si ausente/N/A)
    - fps: stream.avg_frame_rate o r_frame_rate (float, 0.0 si ausente/N/A)
    - frames_aprox: usa nb_frames si existe (y no es N/A); si no, duration*fps
    Nunca levanta excepción por 'N/A'.
    """
    try:
        cmd = ["ffprobe", "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(path)]
        out = subprocess.check_output(cmd, text=True)
        data = json.loads(out or "{}")
        duration = _safe_float(data.get("format", {}).get("duration"), 0.0)
        streams = data.get("streams", [])
        vstream = next((s for s in streams if s.get("codec_type") == "video"), {}) or {}
        nb_frames = vstream.get("nb_frames")
        fps = _parse_fps(vstream.get("avg_frame_rate") or vstream.get("r_frame_rate"))
        if nb_frames and str(nb_frames).upper() != "N/A":
            frames = _safe_int(nb_frames, 0)
        else:
            frames = _safe_int(duration * fps, 0)
        return duration, fps, frames
    except Exception:
        return 0.0, 0.0, 0

def ffprobe_duration(path: Path) -> float:
    # Ahora usa el sondeo robusto
    dur, _, _ = probe_video_stats(path)
    return dur

def secs_to_hms(secs: float) -> str:
    if secs < 0: secs = 0
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def parse_progress_kv(line: str) -> dict:
    if "=" in line:
        k, v = line.split("=", 1)
        return {k.strip(): v.strip()}
    return {}

def run_piped(cmd: list, cwd: Path | None = None):
    si = None
    creation = 0
    if os.name == "nt":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creation = subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        startupinfo=si,
        creationflags=creation
    )
    try:
        for line in iter(proc.stdout.readline, ''):
            yield line.rstrip("\n")
    finally:
        try: proc.stdout.close()
        except Exception: pass
        proc.wait()

class Ema:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.val = None
    def update(self, x):
        if x <= 0: return self.val
        if self.val is None: self.val = x
        else: self.val = self.alpha * x + (1 - self.alpha) * self.val
        return self.val

BaseTk = tk.Tk if not USE_DND else TkinterDnD.Tk

class App(BaseTk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} • v{VERSION}")
        self.geometry("1000x700")
        self.minsize(900, 600)

        self.cfg = load_config()

        start_folder = self.cfg.get("last_folder") or str(Path.cwd())
        self.folder = tk.StringVar(value=start_folder)
        self.exts = tk.StringVar(value="mp4,mov,mkv,avi")
        self.mode = tk.StringVar(value=self.cfg.get("mode", "ts"))

        default_outdir = self.cfg.get("output_dir") or str(default_output_dir())
        self.output_dir = tk.StringVar(value=default_outdir)

        default_out = f"{Path(self.folder.get()).name} VIDEO COMPLETO.mp4"
        self.output_name = tk.StringVar(value=self.cfg.get("output_name", default_out))

        if os.name == "nt":
            local_root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            self.temp_dir = (Path(local_root) / "ConcatGuardTemp").resolve()
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "ConcatGuardTemp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        top = ttk.Frame(self); top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Carpeta origen:").pack(side="left")
        self.entry_folder = ttk.Entry(top, textvariable=self.folder, width=70)
        self.entry_folder.pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(top, text="Elegir…", command=self.choose_folder).pack(side="left", padx=4)
        ttk.Button(top, text="Escanear", command=self.scan_folder).pack(side="left", padx=4)

        opt = ttk.Frame(self); opt.pack(fill="x", padx=10, pady=(0,8))
        ttk.Label(opt, text="Extensiones (coma):").pack(side="left")
        ttk.Entry(opt, textvariable=self.exts, width=25).pack(side="left", padx=6)
        ttk.Label(opt, text="Salida (nombre):").pack(side="left", padx=(12,0))
        ttk.Entry(opt, textvariable=self.output_name, width=36).pack(side="left", padx=6)
        ttk.Label(opt, text="Modo:").pack(side="left", padx=(12,0))
        ttk.Radiobutton(opt, text="TS (compatibilidad)", variable=self.mode, value="ts").pack(side="left", padx=3)
        ttk.Radiobutton(opt, text="COPY (rápido)", variable=self.mode, value="copy").pack(side="left", padx=3)

        row2 = ttk.Frame(self); row2.pack(fill="x", padx=10, pady=(0,8))
        ttk.Label(row2, text="Carpeta salida:").pack(side="left")
        ttk.Entry(row2, textvariable=self.output_dir, width=60).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(row2, text="Elegir…", command=self.choose_output_dir).pack(side="left", padx=4)

        mid = ttk.Frame(self); mid.pack(fill="both", expand=True, padx=10, pady=4)
        self.listbox = tk.Listbox(mid, selectmode="extended")
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self.listbox.yview)
        sb.pack(side="left", fill="y"); self.listbox.config(yscrollcommand=sb.set)

        btns = ttk.Frame(mid); btns.pack(side="left", fill="y", padx=8)
        ttk.Button(btns, text="Orden alfabético", command=self.sort_alpha).pack(fill="x", pady=3)
        ttk.Button(btns, text="Orden natural", command=self.sort_natural).pack(fill="x", pady=3)
        ttk.Button(btns, text="Subir ▲", command=self.move_up).pack(fill="x", pady=3)
        ttk.Button(btns, text="Bajar ▼", command=self.move_down).pack(fill="x", pady=3)
        ttk.Button(btns, text="Quitar", command=self.remove_selected).pack(fill="x", pady=10)
        self.btn_concat = ttk.Button(btns, text="Concatenar", command=self.start_concat)
        self.btn_concat.pack(fill="x", pady=12)
        self.btn_refresh = ttk.Button(btns, text="Refrescar", command=self.scan_folder)
        self.btn_refresh.pack(fill="x", pady=3)

        logf = ttk.LabelFrame(self, text="Registro")
        logf.pack(fill="both", expand=True, padx=10, pady=6)
        self.log = tk.Text(logf, height=10)
        self.log.pack(fill="both", expand=True)

        pf = ttk.Frame(self); pf.pack(fill="x", padx=10, pady=(0,6))
        ttk.Label(pf, text="Progreso:").pack(side="left")
        self.progress = ttk.Progressbar(pf, mode="determinate", maximum=100, value=0, length=300)
        self.progress.pack(side="left", padx=8)
        self.lbl_pct = ttk.Label(pf, text="0%")
        self.lbl_pct.pack(side="left")
        self.lbl_phase = ttk.Label(pf, text="")
        self.lbl_phase.pack(side="left", padx=8)

        self.lbl_friendly = ttk.Label(self, text="Listo para comenzar.")
        self.lbl_friendly.pack(fill="x", padx=10, pady=(0,2))
        self.lbl_sizeprog = ttk.Label(self, text="")
        self.lbl_sizeprog.pack(fill="x", padx=10, pady=(0,6))

        bf = ttk.Frame(self); bf.pack(fill="x", padx=10, pady=(0,10))
        self.btn_open_out = ttk.Button(bf, text="Abrir carpeta de salida", command=self.open_output_dir, state="disabled")
        self.btn_open_out.pack(side="left")
        ttk.Button(self, text="Limpiar log", command=lambda: self.log.delete("1.0", "end")).pack(pady=(0,6))

        self.scan_folder()
        self.log_line(f"[INFO] ffmpeg: {which_path('ffmpeg')}")
        self.log_line(f"[INFO] ffprobe: {which_path('ffprobe')}")
        self.log_line(f"[INFO] Temp dir: {self.temp_dir}")
        if USE_DND:
            self.log_line("[INFO] DnD: soltar una carpeta sobre la ventana.")

        geom = self.cfg.get("geometry")
        if geom:
            try: self.geometry(geom)
            except Exception: pass

        self._busy = False

        if USE_DND:
            try:
                self.drop_target_register(DND_FILES)
                self.dnd_bind('<<Drop>>', self._on_drop)
            except Exception as e:
                self.log_line(f"[WARN] DnD no disponible: {e}")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.cfg["last_folder"] = self.folder.get()
        self.cfg["mode"] = self.mode.get()
        self.cfg["output_dir"] = self.output_dir.get()
        self.cfg["output_name"] = self.output_name.get()
        try:
            self.cfg["geometry"] = self.geometry()
        except Exception:
            pass
        save_config(self.cfg)
        self.destroy()

    def open_output_dir(self):
        path = Path(self.output_dir.get() or self.folder.get())
        if not path.exists(): return
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])

    def log_line(self, s: str):
        self.log.insert("end", s + "\n"); self.log.see("end"); self.update_idletasks()

    def choose_folder(self):
        cur = Path(self.folder.get())
        new = filedialog.askdirectory(initialdir=str(cur if cur.exists() else Path.home()))
        if new:
            self.folder.set(new)
            self.output_name.set(f"{Path(new).name} VIDEO COMPLETO.mp4")
            self.scan_folder()

    def choose_output_dir(self):
        base = Path(self.output_dir.get())
        new = filedialog.askdirectory(initialdir=str(base if base.exists() else Path.home()))
        if new:
            self.output_dir.set(new)

    def scan_folder(self):
        self.listbox.delete(0, "end")
        folder = Path(self.folder.get())
        if not folder.exists():
            messagebox.showerror("Carpeta inválida", str(folder))
            return
        exts = [e.strip() for e in self.exts.get().split(",") if e.strip()]
        files = collect_files(folder, exts)
        files = [f for f in files if f.suffix.lower() != ".ts"]
        files.sort(key=lambda p: natural_key(p.name))
        for f in files:
            self.listbox.insert("end", str(f.name))
        if not files:
            self.log_line("[INFO] No se encontraron archivos de video.")

    def get_current_files(self) -> List[Path]:
        folder = Path(self.folder.get())
        return [folder / self.listbox.get(i) for i in range(self.listbox.size())]

    def sort_alpha(self):
        items = [self.listbox.get(i) for i in range(self.listbox.size())]
        items.sort(key=str.lower)
        self.listbox.delete(0, "end")
        for it in items: self.listbox.insert("end", it)

    def sort_natural(self):
        items = [self.listbox.get(i) for i in range(self.listbox.size())]
        items.sort(key=natural_key)
        self.listbox.delete(0, "end")
        for it in items: self.listbox.insert("end", it)

    def move_up(self):
        sel = list(self.listbox.curselection())
        if not sel or sel[0] == 0: return
        for i in sel:
            txt = self.listbox.get(i)
            self.listbox.delete(i)
            self.listbox.insert(i-1, txt)
            self.listbox.selection_set(i-1)

    def move_down(self):
        sel = list(self.listbox.curselection())
        if not sel: return
        n = self.listbox.size()
        if sel[-1] == n-1: return
        for i in reversed(sel):
            txt = self.listbox.get(i)
            self.listbox.delete(i)
            self.listbox.insert(i+1, txt)
            self.listbox.selection_set(i+1)

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        for i in reversed(sel):
            self.listbox.delete(i)

    def _set_phase_label(self, phase: int):
        if phase == 1:
            self.lbl_phase.config(text="Convirtiendo a intermedio (1/2)")
        elif phase == 2:
            self.lbl_phase.config(text="Uniendo archivo final (2/2)")
        else:
            self.lbl_phase.config(text="")

    def _set_combined_progress(self, frac: float):
        pct = max(0, min(100, int(frac * 100)))
        self.progress.config(value=pct)
        self.lbl_pct.config(text=f"{pct}%")
        self.update_idletasks()

    def _progress_update_friendly(self, processed: float, total: float, eta: float, phase: int, sp: float | None = None):
        speed_txt = f"{sp:.0f}×" if sp else "—"
        self._set_phase_label(phase)
        self.lbl_friendly.config(text=f"Procesando {secs_to_hms(processed)} de {secs_to_hms(total)} • Velocidad ≈ {speed_txt} • ETA ≈ {secs_to_hms(eta)}")
        self.update_idletasks()

    def start_concat(self):
        if getattr(self, "_busy", False):
            return
        files = self.get_current_files()
        if not files:
            messagebox.showerror("Sin archivos", "La lista está vacía.")
            return

        out_dir = Path(self.output_dir.get() or default_output_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = self.output_name.get().strip() or f"{Path(self.folder.get()).name} VIDEO COMPLETO.mp4"
        out_path = out_dir / out_name
        if out_path.exists() and not messagebox.askyesno("Sobrescribir", f"{out_name} ya existe. ¿Reemplazar?"):
            return

        for f in files:
            ensure_local_onedrive(f)

        if os.name == "nt":
            local_root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            workdir = (Path(local_root) / "ConcatGuardTemp").resolve()
        else:
            workdir = Path(tempfile.gettempdir()) / "ConcatGuardTemp"
        workdir.mkdir(parents=True, exist_ok=True)

        list_path = workdir / "lista_ffmpeg.txt"
        try:
            write_concat_list(files, list_path)
        except Exception as e:
            messagebox.showerror("Error al crear lista", str(e)); return

        tmp_out = workdir / (".__concat_tmp_" + next(tempfile._get_candidate_names()) + ".mp4")
        if tmp_out.exists():
            tmp_out.unlink()

        total_secs = sum(ffprobe_duration(f) for f in files)
        combined_total = max(1.0, 2 * total_secs)

        self._busy = True
        self.btn_concat.config(state="disabled")
        self.btn_refresh.config(state="disabled")
        self.btn_open_out.config(state="disabled")
        self._set_combined_progress(0)
        self.lbl_friendly.config(text="Preparando…")
        self.lbl_sizeprog.config(text="")
        self._set_phase_label(1 if self.mode.get()=="ts" else 2)

        import threading
        ema = Ema(alpha=0.2)
        start = time.time()

        def on_progress_phase1(proc_secs):
            frac = (proc_secs) / combined_total
            self._set_combined_progress(frac)

        def on_progress_phase2(phase2_secs):
            frac = (total_secs + phase2_secs) / combined_total
            self._set_combined_progress(frac)

        def worker():
            rc = 1
            try:
                self.log_line(f"[INFO] Carpeta origen: {self.folder.get()}")
                self.log_line(f"[INFO] Salida: {out_path}")
                self.log_line(f"[INFO] Modo: {self.mode.get().upper()}")
                self.log_line(f"[INFO] Archivos: {len(files)}")

                if self.mode.get() == "copy":
                    pass
                else:
                    processed_global = 0.0
                    for p in files:
                        dur = ffprobe_duration(p)
                        ts = workdir / (p.stem + ".ts")
                        cmd = ["ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-y",
                               "-i", str(p), "-c", "copy", "-bsf:v", "h264_mp4toannexb",
                               "-f", "mpegts", str(ts),
                               "-progress", "pipe:1"]
                        for line in run_piped(cmd):
                            d = parse_progress_kv(line)
                            if "out_time_ms" in d:
                                cur = min(dur, int(d["out_time_ms"]) / 1000.0) if dur > 0 else 0.0
                                elapsed = max(0.01, time.time() - start)
                                inst_speed = (processed_global + cur) / elapsed
                                sp = ema.update(inst_speed) or inst_speed
                                eta = ((combined_total - (processed_global + cur)) / sp) if sp > 0 else 0.0
                                self.after(0, lambda c=processed_global+cur: on_progress_phase1(c))
                                self.after(0, lambda c=processed_global+cur, t=combined_total, e=eta: self._progress_update_friendly(c, combined_total, e, phase=1, sp=sp))
                        if not ts.exists() or ts.stat().st_size == 0:
                            raise RuntimeError("Falló la conversión a TS.")
                        processed_global += dur
                        self.after(0, lambda c=processed_global: on_progress_phase1(c))

                if self.mode.get() == "copy":
                    list_for_concat = list_path
                else:
                    ts_files = [workdir / (Path(p).stem + ".ts") for p in files]
                    list_for_concat = workdir / "lista_ts.txt"
                    write_concat_list(ts_files, list_for_concat)

                cmd2 = ["ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-y",
                        "-f", "concat", "-safe", "0", "-i", str(list_for_concat),
                        "-c", "copy", "-movflags", "+faststart",
                        "-progress", "pipe:1",
                        str(tmp_out)]
                expected_total_size = sum((f.stat().st_size for f in files))
                seen_end = False
                for line in run_piped(cmd2):
                    d = parse_progress_kv(line)
                    if "out_time_ms" in d:
                        phase2_secs = int(d["out_time_ms"]) / 1000.0
                        elapsed = max(0.01, time.time() - start)
                        inst_speed = (total_secs + phase2_secs) / elapsed
                        sp = ema.update(inst_speed) or inst_speed
                        remaining_units = max(0.0, (2*total_secs) - (total_secs + phase2_secs))
                        eta = remaining_units / sp if sp > 0 else 0.0
                        self.after(0, lambda s=phase2_secs: on_progress_phase2(s))
                        self.after(0, lambda c=total_secs+phase2_secs, t=2*total_secs, e=eta: self._progress_update_friendly(c, 2*total_secs, e, phase=2, sp=sp))
                    if "total_size" in d and expected_total_size > 0:
                        try:
                            cur_size = int(d["total_size"])
                            size_pct = max(0, min(100, int(cur_size * 100 / expected_total_size)))
                            self.after(0, lambda p=size_pct: self.lbl_sizeprog.config(text=f"Escritura de archivo: {p}% (aprox.)"))
                        except Exception:
                            pass
                    if d.get("progress") == "end":
                        seen_end = True

                if not (tmp_out.exists() and tmp_out.stat().st_size > 0 and seen_end):
                    raise RuntimeError("Falló la unión final.")

                # Mover de forma segura aun entre unidades distintas (C: ↔ D:)
                try:
                    if out_path.exists():
                        out_path.unlink()
                    shutil.move(str(tmp_out), str(out_path))
                except Exception as e:
                    try:
                        if out_path.exists():
                            out_path.unlink()
                        shutil.copy2(str(tmp_out), str(out_path))
                        try:
                            tmp_out.unlink(missing_ok=True)
                        except Exception:
                            pass
                    except Exception as e2:
                        raise RuntimeError(f"No se pudo colocar la salida en destino: {e2}") from e

                # (Opcional) escribir pequeño resumen, si alguna vez definís 'summary'
                try:
                    (out_path.parent / (out_path.stem + ".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")  # noqa
                except Exception:
                    pass

                self.after(0, lambda: (self._set_combined_progress(1.0),
                                       self.lbl_friendly.config(text="Completado"),
                                       self.btn_open_out.config(state="normal")))

                rc = 0

            except Exception as e:
                self.log_line(f"[ERROR] {e}")
                rc = 1

            finally:
                try: (workdir / "lista_ts.txt").unlink(missing_ok=True)
                except Exception: pass
                try: list_path.unlink(missing_ok=True)
                except Exception: pass

            def wrap_finish():
                if rc == 0:
                    self.open_output_dir()
                    try:
                        if HAS_WINSOUND:
                            winsound.Beep(880, 120); winsound.Beep(1175, 120)
                        else:
                            self.bell()
                    except Exception:
                        pass
                    self.log_line(f"[OK] Generado: {out_path.name}")
                else:
                    messagebox.showerror("Error", "FFmpeg falló o la salida no es válida.")

                self.btn_open_out.config(state="normal" if rc == 0 else "disabled")
                self._busy = False
                self.btn_concat.config(state="normal")
                self.btn_refresh.config(state="normal")

            self.after(0, wrap_finish)

        threading.Thread(target=worker, daemon=True).start()

    def _on_drop(self, event):
        data = event.data
        paths, cur, in_brace = [], "", False
        for ch in data:
            if ch == "{": in_brace = True; cur = ""
            elif ch == "}": in_brace = False; paths.append(cur)
            elif ch == " " and not in_brace:
                if cur: paths.append(cur); cur = ""
            else: cur += ch
        if cur: paths.append(cur)
        if not paths: return
        p = Path(paths[0])
        if p.is_file(): p = p.parent
        if p.is_dir():
            self.folder.set(str(p))
            self.output_name.set(f"{p.name} VIDEO COMPLETO.mp4")
            self.scan_folder()

if __name__ == "__main__":
    app = App()
    app.mainloop()