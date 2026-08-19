import ctypes
import json
import sys
import time
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from audio_engine import (
    ShuffleCrossfadePlayer,
    DEFAULT_CROSSFADE_SECONDS,
    MIN_CROSSFADE_SECONDS,
    MAX_CROSSFADE_SECONDS,
)
from background_engine import BackgroundImageManager

APP_TITLE = "Space Player"

if getattr(sys, "frozen", False):
    # Running as a PyInstaller-frozen exe: bundled data (added via --add-data)
    # is extracted at startup into sys._MEIPASS, not next to the exe itself.
    _RESOURCE_DIR = Path(sys._MEIPASS)
else:
    _RESOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_BACKGROUNDS_DIR = _RESOURCE_DIR / "backgrounds"
ANIM_INTERVAL_MS = 33  # ~30fps

SETTINGS_PATH = Path.home() / ".space_player" / "settings.json"


def _load_settings():
    try:
        with open(SETTINGS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(settings):
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f)
    except Exception:
        pass


def _format_time_remaining(seconds):
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"-{minutes}:{secs:02d}"

BG_COLOR = "#05050f"
PANEL_BG = "#0b0b1e"
TEXT_FG = "#ffed7f"
ACCENT_FG = "#ffed7f"
DIM_FG = "#ffed7f"
TROUGH_COLOR = "#22224a"

TITLEBAR_PURPLE = (90, 30, 140)
TITLEBAR_TEXT_YELLOW = (255, 220, 0)

BUTTON_BG = "#6a1fb0"
BUTTON_ACTIVE_BG = "#8a3ed0"
BUTTON_FG = "#ffdc00"

TIMELINE_TOGGLE_BG = "#2a1245"
TIMELINE_TOGGLE_ACTIVE_BG = "#3a1a5e"

_DWMWA_BORDER_COLOR = 34
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36


def _rgb_to_colorref(rgb):
    r, g, b = rgb
    return r | (g << 8) | (b << 16)


def _apply_windows_titlebar_theme(root):
    """Color the native window border/title bar/title text on Windows 11 (build 22000+).
    No-op (harmlessly) on other platforms or older Windows where this DWM API is absent."""
    if sys.platform != "win32":
        return
    try:
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        for attr, rgb in (
            (_DWMWA_BORDER_COLOR, TITLEBAR_PURPLE),
            (_DWMWA_CAPTION_COLOR, TITLEBAR_PURPLE),
            (_DWMWA_TEXT_COLOR, TITLEBAR_TEXT_YELLOW),
        ):
            cref = ctypes.c_int(_rgb_to_colorref(rgb))
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(cref), ctypes.sizeof(cref))
    except Exception:
        pass


def _cover_resize(img, target_w, target_h):
    """Resize+crop a PIL image to exactly fill (target_w, target_h), cropping overflow."""
    if target_w <= 0 or target_h <= 0:
        return img
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _rounded_rect_points(x1, y1, x2, y2, radius):
    radius = max(0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    return [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]


class CanvasButton:
    """A purple pill-shaped button drawn on a Canvas rather than a native
    tk.Button. Classic tk.Button on macOS's Aqua theme honors custom text
    color but ignores custom background color, rendering as a plain native
    gray button regardless of `bg` - drawing the shape ourselves on a Canvas
    sidesteps that and renders identically on Windows and Mac."""

    def __init__(self, parent, text, command, width=100, height=32,
                 bg=BUTTON_BG, active_bg=BUTTON_ACTIVE_BG, fg=BUTTON_FG,
                 font=("Segoe UI", 9), radius=10, stretch=False):
        self.command = command
        self.bg = bg
        self.active_bg = active_bg
        self.radius = radius

        self.canvas = tk.Canvas(parent, width=width, height=height, bg=PANEL_BG, highlightthickness=0, cursor="hand2")
        self.shape_id = self.canvas.create_polygon(
            _rounded_rect_points(1, 1, width - 1, height - 1, radius),
            smooth=True, fill=bg, outline="",
        )
        self.text_id = self.canvas.create_text(width / 2, height / 2, text=text, fill=fg, font=font)

        self.canvas.tag_bind(self.shape_id, "<Button-1>", self._on_click)
        self.canvas.tag_bind(self.text_id, "<Button-1>", self._on_click)
        self.canvas.bind("<Enter>", lambda e: self.canvas.itemconfig(self.shape_id, fill=self.active_bg))
        self.canvas.bind("<Leave>", lambda e: self.canvas.itemconfig(self.shape_id, fill=self.bg))

        if stretch:
            self.canvas.bind("<Configure>", self._on_resize)

    def _on_click(self, event):
        self.command()

    def _on_resize(self, event):
        w, h = event.width, event.height
        self.canvas.coords(self.shape_id, *_rounded_rect_points(1, 1, w - 1, h - 1, self.radius))
        self.canvas.coords(self.text_id, w / 2, h / 2)

    def grid(self, **kwargs):
        self.canvas.grid(**kwargs)

    def set_text(self, text):
        self.canvas.itemconfig(self.text_id, text=text)

    def get_text(self):
        return self.canvas.itemcget(self.text_id, "text")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("900x620")
        self.root.minsize(600, 400)
        self.root.configure(bg=BG_COLOR)
        _apply_windows_titlebar_theme(self.root)

        self.player = ShuffleCrossfadePlayer()
        self.backgrounds = BackgroundImageManager()
        self.folder = None

        self._bg_photo = None  # keep a reference so Tk doesn't garbage-collect it
        self._current_bg_raw = None  # PIL Image currently displayed (full-res, unresized)
        self._anim_from_resized = None
        self._anim_to_resized = None
        self._anim_to_raw = None
        self._anim_start_time = 0.0
        self._anim_duration = 0.0
        self._animating = False
        self._last_seen_crossfade_id = 0
        self._canvas_size = (900, 620)

        self._settings = _load_settings()
        self.timeline_visible = self._settings.get("timeline_visible", False)
        self._timeline_dragging = False

        self._build_widgets()
        self._load_default_backgrounds()
        self._apply_timeline_visibility()
        self._tick()

    def _build_widgets(self):
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg=BG_COLOR)
        self.canvas.pack(fill="both", expand=True)
        self.bg_image_id = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        panel = tk.Frame(self.canvas, bg=PANEL_BG, padx=14, pady=10)
        self.panel_window = self.canvas.create_window(0, 0, window=panel, anchor="s")
        self.panel = panel
        for col in (1, 2, 3, 4):
            panel.columnconfigure(col, weight=1)

        pad = {"padx": 6, "pady": 4}

        self.folder_label = tk.Label(
            panel, text="No music folder selected", anchor="center", justify="center",
            wraplength=560, bg=PANEL_BG, fg=TEXT_FG,
        )
        self.folder_label.grid(row=0, column=0, columnspan=5, sticky="we", **pad)

        choose_folder_button = CanvasButton(
            panel, text="Choose Music Folder...", command=self._choose_folder,
            width=400, height=32, stretch=True,
        )
        choose_folder_button.grid(row=1, column=0, columnspan=5, sticky="we", **pad)

        self.play_button = CanvasButton(panel, text="Play", command=self._on_play_pause, width=90, height=32)
        self.play_button.grid(row=2, column=0, **pad)
        self.stop_button = CanvasButton(panel, text="Stop", command=self._on_stop, width=90, height=32)
        self.stop_button.grid(row=2, column=1, **pad)
        self.skip_button = CanvasButton(panel, text="Skip", command=self._on_skip, width=90, height=32)
        self.skip_button.grid(row=2, column=2, **pad)
        self.shuffle_button = CanvasButton(
            panel, text="Shuffle: ON", command=self._on_toggle_shuffle, width=130, height=32
        )
        self.shuffle_button.grid(row=2, column=3, **pad)

        tk.Label(panel, text="Volume", bg=PANEL_BG, fg=TEXT_FG).grid(row=3, column=0, sticky="w", **pad)
        self.volume_scale = tk.Scale(
            panel, from_=0, to=100, orient="horizontal", command=self._on_volume,
            bg=PANEL_BG, fg=TEXT_FG, highlightthickness=0, troughcolor=TROUGH_COLOR,
        )
        self.volume_scale.set(100)
        self.volume_scale.grid(row=3, column=1, columnspan=4, sticky="we", **pad)

        tk.Label(panel, text="Crossfade (s)", bg=PANEL_BG, fg=TEXT_FG).grid(row=4, column=0, sticky="w", **pad)
        self.crossfade_scale = tk.Scale(
            panel, from_=MIN_CROSSFADE_SECONDS, to=MAX_CROSSFADE_SECONDS, resolution=0.5,
            orient="horizontal", command=self._on_crossfade,
            bg=PANEL_BG, fg=TEXT_FG, highlightthickness=0, troughcolor=TROUGH_COLOR,
        )
        self.crossfade_scale.set(DEFAULT_CROSSFADE_SECONDS)
        self.crossfade_scale.grid(row=4, column=1, columnspan=4, sticky="we", **pad)

        tk.Label(panel, text="Now playing:", bg=PANEL_BG, fg=TEXT_FG).grid(row=5, column=0, sticky="w", **pad)
        self.now_playing_label = tk.Label(
            panel, text="-", anchor="w", wraplength=560, bg=PANEL_BG, fg=ACCENT_FG
        )
        self.now_playing_label.grid(row=5, column=1, columnspan=4, sticky="w", **pad)

        tk.Label(panel, text="Up next:", bg=PANEL_BG, fg=TEXT_FG).grid(row=6, column=0, sticky="w", **pad)
        self.up_next_label = tk.Label(
            panel, text="-", anchor="w", wraplength=560, bg=PANEL_BG, fg=TEXT_FG
        )
        self.up_next_label.grid(row=6, column=1, columnspan=4, sticky="w", **pad)

        self.status_label = tk.Label(panel, text="Idle", anchor="w", bg=PANEL_BG, fg=DIM_FG)
        self.status_label.grid(row=7, column=0, columnspan=5, sticky="we", **pad)

        self.timeline_toggle_canvas = tk.Canvas(
            panel, width=64, height=24, bg=PANEL_BG, highlightthickness=0, cursor="hand2"
        )
        self.timeline_toggle_oval = self.timeline_toggle_canvas.create_oval(
            1, 1, 63, 23, fill=TIMELINE_TOGGLE_BG, outline=""
        )
        self.timeline_toggle_text = self.timeline_toggle_canvas.create_text(
            32, 12, text="Reveal", fill=BUTTON_FG, font=("Segoe UI", 8)
        )
        self.timeline_toggle_canvas.tag_bind(self.timeline_toggle_oval, "<Button-1>", self._on_toggle_timeline)
        self.timeline_toggle_canvas.tag_bind(self.timeline_toggle_text, "<Button-1>", self._on_toggle_timeline)
        self.timeline_toggle_canvas.bind("<Enter>", lambda e: self.timeline_toggle_canvas.itemconfig(self.timeline_toggle_oval, fill=TIMELINE_TOGGLE_ACTIVE_BG))
        self.timeline_toggle_canvas.bind("<Leave>", lambda e: self.timeline_toggle_canvas.itemconfig(self.timeline_toggle_oval, fill=TIMELINE_TOGGLE_BG))
        self.timeline_toggle_canvas.grid(row=8, column=0, columnspan=5, pady=(2, 0))

        self.timeline_row = tk.Frame(panel, bg=PANEL_BG)
        self.timeline_scale = tk.Scale(
            self.timeline_row, from_=0, to=1, orient="horizontal", showvalue=0,
            bg=PANEL_BG, fg=TEXT_FG, highlightthickness=0, troughcolor=TROUGH_COLOR,
        )
        self.timeline_scale.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.timeline_scale.bind("<Button-1>", self._on_timeline_press)
        self.timeline_scale.bind("<ButtonRelease-1>", self._on_timeline_release)

        self.timeline_remaining_label = tk.Label(
            self.timeline_row, text="-0:00", width=6, anchor="e", bg=PANEL_BG, fg=TEXT_FG
        )
        self.timeline_remaining_label.pack(side="left")

    # -- backgrounds ----------------------------------------------------------

    def _load_default_backgrounds(self):
        if DEFAULT_BACKGROUNDS_DIR.is_dir():
            self.backgrounds.set_folder(DEFAULT_BACKGROUNDS_DIR)
            if self.backgrounds.has_images():
                self._show_first_background()

    def _show_first_background(self):
        self._animating = False
        img = self.backgrounds.next_image()
        if img is None:
            return
        self._current_bg_raw = img
        self._render_static_background()

    def _render_static_background(self):
        if self._current_bg_raw is None:
            return
        w, h = self._canvas_size
        resized = _cover_resize(self._current_bg_raw, w, h)
        self._set_canvas_image(resized)

    def _set_canvas_image(self, pil_img):
        self._bg_photo = ImageTk.PhotoImage(pil_img)
        self.canvas.itemconfig(self.bg_image_id, image=self._bg_photo)

    def _on_canvas_resize(self, event):
        self._canvas_size = (event.width, event.height)
        self.canvas.coords(self.panel_window, event.width / 2, event.height - 6)
        if not self._animating:
            self._render_static_background()

    def _start_background_crossfade(self, duration):
        next_img = self.backgrounds.next_image()
        if next_img is None:
            return
        if self._current_bg_raw is None or duration <= 0:
            self._current_bg_raw = next_img
            self._render_static_background()
            self._animating = False
            return

        w, h = self._canvas_size
        self._anim_from_resized = _cover_resize(self._current_bg_raw, w, h)
        self._anim_to_resized = _cover_resize(next_img, w, h)
        self._anim_to_raw = next_img
        self._anim_start_time = time.time()
        self._anim_duration = duration
        self._animating = True

    def _advance_background_animation(self):
        elapsed = time.time() - self._anim_start_time
        alpha = min(1.0, max(0.0, elapsed / self._anim_duration))
        blended = Image.blend(self._anim_from_resized, self._anim_to_resized, alpha)
        self._set_canvas_image(blended)
        if alpha >= 1.0:
            self._animating = False
            self._current_bg_raw = self._anim_to_raw

    # -- music folder / transport ----------------------------------------------

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder of audio files")
        if not folder:
            return
        try:
            self.player.set_folder(folder)
        except ValueError as exc:
            messagebox.showerror("No audio files", str(exc))
            return
        self.folder = folder
        self.folder_label.config(text=Path(folder).name or folder)
        if self.player.is_running():
            self.player.stop()
            self.player.start()

    def _on_toggle_shuffle(self):
        enabled = not self.player.shuffle_enabled
        self.player.set_shuffle(enabled)
        self.shuffle_button.set_text("Shuffle: ON" if enabled else "Shuffle: OFF")
        if self.player.is_running():
            self.player.stop()
            self.player.start()
            self.play_button.set_text("Pause")

    def _on_play_pause(self):
        if not self.folder:
            messagebox.showinfo("Choose a folder", "Pick a folder of audio files first.")
            return
        if not self.player.is_running():
            try:
                self.player.start()
            except ValueError as exc:
                messagebox.showerror("Error", str(exc))
                return
            self.play_button.set_text("Pause")
        else:
            self.player.toggle_pause()
            self.play_button.set_text("Play" if self.player.is_paused() else "Pause")

    def _on_stop(self):
        self.player.stop()
        self.play_button.set_text("Play")
        self.now_playing_label.config(text="-")
        self.up_next_label.config(text="-")

    def _on_skip(self):
        self.player.skip()

    def _on_toggle_timeline(self, event=None):
        self.timeline_visible = not self.timeline_visible
        self._apply_timeline_visibility()
        self._settings["timeline_visible"] = self.timeline_visible
        _save_settings(self._settings)

    def _apply_timeline_visibility(self):
        pad = {"padx": 6, "pady": 4}
        if self.timeline_visible:
            self.timeline_row.grid(row=9, column=0, columnspan=5, sticky="we", **pad)
            self.timeline_toggle_canvas.itemconfig(self.timeline_toggle_text, text="Hide")
        else:
            self.timeline_row.grid_remove()
            self.timeline_toggle_canvas.itemconfig(self.timeline_toggle_text, text="Reveal")

    def _on_timeline_press(self, event):
        self._timeline_dragging = True

    def _on_timeline_release(self, event):
        self._timeline_dragging = False
        self.player.seek(self.timeline_scale.get())

    def _on_volume(self, value):
        self.player.volume = int(value) / 100.0

    def _on_crossfade(self, value):
        self.player.set_crossfade_seconds(float(value))

    # -- main loop --------------------------------------------------------------

    def _tick(self):
        self.status_label.config(text=self.player.status)
        self.now_playing_label.config(text=self.player.now_playing or "-")
        self.up_next_label.config(text=self.player.up_next or "-")
        if self.player.error:
            self.status_label.config(text=f"{self.player.status} - {self.player.error}")
        if not self.player.is_running() and self.play_button.get_text() == "Pause":
            self.play_button.set_text("Play")

        duration = self.player.duration_seconds
        position = self.player.position_seconds
        if self.timeline_visible and not self._timeline_dragging:
            self.timeline_scale.config(to=duration if duration > 0 else 1)
            self.timeline_scale.set(position)
            self.timeline_remaining_label.config(text=_format_time_remaining(duration - position))

        if self.backgrounds.has_images():
            event_id = self.player.crossfade_event_id
            if event_id != self._last_seen_crossfade_id:
                self._last_seen_crossfade_id = event_id
                self._start_background_crossfade(self.player.crossfade_event_duration)

        if self._animating:
            self._advance_background_animation()

        self.root.after(ANIM_INTERVAL_MS, self._tick)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
