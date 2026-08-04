"""
fish_analyzer/gui/inspector_export.py
=====================================
Export controls for the Video Inspector: In/Out range markers, Save Frame and
Export Clip.

Split from inspector_tab.py, which was already the largest module in the repo
before this feature existed. The tab keeps rendering and navigation; this mixin
keeps everything to do with getting pixels out of the application.

All the real work lives in fish_analyzer/overlay_render.py and
fish_analyzer/media_export.py - neither of which imports tkinter. This file is
the wiring between them and the widgets.
"""
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog

from ..overlay_render import compose_frame, fish_colors


class InspectorExportMixin:
    """Mixin providing the Video Inspector's export controls.

    Expects from the host class:
        - self.loaded_files, self.bout_results, self.video_readers
        - self.inspector_* control variables
        - self._insp_fig, self._insp_ax_time, self._insp_cached_background
        - self.render_settings_from_vars(), self._get_inspector_frame_idx(),
          self._inspector_stop_playback(), self._inspector_update_fast()
    """

    # =========================================================================
    # EXPORT RANGE
    # =========================================================================

    def _inspector_export_range(self, n_frames):
        """The marked range, normalised and clamped, inclusive of both ends.

        An unset marker means "from the start" or "to the end" rather than
        frame 0, so marking only one end still gives a usable range.
        """
        start = self.inspector_mark_in
        end = self.inspector_mark_out
        if start is None:
            start = 0
        if end is None:
            end = n_frames - 1
        if start > end:
            start, end = end, start
        return max(0, start), min(n_frames - 1, end)

    def _inspector_set_mark_in(self):
        self.inspector_mark_in = self._get_inspector_frame_idx()
        self._inspector_update_mark_label()

    def _inspector_set_mark_out(self):
        self.inspector_mark_out = self._get_inspector_frame_idx()
        self._inspector_update_mark_label()

    def _inspector_clear_marks(self):
        self.inspector_mark_in = None
        self.inspector_mark_out = None
        self._inspector_update_mark_label()

    def _inspector_update_mark_label(self):
        """Show the marked range in frames and seconds."""
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            self.inspector_mark_label.config(text="In -- | Out --")
            return

        loaded = self.loaded_files[selected]
        fps = loaded.calibration.frame_rate
        start, end = self._inspector_export_range(loaded.n_frames)
        n = end - start + 1
        marked = (self.inspector_mark_in is not None
                  or self.inspector_mark_out is not None)
        prefix = "" if marked else "(whole recording) "
        self.inspector_mark_label.config(
            text=f"{prefix}In {start} → Out {end} "
                 f"({n} frames, {n / fps:.1f} s)"
        )

    # =========================================================================
    # EXPORT
    # =========================================================================

    def _inspector_current_composite(self):
        """The current frame with overlays, at full video resolution.

        Not the canvas image: that is downscaled to fit the widget, which is
        no use for a figure.

        Returns (rgb_array, loaded, frame_idx), or (None, None, None) when no
        file is selected.
        """
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return None, None, None

        loaded = self.loaded_files[selected]
        frame_idx = min(self._get_inspector_frame_idx(), loaded.n_frames - 1)

        base = None
        if self.inspector_video_var.get() and selected in self.video_readers:
            base = self.video_readers[selected].read_frame(frame_idx)
        if base is None and self._insp_cached_background is not None:
            base = self._insp_cached_background
        if base is None:
            base = np.ones((loaded.metadata.video_height,
                            loaded.metadata.video_width, 3),
                           dtype=np.uint8) * 200

        composed = compose_frame(base, loaded.trajectories, frame_idx,
                                 self.render_settings_from_vars(),
                                 loaded.calibration.scale_factor)
        return composed, loaded, frame_idx

    def _inspector_save_frame(self):
        """Write the current composite to a PNG at full resolution."""
        composed, loaded, frame_idx = self._inspector_current_composite()
        if composed is None:
            messagebox.showwarning(
                "No File Selected",
                "Select a file in the Video Inspector first."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save Frame",
            defaultextension=".png",
            initialfile=f"{loaded.nickname}_frame{frame_idx:06d}.png",
            filetypes=[("PNG image", "*.png")]
        )
        if not path:
            return

        try:
            from ..media_export import save_frame_png
            save_frame_png(composed, path)
        except Exception as e:
            messagebox.showerror("Save Failed",
                                 f"Could not write {path}:\n{e}")
            return

        self.set_status(f"Saved frame to {Path(path).name}")
        messagebox.showinfo(
            "Frame Saved",
            f"Saved to:\n{path}\n\n"
            f"{composed.shape[1]} x {composed.shape[0]} px"
        )

    def _inspector_can_export(self):
        """(ok, reason) - whether an export can produce a meaningful clip.

        The point is to refuse before writing anything, rather than encode a
        panel reading "Run Shoaling Analysis first" into a video.
        """
        selected = self.inspector_file_var.get()
        if not selected or selected not in self.loaded_files:
            return False, "Select a file in the Video Inspector first."

        loaded = self.loaded_files[selected]
        time_mode = self.inspector_time_mode_var.get()

        if time_mode in ('nnd', 'iid', 'hull') and not loaded.shoaling_results:
            return False, (
                f"The Time Panel is set to {time_mode.upper()}, but no "
                f"shoaling results exist for '{selected}'.\n\n"
                "Run Shoaling Analysis first, or set the Time Panel to 'None'."
            )

        if time_mode == 'bout' and not self.bout_results.get(selected):
            return False, (
                "The Time Panel is set to Bout Speed + Heading, but no bout "
                f"results exist for '{selected}'.\n\n"
                "Run Bout Analysis first, or set the Time Panel to 'None'."
            )

        return True, ""

    def _inspector_export_clip_dialog(self):
        """Collect export options, then run the export."""
        ok, reason = self._inspector_can_export()
        if not ok:
            messagebox.showwarning("Cannot Export", reason)
            return

        selected = self.inspector_file_var.get()
        loaded = self.loaded_files[selected]
        start, end = self._inspector_export_range(loaded.n_frames)
        fps = loaded.calibration.frame_rate
        n_frames = end - start + 1

        time_mode = self.inspector_time_mode_var.get()
        # Measured at 28 ms/frame exporting 1288x964 + strip with positions,
        # NND, hull and trails on. Bout redraws matplotlib every frame instead
        # of stamping a cursor onto a strip rasterised once, so it is roughly
        # three times slower.
        per_frame_ms = 90 if time_mode == 'bout' else 28
        estimate_s = n_frames * per_frame_ms / 1000.0

        marked = (self.inspector_mark_in is not None
                  or self.inspector_mark_out is not None)
        message = (
            f"Export frames {start}-{end}"
            f"{'' if marked else ' (whole recording - no In/Out marked)'}\n"
            f"{n_frames} frames, {n_frames / fps:.1f} s of video\n"
            f"Estimated time: {estimate_s:.0f} s\n\n"
        )
        if time_mode == 'bout':
            message += ("The Bout panel has to be redrawn for every frame, "
                        "which is around four times slower than the other "
                        "panels.\n\n")
        message += "Continue?"

        if not messagebox.askyesno("Export Clip", message):
            return

        as_png = messagebox.askyesno(
            "Output Format",
            "Yes  =  PNG sequence (lossless, one file per frame)\n"
            "No   =  MP4 video"
        )

        if as_png:
            target = filedialog.askdirectory(
                title="Choose a folder for the PNG sequence"
            )
        else:
            target = filedialog.asksaveasfilename(
                title="Save Clip",
                defaultextension=".mp4",
                initialfile=f"{selected}_{start}-{end}.mp4",
                filetypes=[("MP4 video", "*.mp4")]
            )
        if not target:
            return

        self._inspector_run_export(loaded, selected, start, end, target,
                                   as_png)

    @staticmethod
    def _inspector_time_scale_for(time_mode):
        """Data units per second on the time panel's x axis.

        The NND, IID and Hull panels are plotted against minutes
        (_inspector_rebuild_figure divides timestamps by 60, and the live
        cursor does the same), while the export loop counts seconds. The Bout
        panel's window is already in seconds.

        A wrong value here does not raise: the cursor is clamped to an edge and
        silently never moves.
        """
        return 1.0 if time_mode == 'bout' else 1.0 / 60.0

    def _inspector_run_export(self, loaded, selected, start, end, target,
                              as_png):
        """Run the export in chunks so Tk keeps painting and Cancel works.

        Not a worker thread: neither Tk nor matplotlib is safe to drive off
        the main thread.
        """
        from ..media_export import (CodecUnavailable, ExportFrameSource,
                                    Mp4Sink, PngSequenceSink, ScrollingStrip,
                                    TimeStrip)

        self._inspector_stop_playback()

        fps = loaded.calibration.frame_rate
        settings = self.render_settings_from_vars()
        time_mode = self.inspector_time_mode_var.get()

        video_path = None
        if self.inspector_video_var.get() and selected in self.video_readers:
            video_path = self.video_readers[selected].video_path
        if video_path is None:
            messagebox.showwarning(
                "No Video",
                "Clip export needs a video.\n\n"
                "Tick 'Use video frames' and load one with Browse Video..."
            )
            return

        frame_h = loaded.metadata.video_height
        frame_w = loaded.metadata.video_width

        strip = None
        if time_mode != 'none' and self._insp_fig is not None:
            if time_mode == 'bout':
                # Scrolling axes: cannot be rasterised once.
                strip = ScrollingStrip(
                    self._insp_fig, self._insp_ax_time,
                    window_s=float(self.inspector_bout_window_var.get()),
                    total_s=loaded.n_frames / fps,
                    target_width=frame_w,
                )
            else:
                strip = TimeStrip(
                    self._insp_fig, self._insp_ax_time,
                    target_width=frame_w,
                    time_scale=self._inspector_time_scale_for(time_mode),
                )

        out_h = frame_h + (strip.height if strip is not None else 0)

        try:
            source = ExportFrameSource(video_path, start_frame=start)
            if as_png:
                sink = PngSequenceSink(target, fps, (frame_w, out_h))
            else:
                sink = Mp4Sink(target, fps, (frame_w, out_h))
        except CodecUnavailable as e:
            messagebox.showerror("Export Failed", str(e))
            return
        except Exception as e:
            messagebox.showerror("Export Failed",
                                 f"Could not start the export:\n{e}")
            return

        progress = tk.Toplevel(self.root)
        progress.title("Exporting")
        progress.transient(self.root)
        progress.grab_set()
        label = tk.Label(progress, text="Starting...", width=32)
        label.pack(padx=20, pady=10)
        state = {"cancel": False}
        tk.Button(progress, text="Cancel",
                  command=lambda: state.__setitem__("cancel", True)
                  ).pack(pady=(0, 10))

        total = end - start + 1
        colors = fish_colors(loaded.n_fish)
        cursor = {"frame": start, "written": 0}
        out_buffer = {"array": None}

        def finish(message, warn):
            try:
                progress.destroy()
            except Exception:
                pass
            # Leaving this set means a later after_cancel could target a
            # scheduled callback that has already run.
            self._insp_export_after_id = None
            source.close()
            # ScrollingStrip leaves the live axes on the last exported window,
            # and both strips force a draw on the shared canvas. Rebuild so
            # the tab is not left showing export state.
            self._insp_needs_rebuild = True
            try:
                self._inspector_update_fast()
            except Exception:
                pass
            self.set_status(message)
            if warn:
                messagebox.showwarning("Export Stopped", message)
            else:
                messagebox.showinfo("Export Complete", message)

        def do_chunk():
            if state["cancel"]:
                sink.discard_partial()
                finish(f"Export cancelled after {cursor['written']} frames.",
                       True)
                return

            for _ in range(10):
                if cursor["frame"] > end:
                    sink.close()
                    finish(f"Wrote {cursor['written']} frames to "
                           f"{getattr(sink, 'path', target)}", False)
                    return

                base = source.read()
                if base is None:
                    sink.close()
                    finish(f"The video ended early; wrote "
                           f"{cursor['written']} of {total} frames.", True)
                    return

                composed = compose_frame(base, loaded.trajectories,
                                         cursor["frame"], settings,
                                         loaded.calibration.scale_factor,
                                         colors)

                if strip is None:
                    frame_out = composed
                else:
                    if out_buffer["array"] is None:
                        out_buffer["array"] = np.zeros(
                            (out_h, frame_w, 3), dtype=np.uint8
                        )
                    out_buffer["array"][:frame_h] = composed
                    out_buffer["array"][frame_h:] = strip.at(
                        cursor["frame"] / fps
                    )
                    frame_out = out_buffer["array"]

                try:
                    sink.write(frame_out)
                except Exception as e:
                    sink.close()
                    finish(f"Write failed at frame {cursor['frame']}: {e}",
                           True)
                    return

                cursor["written"] += 1
                cursor["frame"] += 1

            label.config(text=f"Frame {cursor['written']} / {total}")
            # Not animation_after_id: that belongs to playback, and reusing it
            # would let a stop_playback cancel the export or vice versa.
            self._insp_export_after_id = self.root.after(1, do_chunk)

        self.root.after(1, do_chunk)
